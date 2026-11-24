import os
import runpy
from pathlib import Path
import torch
import torch.nn.functional as F
from quantwm.runtime import Runtime

def main():
    from wan.inference import pipeline_wan_w_mem_relative_rope as native
    runtime=Runtime(30)
    if runtime.method == 'bf16':
        native.WanPipeline.quantize_kv_cache=lambda self,*args: None
    if runtime.method == 'quantwm':
        original=F.scaled_dot_product_attention
        def tracked(query,key,value,*args,**kwargs):
            if query.ndim == 4 and query.shape[0]%2 == 0 and query.shape[-1] == 128:
                batch=query.shape[0]//2
                joined=torch.cat((query[:batch],query[batch:]),dim=-1)
                runtime.tracker.observe(joined.permute(0,2,1,3).contiguous())
            return original(query,key,value,*args,**kwargs)
        F.scaled_dot_product_attention=torch.compiler.disable(tracked)

    if runtime.method != 'bf16':
        @torch.no_grad()
        def quantize(self,start,end):
            pending=runtime.tracker.drain_pending()
            runtime.tracker.commit_deferred()
            for layer,cache in enumerate(self._kv_cache):
                k,v=cache['k'].read(start,end),cache['v'].read(start,end)
                kq,vq=runtime.pair(k,v,config=self._quant_config,layer=layer,
                    span_start=start,tokens_per_frame=880)
                cache['k'].store_quantized(start,end,kq)
                cache['v'].store_quantized(start,end,vq)
            runtime.tracker.defer(pending)
            runtime.quantization_events+=1
        native.WanPipeline.quantize_kv_cache=quantize
    runpy.run_path(str(Path(os.environ['QUANTWM_NATIVE_ROOT'])/'wan/generate.py'),run_name='__main__')

if __name__ == '__main__':
    main()
