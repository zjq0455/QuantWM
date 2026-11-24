import types
import torch
from quantwm.runtime import Runtime

def install(pipeline):
    runtime = Runtime(len(pipeline.model.blocks))
    if runtime.method == 'bf16':
        return runtime
    import wan.modules.model_fast as model_fast
    if runtime.method == 'quantwm':
        model_fast.attention = runtime.tracker.wrap(model_fast.attention)

    @torch.no_grad()
    def quantize(self,caches,start,end,max_tokens):
        if end > max_tokens:
            raise ValueError('KV span exceeds written cache')
        pending = runtime.tracker.drain_pending()
        tpf = int(self._runtime_frame_seqlen)
        for layer,cache in enumerate(caches):
            k=cache['k'].read(start,end).permute(0,2,1,3).contiguous()
            v=cache['v'].read(start,end).permute(0,2,1,3).contiguous()
            kq,vq=runtime.pair(k,v,config=self.quant_config,layer=layer,
                span_start=start,tokens_per_frame=tpf)
            cache['k'].store_quantized(start,end,kq.permute(0,2,1,3).contiguous())
            cache['v'].store_quantized(start,end,vq.permute(0,2,1,3).contiguous())
        runtime.tracker.defer(pending)
        runtime.tracker.commit_deferred()
        runtime.quantization_events += 1
    pipeline._quantize_kv_cache=types.MethodType(quantize,pipeline)
    return runtime

