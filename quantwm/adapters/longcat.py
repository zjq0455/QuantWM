import types
import torch
from quantwm.runtime import Runtime

def install(dit):
    from longcat_video.modules.utils import _offload_kv_cache,_onload_kv_cache
    runtime=Runtime(len(dit.blocks))
    if runtime.method == 'bf16':
        # The upstream "none" hook also performs cache memory reporting and
        # CPU transfers. This generation-only baseline needs neither.
        def untouched(self,cache,offload_kv_cache=False):
            return cache
        dit.quantize_kv_cache=types.MethodType(untouched,dit)
        return runtime
    if runtime.method == 'quantwm':
        for layer,block in enumerate(dit.blocks):
            attn=block.attn
            def capture(_module,_inputs,output,attn=attn,layer=layer):
                batch,tokens,_=output.shape
                qkv=output.view(batch,tokens,3,attn.num_heads,attn.head_dim).permute(2,0,3,1,4)
                q=attn.q_norm(qkv[0])
                expected=runtime.tracker.call_index%runtime.tracker.layer_count
                runtime.tracker.call_index+=(layer-expected)%runtime.tracker.layer_count
                runtime.tracker.observe(q.transpose(1,2).contiguous())
            attn.qkv.register_forward_hook(capture)

    @torch.no_grad()
    def quantize(self,cache,offload_kv_cache=False):
        # LongCat builds its condition cache before continuation begins.
        runtime.tracker.defer(runtime.tracker.drain_pending())
        runtime.tracker.commit_deferred()
        for layer,(k,v) in cache.items():
            if offload_kv_cache:
                k,v=_onload_kv_cache((k,v),'cuda')
            kq,vq=runtime.pair(k.contiguous(),v.contiguous(),config=self.quant_config,
                layer=layer,span_start=0,tokens_per_frame=1560)
            if offload_kv_cache:
                kq,vq=_offload_kv_cache((kq,vq))
            cache[layer]=(kq,vq)
        runtime.quantization_events+=1
        return cache
    dit.quantize_kv_cache=types.MethodType(quantize,dit)
    return runtime
