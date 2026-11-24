import os
import runpy
from pathlib import Path
import torch
from quantwm.runtime import Runtime

def main():
    from pipeline.causal_inference import CausalInferencePipeline
    from wan.modules.causal_model import CausalWanSelfAttention
    runtime=Runtime(30)
    if runtime.method == 'bf16':
        CausalInferencePipeline.quantize_kv_cache=lambda self,*args: None
    recent={}
    if runtime.method == 'quantwm':
        original=CausalWanSelfAttention.attn_kv_cache_prerope
        def tracked(self,kv_cache,current_start,q,k,v,block_mask,grid_sizes,freqs,seq_lens,s):
            layer=runtime.tracker.call_index%30
            # Native Self-Forcing stores pre-RoPE K; preserve pair-invariant
            # energy while using the sampled pre-RoPE Q for QLR.
            paired=q.float().reshape(*q.shape[:-1],q.shape[-1]//2,2)
            surrogate=paired.square().mean(-1,keepdim=True).expand_as(paired).sqrt().reshape_as(q)
            runtime.tracker.observe(surrogate)
            indices=torch.linspace(0,q.shape[1]-1,min(16,q.shape[1]),device=q.device).round().long().unique()
            recent[layer]=q.index_select(1,indices).permute(0,2,1,3).float().detach()
            return original(self,kv_cache,current_start,q,k,v,block_mask,grid_sizes,freqs,seq_lens,s)
        CausalWanSelfAttention.attn_kv_cache_prerope=tracked
    if runtime.method != 'bf16':
        @torch.no_grad()
        def quantize(self,start,end,max_tokens):
            if end>max_tokens:
                raise ValueError('KV span exceeds written cache')
            pending=runtime.tracker.drain_pending()
            runtime.tracker.commit_deferred()
            for layer,cache in enumerate(self.kv_cache1):
                k=cache['k'].read(start,end).permute(0,2,1,3).contiguous()
                v=cache['v'].read(start,end).permute(0,2,1,3).contiguous()
                kq,vq=runtime.pair(k,v,config=self.quant_config,layer=layer,
                    span_start=start,tokens_per_frame=self.frame_seq_length,
                    queries=recent.get(layer))
                cache['k'].store_quantized(start,end,kq.permute(0,2,1,3).contiguous())
                cache['v'].store_quantized(start,end,vq.permute(0,2,1,3).contiguous())
            runtime.tracker.defer(pending)
            runtime.quantization_events+=1
        CausalInferencePipeline.quantize_kv_cache=quantize
    runpy.run_path(str(Path(os.environ['QUANTWM_NATIVE_ROOT'])/'inference.py'),run_name='__main__')

if __name__ == '__main__':
    main()
