import os
import types
import torch
from quantwm.runtime import Runtime

def main():
    import inference as native
    from quant_videogen.sim.quant.quantize_config import QuantizeConfig

    class Inference(native.InteractiveGameInference):
        def _init_models(self):
            super()._init_models()
            runtime=Runtime(len(self.pipeline.generator.model.blocks))
            if runtime.method == 'bf16':
                return
            config=QuantizeConfig(quant_type='nstages-kmeans-int2',
                cache_num_k_centroids=256,cache_num_v_centroids=256,
                kmeans_max_iters=2,quant_block_size=64,num_prq_stages=1)
            import wan.modules.causal_model as causal
            if runtime.method == 'quantwm':
                causal.attention=runtime.tracker.wrap(causal.attention)
            model=self.pipeline.generator.model
            original=model.forward
            last_end=[None]
            tpf=self.pipeline.frame_seq_length

            @torch.no_grad()
            def wrapped(_model,*args,**kwargs):
                result=original(*args,**kwargs)
                caches=kwargs.get('kv_cache')
                if not caches:
                    return result
                latent=args[0] if args else kwargs.get('x')
                new_tokens=int(latent.shape[2])*tpf
                pending=runtime.tracker.drain_pending()
                end=max(int(c['global_end_index'].item()) for c in caches)
                if last_end[0] is not None and end>last_end[0]:
                    runtime.tracker.commit_deferred()
                last_end[0]=end
                for layer,cache in enumerate(caches):
                    local_end=int(cache['local_end_index'].item())
                    start=max(0,local_end-new_tokens)
                    if start==local_end:
                        continue
                    k=cache['k'][:,start:local_end].permute(0,2,1,3).contiguous()
                    v=cache['v'][:,start:local_end].permute(0,2,1,3).contiguous()
                    kq,vq=runtime.pair(k,v,config=config,layer=layer,
                        span_start=start,tokens_per_frame=tpf)
                    cache['k'][:,start:local_end].copy_(kq.permute(0,2,1,3))
                    cache['v'][:,start:local_end].copy_(vq.permute(0,2,1,3))
                runtime.tracker.defer(pending)
                runtime.quantization_events+=1
                return result
            model.forward=types.MethodType(wrapped,model)

    args=native.parse_args()
    native.set_seed(args.seed)
    os.makedirs(args.output_folder,exist_ok=True)
    Inference(args).generate_videos()

if __name__ == '__main__':
    main()

