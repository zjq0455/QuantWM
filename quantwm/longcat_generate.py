"""LongCat 16-step distilled initial generation and cached continuation."""
import argparse
import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torchvision.io import read_video,write_video
from transformers import AutoTokenizer,UMT5EncoderModel
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel.context_parallel_util import init_context_parallel
from longcat_video.context_parallel import context_parallel_util
from quant_videogen.sim.quant.quantize_config import QuantizeConfig
from quantwm.adapters.longcat import install

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--segments',type=int,default=3)
    p.add_argument('--context-parallel-size',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial-output',type=Path,required=True)
    p.add_argument('--init-video',type=Path)
    a=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    init_context_parallel(context_parallel_size=a.context_parallel_size,
                          global_rank=rank,world_size=dist.get_world_size())
    cp=context_parallel_util.get_optimal_split(context_parallel_util.get_cp_size())
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    kwargs={'torch_dtype':torch.bfloat16}
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint_dir,subfolder='tokenizer')
    text=UMT5EncoderModel.from_pretrained(a.checkpoint_dir,subfolder='text_encoder',**kwargs)
    vae=AutoencoderKLWan.from_pretrained(a.checkpoint_dir,subfolder='vae',**kwargs)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(a.checkpoint_dir,subfolder='scheduler')
    dit=LongCatVideoTransformer3DModel.from_pretrained(a.checkpoint_dir,subfolder='dit',cp_split_hw=cp,**kwargs)
    pipe=LongCatVideoPipeline(tokenizer=tokenizer,text_encoder=text,vae=vae,scheduler=scheduler,dit=dit)
    pipe.to(local)
    dit.load_lora(str(Path(a.checkpoint_dir)/'lora/cfg_step_lora.safetensors'),'cfg_step_lora')
    dit.enable_loras(['cfg_step_lora'])
    with torch.no_grad():
        if a.init_video:
            initial=read_video(str(a.init_video),pts_unit='sec')[0].numpy()
        else:
            output=pipe.generate_t2v(prompt=a.prompt,height=480,width=832,num_frames=93,
                num_inference_steps=16,use_distill=True,guidance_scale=1.0)[0]
            initial=np.clip(np.array(output)*255,0,255).astype(np.uint8)
            if rank==0:
                write_video(str(a.initial_output),torch.from_numpy(initial),fps=15,
                            video_codec='libx264',options={'crf':'18'})
        if len(initial)<73 or initial.shape[1:3]!=(480,832):
            raise ValueError('LongCat initial clip requires >=73 frames at 832x480')
        # Install after the BF16 initial branch. All methods use the same
        # initial algorithm/seed; --init-video can enforce identical pixels.
        install(dit)
        dit.quant_config=QuantizeConfig(quant_type='none' if os.environ['QUANTWM_METHOD']=='bf16' else 'nstages-kmeans-int2',
            quant_block_size=64,num_prq_stages=1,cache_num_k_centroids=256,
            cache_num_v_centroids=256,kmeans_max_iters=2)
        current=[Image.fromarray(frame) for frame in initial]
        frames=list(initial)
        for segment in range(a.segments):
            # Each segment rebuilds its condition cache from the last 73 RGB
            # frames. Previous segment KV is no longer used, and retaining it
            # while constructing the next cache doubles the resident cache.
            pipe._clear_cache()
            os.environ['SEGMENT_IDX']=str(segment)
            output=pipe.generate_vc(video=current[-73:],prompt=a.prompt,resolution='480p',
                num_frames=93,num_cond_frames=73,num_inference_steps=16,use_distill=True,
                guidance_scale=1.0,use_kv_cache=True,offload_kv_cache=False,enhance_hf=False)[0]
            pixels=np.clip(np.array(output)*255,0,255).astype(np.uint8)
            frames.extend(pixels[73:])
            current=[Image.fromarray(frame) for frame in pixels]
            pipe._clear_cache()
            print(f'Completed segment {segment+1}/{a.segments}',flush=True)
        if rank==0:
            write_video(str(a.output),torch.from_numpy(np.stack(frames)),fps=15,
                        video_codec='libx264',options={'crf':'18'})
    dist.destroy_process_group()

if __name__=='__main__':
    main()
