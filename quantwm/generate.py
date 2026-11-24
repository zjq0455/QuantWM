"""Single-sample CLI. Each invocation generates one video for one method."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODELS={'matrix_game2':'Matrix-Game-2','lingbot_v2':'LingBot-v2',
        'hy_worldplay':'HY-WorldPlay','self_forcing':'Self-Forcing','longcat':'LongCat'}
WEIGHT_VARIABLES={
    'matrix_game2':('MATRIX_MODEL_PATH',),
    'lingbot_v2':('LINGBOT_MODEL_PATH',),
    'hy_worldplay':('HY_MODEL_ID','HY_AR_MODEL_PATH','HY_CHECKPOINT_PATH'),
    'self_forcing':('SELF_FORCING_MODEL_PATH','SELF_FORCING_CHECKPOINT'),
    'longcat':('LONGCAT_MODEL_PATH',),
}

def configure_cuda_linker(env):
    """Use available CUDA stubs without requiring a machine-specific install."""
    homes=[]
    for key in ('CUDA_HOME','CUDA_PATH'):
        if env.get(key):
            homes.append(Path(env[key]))
    nvcc=shutil.which('nvcc',path=env.get('PATH'))
    if nvcc:
        homes.append(Path(nvcc).resolve().parents[1])
    homes.append(Path('/usr/local/cuda'))
    existing=[s for s in env.get('LIBRARY_PATH','').split(os.pathsep) if s]
    candidates=[home/suffix for home in homes for suffix in
                ('targets/x86_64-linux/lib/stubs','lib64/stubs')]
    stubs=list(dict.fromkeys(str(p) for p in candidates if p.is_dir()))
    if stubs:
        env['LIBRARY_PATH']=os.pathsep.join(dict.fromkeys(stubs+existing))

def run(command,cwd,env,log):
    print(shlex.join(map(str,command)),flush=True)
    with log.open('a') as stream:
        stream.write('$ '+shlex.join(map(str,command))+'\n')
        stream.flush()
        subprocess.run(list(map(str,command)),cwd=cwd,env=env,stdout=stream,
                       stderr=subprocess.STDOUT,check=True)

def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_video(path, expected_frames):
    """Container sanity only; no frame similarity or video quality evaluation."""
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_frames','-of','json',str(path)],
        capture_output=True,text=True,check=True)
    streams=json.loads(result.stdout).get('streams',[])
    if len(streams)!=1 or int(streams[0].get('nb_frames',0))!=expected_frames:
        raise RuntimeError(f'Incomplete video, expected {expected_frames} frames: {path}')
    return streams[0]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--method',choices=('bf16','qvg','quantwm'),required=True)
    parser.add_argument('--gpu',default='0',help='GPU ID, or comma-separated IDs for LingBot/LongCat')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prompt-file',type=Path,default=ROOT/'examples/prompt.txt')
    parser.add_argument('--image',type=Path)
    parser.add_argument('--init-video',type=Path)
    parser.add_argument('--action-path',type=Path,default=ROOT/'examples/lingbot_actions')
    parser.add_argument('--action',default='turn_left:45:2,idle:2,turn_right:45:2')
    parser.add_argument('--pose',default='w-8,s-8,a-8,d-8,up-8,down-8,left-8')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--segments',type=int,default=3,help='LongCat extension segments')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    missing=[name for name in WEIGHT_VARIABLES[args.model] if not os.environ.get(name)]
    if missing:
        parser.error('Missing model configuration: '+', '.join(missing)+
            '. Copy configs/paths.env.example to configs/paths.env and set your checkpoint paths, '
            'or export these variables before running.')
    lines=[s.strip() for s in args.prompt_file.read_text().splitlines() if s.strip()]
    if len(lines)!=1:
        raise ValueError('Exactly one prompt is required')
    prompt=lines[0]
    out=(args.output or ROOT/'outputs'/args.model/args.method).resolve()
    native=ROOT/'vendor/models'/MODELS[args.model]
    image=args.image or ROOT/'examples'/('matrix_scene.png' if args.model=='matrix_game2' else 'robot_scene.png')
    image=image.resolve()
    nproc=len(args.gpu.split(','))
    if nproc>1 and args.model not in {'lingbot_v2','longcat'}:
        raise ValueError('This adapter supports a single GPU')
    if args.segments<1:
        raise ValueError('LongCat must contain at least one continuation segment')
    expected_frames=93+20*args.segments if args.model=='longcat' else 93
    if args.model=='longcat' and args.init_video:
        initial_info=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames','-of','json',str(args.init_video)],
            capture_output=True,text=True,check=True)
        expected_frames=int(json.loads(initial_info.stdout)['streams'][0]['nb_frames'])+20*args.segments
    env=os.environ.copy()
    # Prevent stale experimental switches in the parent shell from changing methods.
    for key in tuple(env):
        if key.startswith(('KV_TRICK_','KV_META_','QLR_HISTORICAL_','QUANTWM_')):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,QUANTWM_METHOD=args.method,
        QUANTWM_RUNTIME_REPORT=str(out/'runtime.json'),QUANTWM_NATIVE_ROOT=str(native),
        PYTHONPATH=os.pathsep.join(map(str,[native,ROOT,ROOT/'vendor'])),
        KV_META_SCALE_ZERO_FORMAT='fp8',KV_META_CENTROID_FORMAT='native',
        KV_TRICK_QLR_COEFF_FORMAT='int8',QVG_QUANT_FACTOR='2',
        TOKENIZERS_PARALLELISM='false')
    configure_cuda_linker(env)
    py=os.environ.get('QUANTWM_PYTHON',sys.executable)
    launch=[py,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={nproc}']
    quant=['--quant_type','none' if args.method=='bf16' else 'nstages-kmeans-int2']
    if args.method!='bf16':
        quant+=['--cache_num_k_centroids','256','--cache_num_v_centroids','256',
                '--kmeans_max_iters','2','--quant_block_size','64','--num_prq_stages','1']
    paths=[]
    metadata={'model':args.model,'method':args.method,'seed':args.seed,'gpu':args.gpu,
              'prompt':prompt,'image_sha256':file_sha(image) if args.model in {'matrix_game2','lingbot_v2','hy_worldplay'} else None,
              'action':args.action if args.model=='matrix_game2' else args.pose if args.model=='hy_worldplay' else None,
              'segments':args.segments if args.model=='longcat' else None,
              'rank':8 if args.method=='quantwm' else 0,
              'query_subspace':'Q16/eigh' if args.method=='quantwm' else None,
              'self_forcing_quant_factor':2 if args.model=='self_forcing' else None}

    if args.model=='matrix_game2':
        model=Path(env['MATRIX_MODEL_PATH'])
        paths=[model/'base_distilled_model/base_distill.safetensors',model/'Wan2.1_VAE.pth',image]
        command=[py,'-m','quantwm.adapters.matrix','--config_path',out/'matrix_config.yaml',
            '--checkpoint_path',paths[0],'--pretrained_model_path',model,'--img_path',image,
            '--output_folder',out/'native','--num_output_frames','24','--seed',args.seed,
            '--action_mode','custom_sequence','--custom_action',args.action,
            '--action_sequence',args.action,'--turn_degrees','0']
        result=out/'native/demo.mp4'
    elif args.model=='lingbot_v2':
        paths=[Path(env['LINGBOT_MODEL_PATH']),image,args.action_path/'poses.npy',args.action_path/'intrinsics.npy']
        metadata['action_poses_sha256']=file_sha(paths[2])
        metadata['action_intrinsics_sha256']=file_sha(paths[3])
        command=launch+[native/'generate.py','--task','i2v-A14B','--infer_mode','causal_fast',
            '--size','832*480','--chunk_size','4','--ckpt_dir',paths[0],
            '--image',image,'--action_path',args.action_path,'--frame_num','93',
            '--prompt',prompt,'--base_seed',args.seed,'--save_file',out/'native/video.mp4',
            '--save_dir',out/'native','--local_attn_size','18','--sink_size','6',
            '--t5_cpu','--use_chunked_kv','--offload_model','true','--video_crf','18',*quant]
        if nproc>1:
            command+=['--ulysses_size',nproc]
        if args.method!='bf16':
            command+=['--quant_factor','1','--quant_keep_recent_chunks','0','--quant_sink_keep_chunks','0']
        result=out/'native/video.mp4'
    elif args.model=='hy_worldplay':
        env['PYTHONPATH']+=os.pathsep+str(native/'wan')
        paths=[Path(env['HY_MODEL_ID']),Path(env['HY_AR_MODEL_PATH']),Path(env['HY_CHECKPOINT_PATH']),image]
        command=launch+['-m','quantwm.adapters.hy','--input',out/'prompt_image_pair.txt',
            '--num_frames','93','--num_chunk','6','--pose',args.pose,'--seed',args.seed,
            '--model_id',paths[0],'--ar_model_path',paths[1],'--ckpt_path',paths[2],
            '--offload_text_encoder','--out',out/'native','--memory_frames','24',
            '--temporal_context_size','20','--pred_latent_size','4',*quant]
        result=out/f'native/0-{args.seed}.mp4'
    elif args.model=='self_forcing':
        paths=[Path(env['SELF_FORCING_MODEL_PATH']),Path(env['SELF_FORCING_CHECKPOINT'])]
        command=launch+['-m','quantwm.adapters.self_forcing','--config_path',native/'configs/self_forcing_dmd.yaml',
            '--checkpoint_path',paths[1],'--data_path',out/'prompt.txt','--output_folder',out/'native',
            '--num_samples','1','--num_output_frames','24','--local_attn_size','24',
            '--save_with_index','--seed',args.seed,*quant]
        result=out/'native/0-0_regular.mp4'
    else:
        paths=[Path(env['LONGCAT_MODEL_PATH']),Path(env['LONGCAT_MODEL_PATH'])/'lora/cfg_step_lora.safetensors']
        command=launch+[ROOT/'quantwm/longcat_generate.py','--checkpoint-dir',paths[0],
            '--prompt',prompt,'--seed',args.seed,'--segments',args.segments,
            '--output',out/'native/video.mp4','--initial-output',out/'native/initial.mp4',
            '--context-parallel-size',nproc]
        if args.init_video:
            paths.append(args.init_video)
            metadata['init_video_sha256']=file_sha(args.init_video)
            command+=['--init-video',args.init_video]
        result=out/'native/video.mp4'
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.dry_run:
        print(json.dumps(metadata,indent=2,ensure_ascii=False))
        print(shlex.join(map(str,command)))
        return
    out.mkdir(parents=True,exist_ok=True)
    (out/'native').mkdir(exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        # Device migration is allowed; all semantic settings must match.
        old.pop('gpu',None)
        same=dict(metadata);same.pop('gpu',None)
        if old!=same:
            raise RuntimeError(f'Different inputs already exist in {out}; select a new --output')
    manifest.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    (out/'prompt.txt').write_text(prompt+'\n')
    (out/'prompt_image_pair.txt').write_text(prompt+'@'+str(image)+'\n')
    if args.model=='matrix_game2':
        config=(native/'configs/inference_yaml/inference_universal.yaml').read_text()
        config=config.replace('configs/distilled_model/universal',str(native/'configs/distilled_model/universal'))
        (out/'matrix_config.yaml').write_text(config)
    destination=out/'video.mp4'
    if destination.exists() and (out/'runtime.json').exists():
        validate_video(destination,expected_frames)
        report=json.loads((out/'runtime.json').read_text())
        if report['method']!=args.method or (args.method!='bf16' and report['quantized_layer_spans']<=0):
            raise RuntimeError('Existing output does not verify the requested quantization')
        print(destination)
        return
    try:
        run(command,native,env,out/'run.log')
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'Generation failed; inspect {out / "run.log"}') from error
    if not result.is_file():
        raise RuntimeError(f'Generator returned without a video: {result}. Check {out / "run.log"}')
    validate_video(result,expected_frames)
    report=json.loads((out/'runtime.json').read_text())
    if report['method']!=args.method or (args.method!='bf16' and report['quantized_layer_spans']<=0):
        raise RuntimeError('Quantized method did not actually quantize KV')
    # Preserve the native video; only the viewing copy is resized to 480p.
    run(['ffmpeg','-y','-v','error','-i',result,'-vf','scale=832:480',
        '-c:v','libx264','-crf','18','-preset','slow','-an',destination],
        native,env,out/'encode.log')
    validate_video(destination,expected_frames)
    print(f'Video: {destination}',flush=True)

if __name__=='__main__':
    main()
