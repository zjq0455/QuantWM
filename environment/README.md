# Inference environment

[English README](../README.md) | [中文说明](../README_zh-CN.md)

The reference environment uses Linux, Python 3.10, PyTorch 2.6.0 with CUDA 12.4,
torchvision 0.21.0, Triton 3.2.0, and NVIDIA A800 80GB GPUs. This is a reference
configuration, not a guarantee that every model fits a smaller GPU.

## Installation

Run from the repository root:

```bash
conda create -n quantwm python=3.10 -y
conda activate quantwm
conda install -c conda-forge ffmpeg -y

python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install sageattention==1.0.6 --no-build-isolation
```

FlashAttention is used by the video backends. The HY adapter additionally
uses SageAttention. Building CUDA extensions requires a compatible CUDA
toolkit, C++ compiler and PyTorch installation. Set `CUDA_HOME` if the toolkit
is outside the standard location. Adjust the PyTorch CUDA wheel to your
driver/toolkit when necessary; these commands describe the tested stack,
not installation on every CUDA version.

`ffmpeg` and `ffprobe` must both be on `PATH`. Exported videos use software
H.264 encoding; NVENC is not required. No VBench or LPIPS environment is
needed for these generation-only scripts.

## Model configuration

Copy `configs/paths.env.example` to `configs/paths.env`, then set the paths for
the model you plan to use. The latter file is local-only and Git-ignored.
`QUANTWM_PYTHON=python` uses the active environment. Set it to another Python
executable if desired; the launcher and native torchrun processes use that
same interpreter.

An alternative external configuration is supported:

```bash
QUANTWM_PATHS_FILE=/path/to/my-local-paths.env \
  bash scripts/hy_worldplay/quantwm.sh 0 --dry-run
```

The adapter-specific model paths are documented in the two main READMEs.
Weights are not bundled or automatically downloaded. Obtain compatible
checkpoints from the upstream projects and accept their model terms.

## Versions and checks

The compact version record is [tested_versions.txt](tested_versions.txt).
It lists relevant runtime packages rather than a full workstation freeze.
LongCat and HY depend on particular diffusers/transformers interfaces; do
not upgrade those packages independently without testing the backend.

```bash
ffmpeg -version
ffprobe -version
python -c 'import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)'
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
bash scripts/hy_worldplay/quantwm.sh 0 --dry-run
```

Unit tests do not load checkpoints or run generation. `--dry-run` validates
configured paths and prints the command; it does not load a model or create
an output directory. Full-generation validation still requires the relevant
checkpoints and sufficient GPU memory.
