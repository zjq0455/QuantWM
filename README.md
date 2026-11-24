# QuantWM

**English** | [简体中文](README_zh-CN.md)

## Temporally Consistent 2-Bit KV Cache Quantization for Video World Models

[Paper](https://arxiv.org/abs/2609.26425) ·
[Project page](https://quantwm-project.github.io/QuantWM/) ·
[Citation](#citation)

QuantWM is a training-free KV-cache quantization method that aims to preserve
attention behavior and temporal consistency. It combines
**quantization-sensitivity-aware clustering (QSAC)** with
**principal-subspace attention compensation (PSAC)** to reduce the impact of
Key quantization on attention logits and historical token selection.

This repository provides single-sample inference for five backends, with
separate **BF16**, **Quant-VideoGen (QVG)** and **QuantWM** entry points.


## 1. Installation

The reference setup is Linux, Python 3.10, PyTorch 2.6.0 / CUDA 12.4 and NVIDIA
A800 80GB GPUs. GPU memory requirements depend on the backend.

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

## 2. Model weights

Download compatible checkpoints from the upstream projects, then create a
local configuration:

```bash
cp configs/paths.env.example configs/paths.env
# Edit configs/paths.env to set the paths for the model you want to run.
```

`configs/paths.env` is Git-ignored. It is not distributed with machine-specific
paths. `QUANTWM_PYTHON=python` selects the active environment; exported values
take precedence over the configuration defaults.

| Backend | Configuration variables | Upstream |
|---|---|---|
| Matrix-Game-2 | `MATRIX_MODEL_PATH` | [Matrix-Game-2](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) |
| LingBot-World-v2 | `LINGBOT_MODEL_PATH` | [LingBot-World-v2](https://github.com/Robbyant/lingbot-world-v2) |
| HY-WorldPlay | `HY_MODEL_ID`, `HY_AR_MODEL_PATH`, `HY_CHECKPOINT_PATH` | [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay), [QVG integration](https://github.com/svg-project/Quant-VideoGen) |
| Self-/Causal-Forcing | `SELF_FORCING_MODEL_PATH`, `SELF_FORCING_CHECKPOINT` | [Self-Forcing](https://github.com/guandeh17/Self-Forcing) |
| LongCat-Video | `LONGCAT_MODEL_PATH` | [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) |

Use the Matrix-Game-2 distilled checkpoint, not a different Matrix-Game major
version. Its directory must contain `base_distilled_model/base_distill.safetensors`
and `Wan2.1_VAE.pth`. LingBot uses the causal-fast checkpoint. The HY entry
uses the bundled QVG Wan-based AR backend and its base model, AR transformer
directory and distilled checkpoint. Self-/Causal-Forcing also needs the
Wan2.1 T2V 1.3B base weights. LongCat requires
`lora/cfg_step_lora.safetensors` in its model directory.

## 3. Generate a comparison

Choose a GPU and run the three methods with the same inputs:

```bash
bash scripts/hy_worldplay/bf16.sh 0
bash scripts/hy_worldplay/qvg.sh 0
bash scripts/hy_worldplay/quantwm.sh 0
```

Each method writes its own video to
`outputs/<model>/<method>/video.mp4`. The scripts select their Python
interpreter from the configuration, so a separate environment activation
is unnecessary if `QUANTWM_PYTHON` is an absolute executable path.

| Model | Script directory | Input |
|---|---|---|
| Matrix-Game-2 | [`scripts/matrix_game2/`](scripts/matrix_game2/) | First frame + actions; no text conditioning |
| LingBot-World-v2 | [`scripts/lingbot_v2/`](scripts/lingbot_v2/) | First frame + prompt + camera poses |
| HY-WorldPlay | [`scripts/hy_worldplay/`](scripts/hy_worldplay/) | First frame + prompt + pose commands |
| Self-/Causal-Forcing | [`scripts/self_forcing/`](scripts/self_forcing/) | Prompt |
| LongCat-Video | [`scripts/longcat/`](scripts/longcat/) | Prompt; optional shared initial video |

Every directory contains `bf16.sh`, `qvg.sh` and `quantwm.sh`. Other examples:

```bash
bash scripts/matrix_game2/quantwm.sh 0
bash scripts/lingbot_v2/quantwm.sh 0
bash scripts/self_forcing/quantwm.sh 0
bash scripts/longcat/quantwm.sh 0
```

Use `--dry-run` to validate paths and print the native command without loading
a model or creating outputs:

```bash
bash scripts/lingbot_v2/quantwm.sh 0 --dry-run
```

### Custom inputs and multiple GPUs

The prompt file must contain exactly one nonempty line. Keep images, prompts
and actions matched when comparing methods.

```bash
bash scripts/hy_worldplay/quantwm.sh 0 \
  --image /path/to/image.png \
  --prompt-file /path/to/prompt.txt \
  --output outputs/custom-hy/quantwm

bash scripts/matrix_game2/quantwm.sh 0 \
  --image /path/to/game-frame.png \
  --action 'turn_left:45:2,idle:2,turn_right:45:2' \
  --output outputs/custom-matrix/quantwm

# Model parallelism within a single method; supported for these two backends.
bash scripts/lingbot_v2/quantwm.sh 0,1
bash scripts/longcat/quantwm.sh 0,1
```


Core modules:

- [`history.py`](quantwm/history.py): online Query statistics and sampling.
- [`clustering.py`](quantwm/clustering.py): QSAC centroid assignment.
- [`correction.py`](quantwm/correction.py): PSAC low-rank error correction.
- [`rounding.py`](quantwm/rounding.py): INT2 residual quantizers.
- [`adapters/`](quantwm/adapters/): backend-specific cache integration.


## 4. Repository layout

```text
quantwm/       Quantization core, generation CLI and backend adapters
scripts/       Five backends × three single-sample entry points
configs/       Portable path template; local configuration is ignored
environment/   Installation notes and reference versions
examples/      Paired demo inputs and camera conditions
vendor/        Upstream inference sources and license notices
tests/         CPU-only release checks
outputs/       Created at runtime; excluded from version control
```

## Citation

```bibtex
@misc{zhao2026quantwmtemporallyconsistent2bit,
  title={QuantWM: Temporally Consistent 2-Bit KV Cache Quantization for Video World Models},
  author={Jiaqi Zhao and Xiaobin Hu and Bo Yin and Junpeng Jiang and Miao Zhang and Shuicheng Yan},
  year={2026},
  eprint={2609.26425},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.26425}
}
```

## Acknowledgments and licensing

We thank Quant-VideoGen and the Matrix-Game, LingBot, HY-WorldPlay,
Self-Forcing and LongCat teams. Their inference backends and quantization
primitives make these integrations possible. Third-party code and assets
retain their upstream terms; model weights are distributed separately.
See [third-party notices](vendor/NOTICE.md) before redistributing the bundle.
