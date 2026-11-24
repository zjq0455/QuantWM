# QuantWM

[English](README.md) | **简体中文**

## Temporally Consistent 2-Bit KV Cache Quantization for Video World Models

[论文](https://arxiv.org/abs/2609.26425) ·
[项目主页](https://quantwm-project.github.io/QuantWM/) ·
[引用方式](#引用方式)

QuantWM 是一种无需训练的 KV Cache 量化方法，旨在保留注意力行为和视频时序一致性。
方法结合**量化敏感度感知聚类（QSAC）**与**主子空间注意力补偿（PSAC）**，
减小 Key 量化对 attention logits 和历史 token 选择的影响。

本仓库提供五个模型后端的单样本推理，每个模型均有独立的
**BF16、Quant-VideoGen（QVG）和 QuantWM** 入口。


## 1. 环境安装

参考配置为 Linux、Python 3.10、PyTorch 2.6.0 / CUDA 12.4 和 NVIDIA A800 80GB。
各模型的实际显存需求不同。

在仓库根目录执行：

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


## 2. 模型权重与路径

从对应上游项目下载兼容的模型权重，然后创建本地配置：

```bash
cp configs/paths.env.example configs/paths.env
# 编辑 configs/paths.env，填写需要使用的模型路径。
```

`configs/paths.env` 已加入 Git 忽略列表，不随项目分发本机路径。
`QUANTWM_PYTHON=python` 使用当前环境；也可以填写其他 Python 解释器的绝对路径。
调用者已导出的环境变量优先于模板中的默认值。

| 后端 | 配置变量 | 上游项目 |
|---|---|---|
| Matrix-Game-2 | `MATRIX_MODEL_PATH` | [Matrix-Game-2](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) |
| LingBot-World-v2 | `LINGBOT_MODEL_PATH` | [LingBot-World-v2](https://github.com/Robbyant/lingbot-world-v2) |
| HY-WorldPlay | `HY_MODEL_ID`、`HY_AR_MODEL_PATH`、`HY_CHECKPOINT_PATH` | [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)、[QVG 集成](https://github.com/svg-project/Quant-VideoGen) |
| Self-/Causal-Forcing | `SELF_FORCING_MODEL_PATH`、`SELF_FORCING_CHECKPOINT` | [Self-Forcing](https://github.com/guandeh17/Self-Forcing) |
| LongCat-Video | `LONGCAT_MODEL_PATH` | [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) |

Matrix 必须使用 **Matrix-Game-2** 的蒸馏权重，而非其他主版本；目录应包含
`base_distilled_model/base_distill.safetensors` 和 `Wan2.1_VAE.pth`。
LingBot 使用 causal-fast 权重。HY 入口使用随附的 QVG Wan-based AR 后端，需要
base model、AR transformer 目录和蒸馏 checkpoint 三部分。
Self-/Causal-Forcing 还需要 Wan2.1 T2V 1.3B 底座；LongCat 模型目录中需有
`lora/cfg_step_lora.safetensors`。

## 3. 生成对比视频

选择 GPU，以相同输入分别运行三个方法：

```bash
bash scripts/hy_worldplay/bf16.sh 0
bash scripts/hy_worldplay/qvg.sh 0
bash scripts/hy_worldplay/quantwm.sh 0
```

每个方法的独立视频保存在 `outputs/<model>/<method>/video.mp4`。
如果 `QUANTWM_PYTHON` 已指向环境中 Python 的绝对路径，就不必额外 activate 环境。

| 模型 | 脚本目录 | 输入 |
|---|---|---|
| Matrix-Game-2 | [`scripts/matrix_game2/`](scripts/matrix_game2/) | 首帧和 action，不使用文本控制生成 |
| LingBot-World-v2 | [`scripts/lingbot_v2/`](scripts/lingbot_v2/) | 首帧、prompt 和相机轨迹 |
| HY-WorldPlay | [`scripts/hy_worldplay/`](scripts/hy_worldplay/) | 首帧、prompt 和 pose 指令 |
| Self-/Causal-Forcing | [`scripts/self_forcing/`](scripts/self_forcing/) | prompt |
| LongCat-Video | [`scripts/longcat/`](scripts/longcat/) | prompt，可选共用初始视频 |

每个目录都包含 `bf16.sh`、`qvg.sh`、`quantwm.sh`。例如：

```bash
bash scripts/matrix_game2/quantwm.sh 0
bash scripts/lingbot_v2/quantwm.sh 0
bash scripts/self_forcing/quantwm.sh 0
bash scripts/longcat/quantwm.sh 0
```

添加 `--dry-run` 可检查路径并打印实际命令，不加载模型、不创建输出目录：

```bash
bash scripts/lingbot_v2/quantwm.sh 0 --dry-run
```

### 自定义输入与多卡


```bash
bash scripts/hy_worldplay/quantwm.sh 0 \
  --image /path/to/image.png \
  --prompt-file /path/to/prompt.txt \
  --output outputs/custom-hy/quantwm

bash scripts/matrix_game2/quantwm.sh 0 \
  --image /path/to/game-frame.png \
  --action 'turn_left:45:2,idle:2,turn_right:45:2' \
  --output outputs/custom-matrix/quantwm

# 同一个方法内部的模型并行；目前支持以下两个后端。
bash scripts/lingbot_v2/quantwm.sh 0,1
bash scripts/longcat/quantwm.sh 0,1
```


核心代码：

- [`history.py`](quantwm/history.py)：在线 Query 统计与采样。
- [`clustering.py`](quantwm/clustering.py)：QSAC 质心分配。
- [`correction.py`](quantwm/correction.py)：PSAC 低秩误差补偿。
- [`rounding.py`](quantwm/rounding.py)：INT2 残差量化。
- [`adapters/`](quantwm/adapters/)：各模型的 Cache 集成。


## 4. 项目结构

```text
quantwm/       量化核心、生成入口和模型适配器
scripts/       五个后端 × 三种方法的单样本脚本
configs/       可移植路径模板，本地配置不进入版本控制
environment/   环境安装说明与参考版本
examples/      配对示例输入及相机参数
vendor/        上游推理源码与许可证声明
tests/         无需 GPU 的发布检查
outputs/       运行时创建，已加入忽略列表
```

## 引用方式

```bibtex
@article{zhao2026quantwm,
  title={QuantWM: Temporally consistent 2-bit KV cache quantization for video world models},
  author={Zhao, Jiaqi and Hu, Xiaobin and Yin, Bo and Jiang, Junpeng and Zhang, Miao and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2609.26425},
  year={2026}
}
```

## 致谢与许可

感谢 Quant-VideoGen、Matrix-Game、LingBot、HY-WorldPlay、Self-Forcing 和 LongCat
团队提供推理后端与量化原语。第三方代码和素材保留各自的上游许可；模型权重单独分发。
发布和再分发前，请查阅[第三方声明](vendor/NOTICE.md)。
