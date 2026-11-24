# Third-party sources and notices

The files under `vendor/` are upstream inference backends and quantization
primitives, not original QuantWM code. They are not relicensed by a license
that may be applied to the QuantWM core. Model weights are not included and
have separate terms.

## Source attribution

| Directory | Sources |
|---|---|
| `quant_videogen/` | [Quant-VideoGen](https://github.com/svg-project/Quant-VideoGen) |
| `models/HY-WorldPlay/` | [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay), through the QVG integration |
| `models/Self-Forcing/` | [Self-Forcing](https://github.com/guandeh17/Self-Forcing), through the QVG integration |
| `models/LongCat/` | [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video), through the QVG integration |
| `models/Matrix-Game-2/` | [Matrix-Game-2](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) |
| `models/LingBot-v2/` | [LingBot-World-v2](https://github.com/Robbyant/lingbot-world-v2), with QVG-compatible cache hooks |

Integration changes include environment-configurable checkpoint paths,
QuantWM adapter calls, streaming cache hooks and generation settings. This
bundle is not an unmodified mirror of the upstream repositories.

## Preserved license files

- [Matrix-Game-LICENSE](licenses/Matrix-Game-LICENSE): MIT notice.
- [LingBot-v2-LICENSE.txt](licenses/LingBot-v2-LICENSE.txt):
  CC BY-NC-SA 4.0 text from LingBot-World-v2. This is the **v2** backend;
  a license from the older LingBot-World repository must not be substituted.
- [LongCat-LICENSE](licenses/LongCat-LICENSE): MIT notice.

Copyright/license headers in source files remain intact. Example images,
base models and checkpoints must also be used under their respective terms.

## Redistribution checks

The supplied QVG-derived snapshot did not contain a standalone QVG license,
or complete top-level license files for every copied backend. Public source
availability alone is not treated here as an additional license grant.
Before publishing or redistributing this entire bundle, confirm the relevant
upstream permissions and include any additional required license/NOTICE
files. The upstream [Self-Forcing license](https://github.com/guandeh17/Self-Forcing/blob/main/LICENSE)
and [HY-WorldPlay repository](https://github.com/Tencent-Hunyuan/HY-WorldPlay)
are starting points for checking those components.

In particular, do not label the complete vendor tree as uniformly MIT or
Apache-2.0: the preserved LingBot-v2 terms differ. Selecting a license for
the original `quantwm/` code does not replace third-party obligations.
