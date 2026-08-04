# ARIS

**简体中文** | [English](README_EN.md)

ARIS（Analytic Resonance for Interpretable Synthesis）的官方工具包：
面向语音学研究的可微 DDSP/GOLF 分析合成与受控刺激生成。给定几十分钟的
单说话人录音，训练一个可微声码器，然后在其余参数不变的前提下单独操控
F0、能量、噪声、声源形状（`R_d`）或共振峰（F1/F2、谱倾斜），生成成对
实验刺激；从原始文件哈希到输出 WAV 全程留有 provenance。

- 试听 Demo：<https://n1r.github.io/ARIS_nsf/>
- 论文：SLT 2026（引用信息见文末与 `CITATION.cff`）

## 安装

只需要机器上有 [`uv`](https://docs.astral.sh/uv/)（一个 Python 环境
管理器，单个可执行文件即可）。以下三行分别是：进入项目环境、一键安装
全部依赖、自检环境是否就绪。Python 和依赖都装在仓库目录内，不影响系统：

```bash
source scripts/project_env.sh
./scripts/setup_project_env.sh
.venv/bin/aris doctor
```

## 快速开始

```bash
# 1. 数据准备：切分、重采样、F0 提取、质检
.venv/bin/aris split recordings/ segments/ --mode silence
.venv/bin/aris prepare segments/audio data/my_voice --f0-method autocorr
.venv/bin/aris validate data/my_voice

# 2. 建立并启动实验（--dry-run 先检查配置）
.venv/bin/aris init-experiment data/my_voice experiments/my_voice_golf --model golf
.venv/bin/aris train experiments/my_voice_golf --dry-run

# 3. 训练完成后：用模型文件（checkpoint）重建语音并生成操控条件
.venv/bin/aris synthesize experiments/my_voice_golf CKPT out/reconstruction
.venv/bin/aris manipulate experiments/my_voice_golf CKPT out/manipulations \
  --variant 'pitch_down:pitch_semitones=-4'
```

`CKPT` 指训练产出的模型文件（如 `runs/checkpoints/last.ckpt`）。推理与
manipulation 在普通电脑的 CPU 上即可运行；训练建议使用 GPU。如果你有
Slurm 集群，`init-experiment` 会生成可直接提交的作业脚本，但 Slurm 并非
必需。

**数据准备提示**：默认 F0 后端 `autocorr` 无需额外安装任何东西；已经用
Praat 等工具提取过的 `.pv` 音高轨迹可用 `--f0-method sidecar` 直接复用。

## 可操控参数

| 参数 | 范围 | DDSP | GOLF | ARIS-GOLF |
|---|---|:---:|:---:|:---:|
| `pitch_semitones` | `-36..36` | ✓ | ✓ | ✓ |
| `output_gain_db` | `-24..12` dB | ✓ | ✓ | ✓ |
| `noise_gain_db` | `-24..24` dB | ✓ | ✓ | ✓ |
| `glottal_rd_scale` | `0.5..2.0` | — | ✓ | ✓ |
| `f1_scale` / `f2_scale` | `0.7..1.3` | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | — | — | ✓ |

用法与条件设计见 [Manipulation 指南](docs/MANIPULATION_ZH.md)。

## CLI

```text
aris doctor             检查音频、F0、GPU 训练环境
aris fetch-corpus       获取并校验 CMU ARCTIC 示例语料
aris split              按静音边界或固定窗口切分连续录音
aris prepare            重采样、转单声道、提取 F0、确定性划分、生成哈希
aris validate           检查 manifest、路径和文件完整性
aris init-experiment    生成配置、provenance 与训练启动脚本
aris train              验证 dataset fingerprint 后启动 Lightning 训练
aris controls           列出实验模型声明的可操控参数
aris synthesize         用 checkpoint 重建 held-out 测试录音
aris manipulate         渲染命名的 F0/声源/噪声/声道控制条件
```

每个命令都有 `--help`。

## 代码结构

```text
src/aris/          单一 Python 包
├── cli.py …       命令行与数据准备（audio/corpus/segment/manifest/doctor）
├── controls/      manipulation 控制参数
├── presets/       内置模型配置（ddsp / golf / aria_golf）
├── models/        可微合成模型（GOLF / DDSP / ARIS）
├── training/      Lightning 训练模块
├── losses/        频谱损失
└── engine.py      训练/推理引擎入口（python -m aris.engine）
configs/           示例 decoder 配置
scripts/           环境安装脚本
tests/  docs/
```

## 设计约定

- 不默认做逐文件响度/峰值归一化，避免悄悄抹去研究变量。
- F0 上下界、算法、重采样率、划分种子进入 provenance；数据路径为相对路径，数据集可整体迁移。
- 原始文件 SHA-256 和 dataset fingerprint 防止无声的数据漂移。
- manipulation 记录 checkpoint SHA-256、全部控制值和削波统计。

## 开发

```bash
make test    # 全量测试
make lint    # ruff
```

其余 Makefile 目标：`install`、`test-lightweight`、`format`、`doctor`、`verify`。

## 引用

ARIS（SLT 2026）的引用条目将在论文上线后补充；机器可读信息见
[CITATION.cff](CITATION.cff)。底层 GOLF 方法：

- Chin-Yun Yu and György Fazekas, “Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis,” Interspeech 2024, DOI: `10.21437/Interspeech.2024-1187`.
- Chin-Yun Yu and György Fazekas, “Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables,” ISMIR 2023, DOI: `10.5281/zenodo.10265377`.

## License

MIT；详见 [LICENSE](LICENSE)。
