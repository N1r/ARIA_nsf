# ARIS

**简体中文** | [English](README_EN.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/N1r/ARIS_nsf/blob/main/notebooks/ARIS_Tutorial_and_Workflow.ipynb)
[![试听 Demo](https://img.shields.io/badge/demo-%E8%AF%95%E5%90%AC-blue)](https://n1r.github.io/ARIS_nsf/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

## 0. 简介

ARIS（Analytic Resonance for Interpretable Synthesis）是一个面向语音学与言语科学研究的可微分析合成工具。基于神经源–滤波器（Source-Filter）声码器架构，ARIS 能够在保持说话人音色与自然度的前提下，对基频轨迹（F0）、共振峰（F1/F2）、声门波形态（`R_d`）及谱倾斜等声学线索进行精准、正交的解耦调控，用于高效、可复现地批量生成知觉实验所需的成对刺激与声学连续统。

- 试听 Demo：<https://n1r.github.io/ARIS_nsf/>
- 交互教程：[`notebooks/ARIS_Tutorial_and_Workflow.ipynb`](notebooks/ARIS_Tutorial_and_Workflow.ipynb)

训练需要一张支持 CUDA 的 NVIDIA GPU（Google Colab T4 即可试跑）；重建与刺激生成也可在 CPU 上运行。

按使用需求，ARIS 提供以下几种交互路径：

1. **Google Colab 在线教程**：点击页面顶部的 **Open in Colab** 徽章，无需本地配置，即可直接在云端 GPU 上交互式运行完整分析-合成流程（涵盖音频特征提取、轻量模型训练及声学参数操控）。
2. **在线试听 Demo**：直接打开[试听页面](https://n1r.github.io/ARIS_nsf/)，调节参数听取音色变化。
3. **本地图形化工作台 (Studio)**：在仓库目录下运行 `uv sync --locked --all-extras && uv run aris studio`，即可在浏览器可视化界面中拖动滑块调节声学参数并实时试听。
4. **命令行接口**：适合自定义实验脚本、录音切分与大规模模型训练（见第 2–5 节及第 7 节命令一览）。

### 面向语音学研究的控制与验证

| 研究维度 | ARIS 控制 | 建议在输出上独立复测 |
|---|---|---|
| 音高与语调 | `pitch_semitones` | F0 中位数、轨迹形状、有声比例 |
| 元音共振 | `f1_*`、`f2_*` | F1/F2 轨迹、元音空间、可懂度 |
| 发声类型 | `glottal_rd_scale`、`noise_gain_db`、`tilt_alpha_delta` | H1–H2、CPP、HNR、谱斜率 |
| 刺激数字增益（非发声强度） | `output_gain_db` | 峰值、RMS/LUFS、达到数字满幅的采样点数 |


## 1. 快速上手与依赖安装

运行环境：Linux、macOS 或 Windows（建议通过 [WSL](https://learn.microsoft.com/zh-cn/windows/wsl/install) 运行）。

### 推荐安装方式（使用 uv 一键同步）

项目使用 `pyproject.toml`、`uv.lock` 和 [uv](https://docs.astral.sh/uv/) 统一管理 Python 与依赖。克隆仓库后，在仓库根目录执行：

```bash
# 创建项目环境并严格按锁文件同步全部依赖
uv sync --locked --all-extras

# 在项目环境中运行诊断工具
uv run aris doctor
```

本文后续命令均使用 `uv run aris ...`，无需激活 `.venv`。依赖有变更时重新执行
`uv sync --locked --all-extras` 即可；若是主动修改项目依赖，请先运行 `uv lock`
并提交更新后的锁文件。

`doctor` 会检查 Python、音频依赖、PyTorch、CUDA 与 GPU 状态。所有必需项通过后即可继续。

### Google Colab

打开页面顶部的 Colab 链接，将运行时类型设为 **T4 GPU**，然后选择
**运行时 → 全部运行**。教程全程使用 uv 管理环境与依赖，无需手动配置。
该演示配置（batch size 32、1,500 步）约需 2.2 GB 显存，可在 T4 GPU 上快速生成可听示例；若需要更高质量的模型，可参考 Release 中训练了 40,000 步的检查点。

**使用预训练模型快速验证：** 官方 Release 提供了预训练模型（普通话女声，16 kHz）及配套示例数据。下载解压后即可直接运行重建与参数操控：

```bash
# 下载官方示例包（只需执行一次）
curl -LO https://github.com/N1r/ARIS_nsf/releases/download/v0.1.0/aris_f024_demo.zip
unzip -q aris_f024_demo.zip

# 1. 重建测试录音
uv run aris synthesize demo_f024/experiment demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_recon

# 2. 生成共振峰与音高操控刺激
uv run aris manipulate demo_f024/experiment demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_stimuli \
  --variant 'f1_up:f1_scale=1.2' \
  --variant 'f1_down:f1_scale=0.85' \
  --variant 'pitch_down:pitch_semitones=-4' \
  --variant 'rd_high:glottal_rd_scale=1.6'
```

执行完成后，`out/demo_recon/` 目录将保存重建的 WAV 文件，`out/demo_stimuli/` 目录将生成各操控条件的音频文件及记录生成参数的 JSON 元数据。

## 2. 准备数据

将单说话人录音放入指定目录（WAV 格式，建议在安静低混响环境下采集）：

```text
recordings/
├── session1.wav
├── session2.wav
└── ...
```

依次执行切分、特征提取与完整性校验：

```bash
uv run aris split recordings/ segments/ --mode silence               # 按静音切分为短句
uv run aris prepare segments/audio data/my_voice --extract-formants # 提取 F0 与 Praat F1/F2 并划分数据集
uv run aris validate data/my_voice                                  # 检查数据集完整性
```

数据准备要点：

1. **录音规格**：建议有效音频时长 20–60 分钟。保持录音设备、增益、拾音距离与声学环境一致，避免波形截幅失真与强混响。
2. **切分与划分**：长录音切分为数秒短句有助于加快训练。若语料包含多个录音场次或词表，应提前规划数据集划分以防信息泄露。
3. **F0 提取**：推荐 `--f0-method pyworld`（基于 WORLD 的 DIO+StoneMask，兼顾稳定性与速度）；未指定时默认使用自相关法。
4. **共振峰监督**：训练 `aria-golf` 模型必须添加 `--extract-formants`，该选项在 10 ms 帧长上运行 Praat Burg 分析以提取 F1/F2 监督目标。女声通常使用默认上限（5,500 Hz），男声可按需调整（例如 `--formant-ceiling 5000`）。
5. **外部音高轨迹（Sidecar）**：对于声调语言或对 F0 精度敏感的研究，可使用 [RMVPE](https://github.com/Dream-High/RMVPE) 或 Praat 提取音高轨迹，保存为同名 `.pv` 文件并通过 `--f0-method sidecar` 载入。`.pv` 为纯文本格式（固定 5 ms 帧移，每行一个浮点数，`0.0` 为无声帧）。若帧数与音频时长不匹配，`prepare` 将直接报错。
6. **采样率适配**：输入音频无需预先统一采样率，`prepare` 会自动重采样至模型目标采样率。建议始终保留原始录音母带。
7. **示例语料**：若暂无自备录音，可运行 `uv run aris fetch-corpus data/arctic` 下载约 30 分钟的公开语料 CMU ARCTIC 进行全流程测试。

## 3. 训练

完成数据准备后，生成实验配置并启动训练：

```bash
uv run aris init-experiment data/my_voice experiments/my_voice --model aria-golf
uv run aris train experiments/my_voice --dry-run   # 打印实际执行的训练命令
uv run aris train experiments/my_voice
```

说明：

1. **模型选择**：按实验需求选择对应的模型架构：

   | 模型 | 可用控制参数 | 典型用途 |
   |---|---|---|
   | `ddsp` | F0、数字增益、随机源增益 | 基线重建与音高实验 |
   | `golf` | 以上 + 声门 `R_d` | 声源与发声类型研究 |
   | `aria-golf` | 以上 + F1/F2、谱倾斜 | 显式声源–声道联合操控 |

   `aria-golf` 在多尺度谱重建损失的基础上，引入了 F1/F2 监督与时间平滑约束，确保共振峰参数具有明确的解析物理意义。
2. **检查点路径**：模型权重自动保存在 `experiments/my_voice/runs/checkpoints/`。
3. **CUDA 与 PyTorch**：默认环境预装带 CUDA 支持的 PyTorch；若驱动或显卡架构不匹配，请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)通过 `uv pip install` 安装适配版本。
4. **集群支持 (Slurm)**：`init-experiment` 会同步生成 `train.slurm` 脚本，可按集群环境调整 GPU 配置后直接 `sbatch` 提交。单机运行可直接忽略该文件。
5. **路径便携性**：实验目录支持相对路径重定位。将实验目录与对应数据集目录一同移动时（保持二者相对层级），训练与推理脚本仍可自动定位数据。

## 4. 重建（推理）

训练完成后，使用检查点对测试集录音进行重合成评估：

```bash
uv run aris synthesize experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/reconstruction
```

输出为重建后的 WAV 音频。建议在测试集上对比原音与重建结果，检查可懂度、伪影、F0 与共振峰偏差，并确认是否存在硬截幅（clipping）采样点。

## 5. 生成操控刺激

在语音重建的基础上，可独立调整指定的声学控制参数，批量生成成对刺激或连续统：

```bash
uv run aris manipulate experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/stimuli \
  --variant 'pitch_down:pitch_semitones=-4' \
  --variant 'f1_up:f1_scale=1.2'
```

每个 `--variant` 定义一个命名实验条件（格式为 `条件名:参数=值,参数=值`），为每个条件生成独立的音频输出及记录参数配置的 JSON 元数据。

支持的控制参数与调节范围：

| 参数 | 范围 | 含义 | DDSP | GOLF | ARIS-GOLF |
|---|---|---|:---:|:---:|:---:|
| `pitch_semitones` | `-36..36` | 音高偏移行程（半音） | ✓ | ✓ | ✓ |
| `output_gain_db` | `-24..12` | 波形数字增益（dB） | ✓ | ✓ | ✓ |
| `noise_gain_db` | `-24..24` | 随机噪声源增益（dB） | ✓ | ✓ | ✓ |
| `glottal_rd_scale` | `0.5..2.0` | 声门源波形形状参数 $R_d$ 缩放比例 | — | ✓ | ✓ |
| `f1_scale` / `f2_scale` | `0.7..1.3` | F1 / F2 相对缩放比例（保留原有轮廓） | — | — | ✓ |
| `f1_hz` / `f2_hz` | `150..1300` / `600..3200` | F1 / F2 绝对目标频率（Hz） | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | 谱倾斜系数偏移量 | — | — | ✓ |

### 参数说明与质检提示

- **连续统构建**：`f1_hz` / `f2_hz` 将共振峰固定为绝对频率，适合跨条目统一刺激；`f1_scale` / `f2_scale` 按比例平移并保留原有轨迹轮廓。两者不可对同一共振峰同时使用。
- **可复现性追溯**：每个输出目录附带 `manipulation.json`，完整记录控制参数、数据集特征指纹及检查点 SHA-256 哈希。
- **防截幅检查**：检查输出目录中的 `_render.json`；若 `clipped_samples > 0` 说明存在数字满幅限幅，建议适当降低 `output_gain_db`。
- **听感效果与详解**：各参数实际听感可在 [在线 Demo](https://n1r.github.io/ARIS_nsf/) 体验；参数定义与进阶设计详见 [Manipulation 指南](docs/MANIPULATION_ZH.md)。

## 6. 浏览器工作台（Studio）

ARIS 提供基于浏览器的交互式工作台，方便可视化调节参数与试听对比：

```bash
# 本地启动（自动打开 http://127.0.0.1:8765/）
uv run aris studio

# 在 Google Colab 或远程服务器上启动（生成公网 Gradio 分享链接）
uv run aris studio --share
```

界面功能特性：
- **参数调节与连续统构建**：按模型动态生成控制滑块，支持一键生成多步长连续统刺激。
- **A/B 盲听与视觉对比**：支持原始重建与变体音频的 A/B 快速切换，并提供时间对齐的波形与语谱图展示。
- **边界与截幅预警**：硬截幅采样点及超出模型合理边界的参数将以红色高亮提示。
- **统一输出规范**：渲染结果默认保存在 `studio_output/`，其音频和元数据格式与命令行 `manipulate` 完全一致。

## 7. 命令一览

```text
uv run aris doctor             检查音频与训练依赖、CUDA 及 GPU 状态
uv run aris fetch-corpus       下载 CMU ARCTIC 示例语料
uv run aris split              切分连续录音
uv run aris prepare            重采样、提取 F0、划分数据集
uv run aris validate           检查数据完整性
uv run aris init-experiment    生成训练实验目录
uv run aris train              启动训练
uv run aris controls           列出模型支持的操控参数
uv run aris synthesize         用 checkpoint 重建录音
uv run aris manipulate         生成操控刺激
uv run aris studio             启动浏览器工作台（支持 --share 生成公网链接）
```

## 8. 引用

机器可读的引用信息见 [CITATION.cff](CITATION.cff)。

ARIS 的声码器实现源自 GOLF：

- C.-Y. Yu and G. Fazekas, "Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis," *Interspeech 2024*. DOI: `10.21437/Interspeech.2024-1187`
- C.-Y. Yu and G. Fazekas, "Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables," *ISMIR 2023*. DOI: `10.5281/zenodo.10265377`

更早的方法基础是可微 DSP 与神经源–滤波器模型：

- J. Engel, L. Hantrakul, C. Gu, and A. Roberts, "DDSP: Differentiable Digital Signal Processing," *ICLR 2020*. arXiv: `2001.04643`
- X. Wang, S. Takaki, and J. Yamagishi, "Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis," *IEEE/ACM TASLP*, 2020. arXiv: `1904.12088`

代码以 MIT 协议发布，见 [LICENSE](LICENSE)。

## 9. 联系

遇到问题或有建议，欢迎提 [Issue](https://github.com/N1r/ARIS_nsf/issues)，
也可以邮件联系 <dingyr@hum.leidenuniv.nl>（Leiden University）。
