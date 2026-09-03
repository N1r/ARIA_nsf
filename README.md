# ARIS

**简体中文** | [English](README_EN.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/N1r/ARIS_nsf/blob/main/notebooks/ARIS_Tutorial_and_Workflow.ipynb)
[![试听 Demo](https://img.shields.io/badge/demo-%E8%AF%95%E5%90%AC-blue)](https://n1r.github.io/ARIS_nsf/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

## 0. 简介

ARIS（Analytic Resonance for Interpretable Synthesis）是一个为语音学研究者
准备的可微分析合成工具：用单说话人录音训练 DDSP/GOLF 声码器，在同一条
重建语音上施加可记录的 F0、数字增益、随机源、声门源形状（`R_d`）或声道
共振（F1/F2、谱倾斜）控制，批量生成配对或连续统刺激。

- 试听 Demo：<https://n1r.github.io/ARIS_nsf/>
- 交互教程：[`notebooks/ARIS_Tutorial_and_Workflow.ipynb`](notebooks/ARIS_Tutorial_and_Workflow.ipynb)

训练需要一张支持 CUDA 的 NVIDIA GPU（Google Colab T4 即可试跑）；重建与刺激生成也可在 CPU 上运行。

按使用需求，ARIS 提供以下几种交互路径：

1. **Google Colab 在线教程**：点击页面顶部的 **Open in Colab** 徽章，无需配置本地环境，直接在云端 GPU 体验端到端工作流（从音频分析、演示训练到语音操控与试听）。
2. **在线试听 Demo**：直接打开[试听页面](https://n1r.github.io/ARIS_nsf/)，调节参数听取音色变化。
3. **本地图形化工作台 (Studio)**：在仓库目录下运行 `uv sync --locked --all-extras && uv run aris studio`，即可在浏览器可视化界面中拖动滑块调节声学参数并实时试听。
4. **命令行与 Python API**：适合自定义实验脚本、录音切分与大规模模型训练（见第 2–5 节与第 7 节）。

### 面向语音学研究的控制与验证

| 研究维度 | ARIS 控制 | 建议在输出上独立复测 |
|---|---|---|
| 音高与语调 | `pitch_semitones` | F0 中位数、轨迹形状、有声比例 |
| 元音共振 | `f1_*`、`f2_*` | F1/F2 轨迹、元音空间、可懂度 |
| 发声类型 | `glottal_rd_scale`、`noise_gain_db`、`tilt_alpha_delta` | H1–H2、CPP、HNR、谱斜率 |
| 刺激数字增益（非发声强度） | `output_gain_db` | 峰值、RMS/LUFS、达到数字满幅的采样点数 |

这些参数是**模型控制量**，不是生理测量或感知标签。例如，`output_gain_db`
不等于发声力度，`glottal_rd_scale` 也不等于 EGG 测得的声门参数。正式研究应
预注册目标声学指标，在生成后的 WAV 上复测实际效应，并报告重建误差。
ARIS 当前不提供时长、语速或句内局部区间控制。

### 推荐研究流程

1. 明确假设、目标声学指标和排除标准。
2. 以一致的录音链路采集单说话人材料，并保留原始文件。
3. 准备数据、固定 split、训练模型，在 held-out test set 上评估重建。
4. 从小效应量开始设计单参数条件；组合条件保留相应单参数对照。
5. 生成 baseline 与变体，盲听并独立复测 F0、共振峰、音质和波形幅度。
6. 归档 checkpoint、配置、数据 fingerprint、操控元数据和分析脚本。

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
**运行时 → 全部运行**。教程同样全程使用 uv：先用官方安装脚本安装 uv，再通过
`uv pip install --system` 把锁定依赖装入 Colab 当前 kernel；无需重启运行时。
教程默认训练 1,000 步并绘制训练/验证 loss，再用该 checkpoint 完成重建与操控。
这足以观察学习过程和试听初步结果，但不代表模型已经收敛或可直接用于正式实验；
Release 中的示例 checkpoint 训练了 40,000 步，可作为稳定质量参照。

**直接使用现成预训练模型试一试：** 官方 Release 提供训练好的模型（普通话女声，
16 kHz）及配套示例数据。先下载并解压
`aris_f024_demo.zip`，再运行重建与参数操控：

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

成功的标志：`out/demo_recon/` 下出现重建的 WAV 文件，`out/demo_stimuli/`
下出现每个条件一套的 WAV 及记录生成方式的 JSON 元数据。对听 `f1_up`
变体和重建原声，能听出第一共振峰升高带来的音色变化。

## 2. 准备数据

先把你的录音放进一个文件夹（WAV 格式，单说话人，安静环境）：

```text
recordings/
├── session1.wav
├── session2.wav
└── ...
```

然后依次执行切分、预处理和检查：

```bash
uv run aris split recordings/ segments/ --mode silence        # 按静音切成短句
uv run aris prepare segments/audio data/my_voice              # 重采样、提取 F0、划分数据集
uv run aris validate data/my_voice                            # 确认数据完整可用
```

关于数据与实验设计，有几点经验可以参考：

1. 建议总数据量为 20–60 分钟。保持话筒、距离、增益、房间和说话任务一致，
   避免波形过载失真、混响、背景声和自动增益；同时确认录音授权与数据使用范围。
2. 长录音切成数秒的短句训练更快。若材料跨录音场次、词表或语体，正式研究应
   先规划 split，避免同一项目的近重复版本跨越 train/test 造成信息泄漏。
3. F0 提取推荐 `--f0-method pyworld`（WORLD 的 DIO+StoneMask，稳定且快）；
   默认 `auto` 在未安装 WORLD 时退回自相关法。
4. 汉语声调等对 F0 敏感的研究，推荐先用 [RMVPE](https://github.com/Dream-High/RMVPE)
   或 Praat 提取更可靠的音高轨迹，保存为与每条 WAV 同名的 `.pv` 文件，
   再用 `--f0-method sidecar` 读入。`.pv` 是纯文本格式：每行一个浮点数，
   每行对应固定 5 ms 的一帧，`0.0` 表示无声帧，正数表示该帧 F0（Hz）。
   注意 Praat 默认的 pitch time step 不是 5 ms，无声帧标的也是
   `--undefined--` 而不是 `0`——从 Praat 导出后要重采样到 5 ms 网格、把
   `--undefined--` 换成 `0.0`，才能喂给 `prepare`。帧数和音频时长对不上
   时，`prepare` 现在会直接报错，不会静默错位。
5. 采样率不必预先统一，`prepare` 会自动重采样到模型采样率。建议保留未经
   重采样和归一化的原始录音，避免覆盖研究档案。
6. 手头没有录音时，可用 `uv run aris fetch-corpus data/arctic` 下载
   约 30 分钟的公开语料 CMU ARCTIC 试跑全流程；下载完成后用
   `uv run aris prepare data/arctic/selected data/arctic_prepared`
   继续后续步骤。

## 3. 训练

数据准备好了，就可以生成实验目录、启动训练：

```bash
uv run aris init-experiment data/my_voice experiments/my_voice --model aria-golf
uv run aris train experiments/my_voice --dry-run   # 打印将要执行的训练命令
uv run aris train experiments/my_voice
```

几点说明：

1. 按研究假设选择最简单且足够的模型：

   | 模型 | 可用控制 | 典型用途 |
   |---|---|---|
   | `ddsp` | F0、数字增益、随机源增益 | 基线重建与音高实验 |
   | `golf` | 以上 + 声门 `R_d` | 声源与发声类型研究 |
   | `aria-golf` | 以上 + F1/F2、谱倾斜 | 显式声源–声道操控 |

   `aria-golf` 是 ARIS 解码器在代码中的名称。普通 `golf` 的 LPC 系数不能
   直接解释为独立的 F1/F2 控制。
2. checkpoint 保存在 `experiments/my_voice/runs/checkpoints/`。
3. 训练需要支持 CUDA 的 PyTorch。默认安装的 Linux 版 PyTorch 已带
   CUDA 支持；如果与你的显卡或驱动不匹配，需要按设备另装对应版本——
   请参考 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)，再用
   `uv add` 或 `uv pip install` 安装与你的驱动匹配的 PyTorch 构建。
4. 在 Slurm 集群上，`init-experiment` 已生成可直接 `sbatch` 提交的
   `train.slurm`；集群参数（partition、GPU 类型等）可用 `init-experiment`
   的选项调整，提交前按你的集群补上 CUDA module。没有集群则忽略它。
5. 新建的实验目录是可移动的：把 `init-experiment` 生成的实验目录和它
   对应的数据集目录一起搬到别处（保持相对位置不变），在任意工作目录下
   `train`/`synthesize`/`manipulate` 依然能找到数据集。已有的旧实验
   目录（比如前面下载的 demo）仍按原路径解析，不受影响。

## 4. 重建（推理）

训练完成后，先用 checkpoint 重建测试集录音，看看模型质量如何：

```bash
uv run aris synthesize experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/reconstruction
```

输出为重建的 WAV 文件。不要只凭“听起来像”进入下一步：应在 held-out test
set 上把原音与重建并排检查，至少记录可懂度、伪影、F0 偏差、共振峰偏差、
时长、波形幅度，并统计是否存在达到数字满幅而被硬限幅的采样点。模型重建误差
和实验操控效应是两个不同来源，应分别报告。

## 5. 生成操控刺激

这一步是 ARIS 的核心用途：在重建的基础上单独改变某个参数，其余保持不变：

```bash
uv run aris manipulate experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/stimuli \
  --variant 'pitch_down:pitch_semitones=-4' \
  --variant 'f1_up:f1_scale=1.2'
```

每个 `--variant` 是一组命名条件（`名字:参数=值,参数=值`），各生成一套
WAV，附带记录生成方式的 JSON 元数据。可用参数与范围：

| 参数 | 范围 | 含义 | DDSP | GOLF | ARIS-GOLF |
|---|---|---|:---:|:---:|:---:|
| `pitch_semitones` | `-36..36` | 音高（半音） | ✓ | ✓ | ✓ |
| `output_gain_db` | `-24..12` | 保存 WAV 前的数字增益（dB） | ✓ | ✓ | ✓ |
| `noise_gain_db` | `-24..24` | 随机源分支的数字增益（dB） | ✓ | ✓ | ✓ |
| `glottal_rd_scale` | `0.5..2.0` | 模型声门源 `R_d` 的比例 | — | ✓ | ✓ |
| `f1_scale` / `f2_scale` | `0.7..1.3` | 第一/第二共振峰（比例） | — | — | ✓ |
| `f1_hz` / `f2_hz` | `150..1300` / `600..3200` | 第一/第二共振峰（绝对 Hz） | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | 谱倾斜 | — | — | ✓ |

`f1_hz` / `f2_hz` 把共振峰设到一个绝对 Hz 值，适合搭建跨条目一致的 Hz
连续统；`f1_scale` / `f2_scale` 按比例整体平移、保留每帧原有轮廓，更
适合保留自然语调形状的操控。两者不能对同一共振峰同时使用。

### 实验设计与质量控制

- 软件允许范围不等于语音学上有效的范围；先以小步长预试，再根据声学复测和
  可懂度确定最终效应量。
- 单参数条件最容易解释。多参数条件用于明确的交互假设，并应保留 baseline
  及对应的单参数条件。
- 所有正式条件应使用同一个冻结 checkpoint。检查 `manipulation.json` 中的
  checkpoint SHA-256、数据 fingerprint 和完整控制值。
- 检查每个条件目录的 `_render.json`。其中 `clipped_samples` 表示达到数字满幅
  后被硬限幅的采样点数量；正式材料建议该值为 `0`。不要只对个别条件事后归一化。
- 比较页面和 Studio 适合质检，不包含随机化、盲法或实际播放声压级校准，不能直接
  替代正式知觉实验平台。

各参数的听感效果可在 [Demo 页面](https://n1r.github.io/ARIS_nsf/)直接
试听；参数含义与条件设计详见 [Manipulation 指南](docs/MANIPULATION_ZH.md)。

## 6. 浏览器工作台（Studio）

不想在命令行里拼 `--variant` 字符串的话，可以在浏览器里完成条件设计和
试听对比：

```bash
# 本地启动（自动打开 http://127.0.0.1:8765/）：
uv run aris studio

# 在 Google Colab 或远程云服务器上启动（自动生成公网 Gradio 链接）：
uv run aris studio --share
```

页面按模型自动生成参数滑杆，支持命名条件与连续统生成器（比如 F1 从
400 到 600 Hz 分五档，一键生成整组刺激）、一键渲染、原始/变体 A/B
试听，以及时间轴对齐的波形与语谱图对比；检测到硬限幅采样点或共振峰到达模型边界时
会以红色标出。渲染结果保存在 `studio_output/` 下，与命令行 `manipulate`
的输出和元数据格式完全一致。

## 7. Python API 使用方式（Jupyter / Colab / 脚本）

除命令行外，ARIS 也提供 Python API。把下面代码保存为 `workflow.py`，再用
`uv run python workflow.py` 执行；Jupyter 可通过
`uv run --with jupyter jupyter lab` 启动。

```python
import aris

# 1. 切分长音频 & 提取特征划分数据集
aris.split("recordings/", "segments/", mode="silence")
manifest = aris.prepare("segments/audio", "data/my_voice", sample_rate=16000)

# 2. 校验数据集完整性
errors = aris.validate("data/my_voice")
assert not errors

# 3. 初始化实验并启动训练
exp_dir = aris.init_experiment("data/my_voice", "experiments/my_voice", model="aria-golf")
aris.train(exp_dir)

# 4. 重建语音（推理）
aris.synthesize(
    exp_dir,
    "experiments/my_voice/runs/checkpoints/last.ckpt",
    "out/recon",
)

# 5. 生成参数操控刺激（支持直接传入命名字符串或 ControlVariant 对象）
aris.manipulate(
    exp_dir,
    "experiments/my_voice/runs/checkpoints/last.ckpt",
    "out/stimuli",
    variants=[
        "f1_up:f1_scale=1.2",
        "pitch_down:pitch_semitones=-4",
        "rd_high:glottal_rd_scale=1.6",
    ],
)

# 6. 在 Notebook 中一键启动可视化 Web 工作台
aris.launch_studio(workspace=".", share=True)
```

## 8. 命令一览

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

## 9. 引用

机器可读的引用信息见 [CITATION.cff](CITATION.cff)。

ARIS 的声码器实现源自 GOLF：

- C.-Y. Yu and G. Fazekas, "Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis," *Interspeech 2024*. DOI: `10.21437/Interspeech.2024-1187`
- C.-Y. Yu and G. Fazekas, "Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables," *ISMIR 2023*. DOI: `10.5281/zenodo.10265377`

更早的方法基础是可微 DSP 与神经源–滤波器模型：

- J. Engel, L. Hantrakul, C. Gu, and A. Roberts, "DDSP: Differentiable Digital Signal Processing," *ICLR 2020*. arXiv: `2001.04643`
- X. Wang, S. Takaki, and J. Yamagishi, "Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis," *IEEE/ACM TASLP*, 2020. arXiv: `1904.12088`

代码以 MIT 协议发布，见 [LICENSE](LICENSE)。

## 10. 联系

遇到问题或有建议，欢迎提 [Issue](https://github.com/N1r/ARIS_nsf/issues)，
也可以邮件联系 <dingyr@hum.leidenuniv.nl>（Leiden University）。
