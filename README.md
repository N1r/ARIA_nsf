# ARIS

**简体中文** | [English](README_EN.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/N1r/ARIS_nsf/blob/main/notebooks/ARIS_Tutorial_and_Workflow.ipynb)
[![试听 Demo](https://img.shields.io/badge/demo-%E8%AF%95%E5%90%AC-blue)](https://n1r.github.io/ARIS_nsf/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

## 0. 简介

ARIS（Analytic Resonance for Interpretable Synthesis）是一个为语音学研究者
准备的可微分析合成工具：用几十分钟的单说话人录音训练一个 DDSP/GOLF 声码器，
然后在其余参数保持不变的前提下，单独改变 F0、能量、噪声、声源形状（`R_d`）
或共振峰（F1/F2、谱倾斜），批量生成成对的实验刺激。

- 试听 Demo：<https://n1r.github.io/ARIS_nsf/>
- **Google Colab 在线全流程体验**：直接点击上方的 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/N1r/ARIS_nsf/blob/main/notebooks/ARIS_Tutorial_and_Workflow.ipynb) 徽章，免费在云端 GPU 上无需任何配置直接运行全流程。

不需要专门的高端计算设备：训练在一张普通游戏显卡（如 RTX 4060）或 Google Colab T4 上即可完成，
几十分钟数据约训练数小时；重建与刺激生成甚至不需要显卡，普通电脑的 CPU 就能快速运行。

按使用需求，ARIS 提供以下几种交互路径：

1. **Google Colab 在线教程**：点击 [Open In Colab 教程](notebooks/ARIS_Tutorial_and_Workflow.ipynb)，无需配置本地环境，直接在云端 GPU 体验端到端工作流（从音频分析、演示训练到语音操控与试听）。
2. **在线试听 Demo**：直接打开[试听页面](https://n1r.github.io/ARIS_nsf/)，调节参数听取音色变化。
3. **本地图形化工作台 (Studio)**：在仓库目录下运行 `uv sync --all-extras && uv run aris studio`，即可在浏览器可视化界面中拖动滑块调节声学参数并实时试听。
4. **命令行与 Python API**：适合自定义实验脚本、录音切分与大规模模型训练（见第 2–5 节与第 7 节）。

## 1. 快速上手与依赖安装

运行环境：Linux、macOS 或 Windows（建议通过 [WSL](https://learn.microsoft.com/zh-cn/windows/wsl/install) 运行）。

### 推荐安装方式（使用 uv 一键同步）

项目基于 `pyproject.toml` 与 `uv.lock` 进行现代依赖管理。只需安装单文件工具 [uv](https://docs.astral.sh/uv/)，在仓库根目录下执行一条命令即可自动完成 Python 3.11 虚拟环境的建立和全部依赖（音频、训练、Studio 等）的秒级安装：

```bash
# 1. 一键创建虚拟环境并同步全部依赖：
uv sync --all-extras

# 2. 运行内置诊断工具（可通过 uv run 直接调用，无需手动激活环境）：
uv run aris doctor
```

> **提示**：若习惯激活虚拟环境后使用，也可执行 `source .venv/bin/activate`，之后即可直接使用 `aris doctor`、`aris train` 等命令。若需使用传统 `pip`，亦支持 `pip install -e ".[all]"`。

`doctor` 会自动探测您的 Python 环境、CUDA 工具链与 GPU 显卡（如 RTX 4090 / L4 / T4 等），确认依赖完备即可开始！

**直接使用现成预训练模型试一试：** 仓库中已包含训练好的模型（普通话女声，16 kHz）及配套示例数据（若本地缺失可直接下载官方 ZIP 包：`curl -LO https://github.com/N1r/ARIS_nsf/releases/download/v0.1.0/aris_f024_demo.zip && unzip -q aris_f024_demo.zip`），直接运行重建与参数操控：

```bash
# 1. 重建测试录音
uv run aris synthesize demo_f024/experiment demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_recon

# 2. 生成共振峰与音高操控刺激
uv run aris manipulate demo_f024/experiment demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_stimuli \
  --variant 'f1_up:f1_scale=1.2' \
  --variant 'f1_down:f1_scale=0.85' \
  --variant 'pitch_down:pitch_semitones=-4' \
  --variant 'breathy:glottal_rd_scale=1.6'
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
.venv/bin/aris split recordings/ segments/ --mode silence        # 按静音切成短句
.venv/bin/aris prepare segments/audio data/my_voice              # 重采样、提取 F0、划分数据集
.venv/bin/aris validate data/my_voice                            # 确认数据完整可用
```

关于数据，有几点经验可以参考：

1. 建议总数据量为 20–60 分钟；长录音切成数秒的短句训练更快。
2. F0 提取推荐 `--f0-method pyworld`（WORLD 的 DIO+StoneMask，稳定且快）；
   默认 `auto` 在未安装 WORLD 时退回自相关法。
3. 汉语声调等对 F0 敏感的研究，推荐先用 [RMVPE](https://github.com/Dream-High/RMVPE)
   或 Praat 提取更可靠的音高轨迹，保存为与每条 WAV 同名的 `.pv` 文件，
   再用 `--f0-method sidecar` 读入。`.pv` 是纯文本格式：每行一个浮点数，
   每行对应固定 5 ms 的一帧，`0.0` 表示无声帧，正数表示该帧 F0（Hz）。
   注意 Praat 默认的 pitch time step 不是 5 ms，无声帧标的也是
   `--undefined--` 而不是 `0`——从 Praat 导出后要重采样到 5 ms 网格、把
   `--undefined--` 换成 `0.0`，才能喂给 `prepare`。帧数和音频时长对不上
   时，`prepare` 现在会直接报错，不会静默错位。
4. 采样率不必预先统一，`prepare` 会自动重采样到模型采样率。
5. 手头没有录音时，可用 `.venv/bin/aris fetch-corpus data/arctic` 下载
   约 30 分钟的公开语料 CMU ARCTIC 试跑全流程；下载完成后用
   `.venv/bin/aris prepare data/arctic/selected data/arctic_prepared`
   继续后续步骤。

## 3. 训练

数据准备好了，就可以生成实验目录、启动训练：

```bash
.venv/bin/aris init-experiment data/my_voice experiments/my_voice --model aria-golf
.venv/bin/aris train experiments/my_voice --dry-run   # 打印将要执行的训练命令
.venv/bin/aris train experiments/my_voice
```

几点说明：

1. `--model` 可选 `ddsp`、`golf`、`aria-golf`；需要共振峰（F1/F2）和
   谱倾斜操控时选 `aria-golf`（`aria-golf` 是 ARIS 解码器在代码中的名称）。
2. checkpoint 保存在 `experiments/my_voice/runs/checkpoints/`。
3. 训练需要支持 CUDA 的 PyTorch。默认安装的 Linux 版 PyTorch 已带
   CUDA 支持；如果与你的显卡或驱动不匹配，需要按设备另装对应版本——
   可以把显卡型号和操作系统告诉 ChatGPT、Claude 等 AI 助手请它给出
   安装命令，或参考 [pytorch.org](https://pytorch.org) 的版本选择器。
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
.venv/bin/aris synthesize experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/reconstruction
```

输出为重建的 WAV 文件；和原始录音对听一下，觉得接近就可以进入下一步。

## 5. 生成操控刺激

这一步是 ARIS 的核心用途：在重建的基础上单独改变某个参数，其余保持不变：

```bash
.venv/bin/aris manipulate experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/stimuli \
  --variant 'pitch_down:pitch_semitones=-4' \
  --variant 'f1_up:f1_scale=1.2'
```

每个 `--variant` 是一组命名条件（`名字:参数=值,参数=值`），各生成一套
WAV，附带记录生成方式的 JSON 元数据。可用参数与范围：

| 参数 | 范围 | 含义 | DDSP | GOLF | ARIS-GOLF |
|---|---|---|:---:|:---:|:---:|
| `pitch_semitones` | `-36..36` | 音高（半音） | ✓ | ✓ | ✓ |
| `output_gain_db` | `-24..12` | 整体能量（dB） | ✓ | ✓ | ✓ |
| `noise_gain_db` | `-24..24` | 噪声成分（dB） | ✓ | ✓ | ✓ |
| `glottal_rd_scale` | `0.5..2.0` | 声源形状 `R_d`（气声–紧嗓） | — | ✓ | ✓ |
| `f1_scale` / `f2_scale` | `0.7..1.3` | 第一/第二共振峰（比例） | — | — | ✓ |
| `f1_hz` / `f2_hz` | `150..1300` / `600..3200` | 第一/第二共振峰（绝对 Hz） | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | 谱倾斜 | — | — | ✓ |

`f1_hz` / `f2_hz` 把共振峰设到一个绝对 Hz 值，适合搭建跨条目一致的 Hz
连续统；`f1_scale` / `f2_scale` 按比例整体平移、保留每帧原有轮廓，更
适合保留自然语调形状的操控。两者不能对同一共振峰同时使用。

各参数的听感效果可在 [Demo 页面](https://n1r.github.io/ARIS_nsf/)直接
试听；参数含义与条件设计详见 [Manipulation 指南](docs/MANIPULATION_ZH.md)。

## 6. 浏览器工作台（Studio）

不想在命令行里拼 `--variant` 字符串的话，可以在浏览器里完成条件设计和
试听对比：

```bash
# 本地启动（自动打开 http://127.0.0.1:8765/）：
aris studio

# 在 Google Colab 或远程云服务器上启动（自动生成公网 Gradio 链接）：
aris studio --share
```

页面按模型自动生成参数滑杆，支持命名条件与连续统生成器（比如 F1 从
400 到 600 Hz 分五档，一键生成整组刺激）、一键渲染、原始/变体 A/B
试听，以及时间轴对齐的波形与语谱图对比；削波或共振峰到达模型边界时
会以红色标出。渲染结果保存在 `studio_output/` 下，与命令行 `manipulate`
的输出和元数据格式完全一致。

## 7. Python API 使用方式（Jupyter / Colab / 脚本）

除命令行外，ARIS 提供了高阶 Python 编程接口，可直接在 Jupyter Notebook 或脚本中无缝调用：

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
        "breathy:glottal_rd_scale=1.6",
    ],
)

# 6. 在 Notebook 中一键启动可视化 Web 工作台
aris.launch_studio(workspace=".", share=True)
```

## 8. 命令一览

```text
aris doctor             检查音频与训练依赖、CUDA 及 GPU 状态
aris fetch-corpus       下载 CMU ARCTIC 示例语料
aris split              切分连续录音
aris prepare            重采样、提取 F0、划分数据集
aris validate           检查数据完整性
aris init-experiment    生成训练实验目录
aris train              启动训练
aris controls           列出模型支持的操控参数
aris synthesize         用 checkpoint 重建录音
aris manipulate         生成操控刺激
aris studio             启动浏览器工作台（支持 --share 生成公网链接）
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
