# ARIS

**简体中文** | [English](README_EN.md)

## 0. 简介

ARIS（Analytic Resonance for Interpretable Synthesis）是一个为语音学研究者
准备的可微分析合成工具：用几十分钟的单说话人录音训练一个 DDSP/GOLF 声码器，
然后在其余参数保持不变的前提下，单独改变 F0、能量、噪声、声源形状（`R_d`）
或共振峰（F1/F2、谱倾斜），批量生成成对的实验刺激。

- 试听 Demo：<https://n1r.github.io/ARIS_nsf/>

不需要专门的计算设备：训练在一张普通游戏显卡（如 RTX 4060）上即可完成，
几十分钟数据约训练数小时；重建与刺激生成不需要显卡，普通电脑的 CPU 就能运行。

## 1. 安装依赖

先说明运行环境：以下命令在 **Linux 或 macOS 的终端**里执行。
如果你用的是 Windows，建议先安装
[WSL](https://learn.microsoft.com/zh-cn/windows/wsl/install)（微软官方的
Linux 子系统，一条 `wsl --install` 命令即可装好），然后在 WSL 的终端里
按下面的步骤操作。

你只需要机器上有 [uv](https://docs.astral.sh/uv/)（一个单文件的 Python
环境管理器）。在仓库根目录执行：

```bash
source scripts/project_env.sh      # 进入项目环境
./scripts/setup_project_env.sh     # 一键安装全部依赖（装在仓库目录内，不影响系统）
.venv/bin/aris doctor              # 检查音频与训练依赖是否就绪
```

装好之后，所有命令都通过 `.venv/bin/aris` 调用；不确定用法时，
每个命令都支持 `--help`。

**想先听听效果？** 我们提供了一个训练好的模型（普通话女声，16 kHz）
和配套示例数据，不用训练就能直接体验重建和参数操控：

```bash
curl -LO https://github.com/N1r/ARIS_nsf/releases/download/v0.1.0/aris_f024_demo.tar.gz
tar -xzf aris_f024_demo.tar.gz    # 在仓库根目录解压，得到 demo_f024/

.venv/bin/aris synthesize demo_f024/experiment \
  demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_recon
.venv/bin/aris manipulate demo_f024/experiment \
  demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_stimuli \
  --variant 'f1_up:f1_scale=1.2'
```

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
   再用 `--f0-method sidecar` 读入。
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
3. 在 Slurm 集群上，`init-experiment` 已生成可直接 `sbatch` 提交的
   `train.slurm`；集群参数（partition、GPU 类型等）可用 `init-experiment`
   的选项调整，提交前按你的集群补上 CUDA module。没有集群则忽略它。

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
| `f1_scale` / `f2_scale` | `0.7..1.3` | 第一/第二共振峰 | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | 谱倾斜 | — | — | ✓ |

各参数的听感效果可在 [Demo 页面](https://n1r.github.io/ARIS_nsf/)直接
试听；参数含义与条件设计详见 [Manipulation 指南](docs/MANIPULATION_ZH.md)。

## 6. 命令一览

```text
aris doctor             检查音频与训练依赖是否就绪
aris fetch-corpus       下载 CMU ARCTIC 示例语料
aris split              切分连续录音
aris prepare            重采样、提取 F0、划分数据集
aris validate           检查数据完整性
aris init-experiment    生成训练实验目录
aris train              启动训练
aris controls           列出模型支持的操控参数
aris synthesize         用 checkpoint 重建录音
aris manipulate         生成操控刺激
```

## 7. 引用

机器可读的引用信息见 [CITATION.cff](CITATION.cff)。

ARIS 的声码器实现源自 GOLF：

- C.-Y. Yu and G. Fazekas, "Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis," *Interspeech 2024*. DOI: `10.21437/Interspeech.2024-1187`
- C.-Y. Yu and G. Fazekas, "Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables," *ISMIR 2023*. DOI: `10.5281/zenodo.10265377`

更早的方法基础是可微 DSP 与神经源–滤波器模型：

- J. Engel, L. Hantrakul, C. Gu, and A. Roberts, "DDSP: Differentiable Digital Signal Processing," *ICLR 2020*. arXiv: `2001.04643`
- X. Wang, S. Takaki, and J. Yamagishi, "Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis," *IEEE/ACM TASLP*, 2020. arXiv: `1904.12088`

代码以 MIT 协议发布，见 [LICENSE](LICENSE)。

## 8. 联系

遇到问题或有建议，欢迎提 [Issue](https://github.com/N1r/ARIS_nsf/issues)，
也可以邮件联系 <dingyr@hum.leidenuniv.nl>（Leiden University）。
