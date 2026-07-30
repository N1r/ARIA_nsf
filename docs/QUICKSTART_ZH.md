# PhonLab-DDSP 中文快速开始

这套工作流把“原始录音 → 可审计数据 → 训练实验 → 结果”分开保存。不要直接把原始录音丢进旧脚本；那样很难知道重采样、F0 和训练/验证划分究竟发生了什么。

## 1. 安装

仅做数据准备与检查：

```bash
source scripts/project_env.sh
./scripts/setup_project_env.sh
.venv/bin/phonlab doctor
```

安装脚本使用 uv-managed Python，并将虚拟环境、下载和编译缓存全部限制在
当前仓库。训练版本固定为 PyTorch 2.4.1 + CUDA 12.4。GOLF 的 `torchlpc`
还需要完整 CUDA Toolkit 的 NVVM；仓库 Slurm 脚本会加载集群的
`CUDA/12.4.0` module。

## 2. 可选：切分连续录音

按静音边界切分访谈、朗读或连续语音：

```bash
phonlab split path/to/continuous_recordings segments/my_voice \
  --mode silence \
  --silence-threshold-db -40 \
  --min-silence-seconds 0.30 \
  --padding-seconds 0.05
```

按固定 2 秒窗口切分并保留末尾超过 0.25 秒的片段：

```bash
phonlab split path/to/recordings segments/fixed \
  --mode fixed --segment-seconds 2 --overlap-seconds 0
```

如果每条 WAV 都有同名 5 ms `.pv` F0，可以同步切分 sidecar，后续仍能使用
`--f0-method sidecar`：

```bash
phonlab split path/to/recordings segments/with_f0 \
  --mode silence --split-f0-sidecars --f0-hop-seconds 0.005
```

输出中的 `segments.csv` 保存原文件 SHA-256、每个片段的起止时间和输出路径；
`split.json` 保存完整切分参数。详细说明和 GUI 用法见
[切分工具与 GUI](TOOLS_AND_GUI_ZH.md)。

## 3. 准备自己的录音

输入目录可以包含多层子目录；WAV、FLAC、AIFF 和 OGG 均可。输出目录必须是新目录：

```bash
phonlab prepare path/to/my_recordings data/my_voice \
  --sample-rate 16000 \
  --f0-method autocorr \
  --f0-floor 60 \
  --f0-ceiling 500 \
  --validation-ratio 0.1 \
  --test-ratio 0.1
```

女高音、儿童语音或歌声应提高 `--f0-ceiling`；低沉男声可能需要降低 `--f0-floor`。这些参数属于实验定义，工具会写入 `dataset.json`。

如果语料已经为每个 `name.wav` 提供同名 `name.pv`（5 ms、Hz、0 表示无声）
轨迹，可避免安装或重新运行 WORLD：

```bash
phonlab prepare path/to/my_recordings data/my_voice --f0-method sidecar
```

输出包含：

```text
data/my_voice/
├── audio/          # 单声道、统一采样率 WAV
├── f0/             # 5 ms F0 轨迹，0 表示无声
├── manifest.csv    # 每条录音的划分、哈希和声学质检指标
└── dataset.json    # 参数、环境和整个数据集的 fingerprint
```

`--normalize-peak` 默认关闭，因为逐文件峰值归一化会消除录音间的强度差异。只有当响度不是研究变量时才显式启用。

## 4. 质检和可视化

```bash
phonlab validate data/my_voice
phonlab inspect data/my_voice
phonlab parameters data/my_voice
```

浏览器打开 `data/my_voice/report.html`。报告提供时长/F0 分布、削波和无有声段
警告、逐条试听以及完整 provenance；`parameters.csv` 可直接交给 R、Python
或表格软件。FFT 自相关后端容易安装、明确可复现，适合作为完整流程的基线。
对 F0 精度敏感的正式分析应抽样对照 Praat 或其他可信方法。若确需 WORLD，
可用项目内缓存安装可选依赖：

```bash
source scripts/project_env.sh
uv sync --extra audio --extra train --extra dev --extra world
```

之后明确使用 `--f0-method pyworld`，并在 Praat 或其他可信工具上抽样核对。

## 5. 建立实验

```bash
phonlab init-experiment data/my_voice experiments/my_voice_golf \
  --model golf \
  --batch-size 32 \
  --max-steps 40000 \
  --f0-min 60 \
  --f0-max 500

phonlab train experiments/my_voice_golf --dry-run
phonlab submit-job experiments/my_voice_golf --confirm
```

可选模型：

- `golf`：GOLF 声门流/LPC 源滤波器模型，默认选择。
- `ddsp`：谐波加噪声 DDSP 基线。
- `aria-golf`：研究级 ARIA-GOLF 解码器；建议先用前两者跑通，再使用它。

每个实验目录包含 `config.yaml`、`experiment.json`、`train.sh` 和可编辑的
`train.slurm`。`submit-job` 调用 Slurm；不要在登录节点直接运行非 dry-run
训练。真正训练前会重新计算 dataset fingerprint；数据被替换时会拒绝继续，
避免“同名数据、不同结果”。可用以下命令检查作业：

```bash
phonlab job-status JOB_ID --json
phonlab job-log experiments/my_voice_golf JOB_ID --tail-lines 200
```

## 6. Loss、重建与 manipulation

训练结束后先检查 loss、validation 和学习率。下列 checkpoint 推理也必须放入
GPU Slurm 作业；可使用 `phonlab init-postprocess` 自动生成作业包：

```bash
phonlab metrics experiments/my_voice_golf

phonlab init-postprocess \
  experiments/my_voice_golf \
  experiments/my_voice_golf/runs/checkpoints/last.ckpt \
  experiments/my_voice_golf/postprocess \
  --semitones -4 4

# 使用上一条命令打印的 “Post-processing job” 路径
phonlab submit-job PRINTED_JOB_BUNDLE_PATH --confirm
```

作业包会执行等价于以下的 held-out 重建、原音对照和有声 F0 操控：

```bash
phonlab synthesize EXPERIMENT CHECKPOINT OUTPUT/reconstruction

phonlab compare data/my_voice OUTPUT/reconstruction \
  --output OUTPUT/comparison.html

phonlab manipulate EXPERIMENT CHECKPOINT OUTPUT/manipulations \
  --semitones -4 4 --baseline OUTPUT/reconstruction \
  --report OUTPUT/manipulation.html
```

`comparison.html` 可并排试听原音和重建音，`manipulation.html` 增加多个半音
条件。它们没有随机化、盲法或响度校准，因此是质检页面，不能替代正式知觉
实验。

## 7. 图形界面

```bash
phonlab gui
```

浏览器工作台覆盖语料获取、切分、参数/F0、质检、实验、loss、Slurm 作业和
checkpoint manipulation。它只绑定 `127.0.0.1`；训练仅在卡片 6 明确确认
后提交给 Slurm。在远程集群使用时，以 SSH 端口转发访问，不要公开暴露服务。

## 8. 解释结果时的最低要求

- 保存 train/validation/test 划分，不根据测试集试听结果改模型。
- 报告 F0 提取器、上下界、采样率、随机种子、模型配置和 checkpoint。
- 把“模型控制参数”与“可直接解释的生理/构音量”区分开。可微参数的可控性不自动证明其语音学效度。
- 至少人工检查随机样本、异常值和无声段；不要只报告平均损失。
