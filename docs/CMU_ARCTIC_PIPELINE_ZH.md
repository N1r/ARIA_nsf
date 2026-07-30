# CMU ARCTIC SLT：30 分钟语料到 F0 manipulation

本文给出一个可以审计的完整示例：固定下载 CMU ARCTIC 的单说话人英语
SLT 语料，选取约 30 分钟，切分、提取 F0、建立训练实验，经 Slurm GPU
训练后重建测试集，并生成基频操控和试听报告。

这是一条复现与工具验收流水线，不是模型科学有效性的证明。尤其要区分：
“程序成功运行”“重建听起来合理”和“某个声学或生理假设得到验证”是三件
不同的事。

## 当前示例状态

真实示例根目录是 `artifacts/cmu_arctic_slt_demo`。以下结果已由实际运行写入
磁盘并通过机械验收：

| 项目 | 已记录结果 |
|---|---:|
| 固定语料 | CMU ARCTIC 0.95，speaker `slt` |
| 选中原始话语 | 606 条 |
| 选中语音时长 | 1800.87525 秒 |
| 静音切分结果 | 633 条 |
| 去静音训练音频 | 1649.76 秒；train/validation/test = 507/63/63 |
| F0 方法 | `autocorr`，70–450 Hz |
| 数据集指纹 | `b321f4242fcd2831d42409f9fac4a81128341797080d9a6e8802008c5df441c6` |
| 实验配置 | GOLF，400 steps，batch size 8，4 workers |
| Slurm 证据 | Job `4558227`，`COMPLETED`，exit `0:0`，`node887` |
| GPU / 用时 | NVIDIA L4 23034 MiB；总计 `00:03:24` |
| Loss | train 297.72 → 5.52；validation 6.675 → 6.036；0 个 NaN/Inf |
| checkpoint | `last.ckpt`；SHA-256 `d7d6556459e46dfbf3b270ef05a3e9cfbf5ffa03e55d2a5ae535378e84e258c0` |
| 推理产物 | 63 条重建；−4/+4 半音各 63 条 |
| 输出 F0 抽查 | 20 条中位比值 0.7925 / 1.2585；理论值 0.7937 / 1.2599 |

这些数字证明约定的工程闭环实际执行成功，不代表 400 步模型已经充分收敛。

## 1. 语料来源、固定版本与许可

- 项目主页：[CMU ARCTIC Speech Synthesis Databases](http://festvox.org/cmu_arctic/)
- 本示例固定的官方归档：
  `http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2`
- 精确大小：`119914432` bytes
- SHA-256：
  `9fddec16fbfbfb7d4989dff0fe77ccbe31f80b07b57be49d09994aa7a67d6dba`

归档内的 `COPYING` 允许免费使用、复制、修改和再许可，包括商业用途，但
要求保留版权声明、条件和免责声明，明确标记修改，且不得删除原作者姓名。
它不是一个可以只凭名称推断条款的 SPDX 标识；发布衍生数据或模型前，应
保留并重新阅读本地原文：

```text
artifacts/cmu_arctic_slt_demo/corpus/extracted/cmu_us_slt_arctic/COPYING
```

`fetch-corpus` 会校验归档的精确字节数和 SHA-256，安全解包，并按官方
`etc/txt.done.data` 顺序选取确定性前缀。下载中断时保留 `.part` 文件供续传；
散列不符时停止，不会把不明文件当作正确语料使用。

## 2. 只在项目内建立 uv 环境

所有命令都应从仓库根目录运行。先加载项目环境，再让 uv 安装受管理的
Python 3.11、依赖和命令行程序：

```bash
cd /path/to/phonlab-ddsp  # 替换为本仓库实际位置
source scripts/project_env.sh
./scripts/setup_project_env.sh
.venv/bin/phonlab doctor
```

`scripts/project_env.sh` 把可变内容限制在当前仓库：

| 内容 | 项目内位置 |
|---|---|
| Python 虚拟环境 | `.venv/` |
| uv 下载与受管理 Python | `.cache/uv/` |
| pip、XDG、Torch、Numba、Matplotlib 缓存 | `.cache/` 下对应目录 |
| Hugging Face 缓存 | `.cache/huggingface/` |
| 实验和下载产物 | `artifacts/` |

每次新开终端都先执行 `source scripts/project_env.sh`。不要另建用户级
Conda 环境，不要用系统 `pip install` 补包；否则依赖和缓存可能逃出项目，
也会削弱复现性。可用下列只读命令检查空间占用：

```bash
du -sh .venv .cache artifacts
```

## 3. GUI：按 0–7 顺序使用

启动只绑定本机回环地址的工作台：

```bash
source scripts/project_env.sh
.venv/bin/phonlab gui
```

若程序运行在远程登录机，先从自己的电脑建立 SSH 隧道，再在远端用
`--no-browser` 启动；不要把 GUI 绑定到公共网卡：

```bash
# 在自己的电脑运行，替换 USER 和 LOGIN_HOST
ssh -L 8765:127.0.0.1:8765 USER@LOGIN_HOST

# 在远端仓库根目录运行
source scripts/project_env.sh
.venv/bin/phonlab gui --no-browser
```

然后在自己的浏览器打开 `http://127.0.0.1:8765/`。八张卡片对应：

0. **获取可复现示例语料**：下载或复用官方归档，校验后固定 30–60 分钟。
1. **音频切分**：按静音或固定窗口输出独立 WAV 和切分清单。
2. **准备数据集**：重采样、提 F0、确定性划分，并生成数据指纹。
3. **质检与试听**：浏览时长、F0、削波等指标并逐条试听。
4. **建立训练实验**：生成配置、provenance 和 Slurm 启动器；此步不训练。
5. **Loss 与训练指标**：绘制 train/validation loss、学习率并报告 NaN/Inf。
6. **Slurm 作业中心**：显式确认后提交，查询状态/日志；取消时需输入
   `CANCEL`。
7. **推理与 Manipulation**：从 checkpoint 生成 GPU 后处理作业包，不在
   当前网页请求中直接运行神经网络推理。

GUI 和 CLI 调用同一套 Python 核心；GUI 不是另一条计算路径。训练实验由
卡片 4 创建，真正提交则在卡片 6 明确确认。

## 4. CLI：从下载到可提交实验

以下命令假定目标目录尚不存在，适用于从空白工作区复现当前示例。已有真实
结果时不要覆盖；请选择新的 `PIPELINE_ROOT` 保存新运行。

```bash
source scripts/project_env.sh
PIPELINE_ROOT=artifacts/cmu_arctic_slt_demo
```

### 4.1 下载并固定约 30 分钟

```bash
.venv/bin/phonlab fetch-corpus \
  "$PIPELINE_ROOT/corpus" \
  --target-minutes 30 \
  --max-minutes 60 \
  --silence-gap 0.35
```

该命令输出语料名称、选中条数、语音时长、归档散列、`continuous.wav` 和
`corpus.json` 的位置。已有官方归档时可避免重复联网：

```bash
.venv/bin/phonlab fetch-corpus \
  "$PIPELINE_ROOT/corpus" \
  --archive path/to/cmu_us_slt_arctic-0.95-release.tar.bz2 \
  --target-minutes 30 \
  --max-minutes 60 \
  --silence-gap 0.35
```

`continuous.wav` 是把选中话语按官方顺序连接、并插入 0.35 秒静音得到的
切分输入；`selected_duration_s` 统计所选原始 WAV 的总时长，不含后加的
话语间静音。

### 4.2 按静音切分

```bash
.venv/bin/phonlab split \
  "$PIPELINE_ROOT/corpus/continuous.wav" \
  "$PIPELINE_ROOT/segments" \
  --mode silence \
  --silence-threshold-db -45 \
  --min-silence-seconds 0.20 \
  --padding-seconds 0.04 \
  --min-duration-seconds 0.25 \
  --max-duration-seconds 12 \
  --sample-rate 16000
```

当前固定输入得到 633 个片段。边界、源文件散列和时间戳记录在
`segments/segments.csv` 与 `segments/split.json`，不要只保留切出的 WAV。

### 4.3 提取参数并准备数据集

```bash
.venv/bin/phonlab prepare \
  "$PIPELINE_ROOT/segments/audio" \
  "$PIPELINE_ROOT/dataset" \
  --sample-rate 16000 \
  --f0-method autocorr \
  --f0-floor 70 \
  --f0-ceiling 450 \
  --validation-ratio 0.10 \
  --test-ratio 0.10 \
  --seed 20260730 \
  --min-duration 0.25

.venv/bin/phonlab validate "$PIPELINE_ROOT/dataset" --json
.venv/bin/phonlab inspect "$PIPELINE_ROOT/dataset"
.venv/bin/phonlab parameters "$PIPELINE_ROOT/dataset"
```

`manifest.csv` 是可复现数据契约；`parameters.csv` 是稳定、便于统计软件
读取的 13 列参数导出表。它们逐条记录 train/validation/test 划分、相对
音频与 F0 路径、SHA-256、采样率、时长、peak、RMS dBFS、clipping、DC
offset、F0 方法、median F0 和 voiced fraction。逐帧 F0 位于
`dataset/f0/*.f0.txt`；`dataset/report.html` 提供汇总、异常提示和试听。

本示例明确写 `--f0-method autocorr`，而不是 `auto`。`pyworld` 是可选依赖，
在部分系统上难以安装；FFT 加速的自相关后端只依赖本项目已有环境，保留
无声帧为 0，足以跑通训练接口。两种算法的数值不能假定相同，因此跨机器
复现时不要让 `auto` 因可用依赖不同而悄悄换算法。若已有经过审计的同名
`.pv` 文件，也可另建数据集并使用 `--f0-method sidecar`。

### 4.4 建立 400-step GOLF 实验

```bash
.venv/bin/phonlab init-experiment \
  "$PIPELINE_ROOT/dataset" \
  "$PIPELINE_ROOT/experiment" \
  --model golf \
  --batch-size 8 \
  --max-steps 400 \
  --seed 20260730 \
  --f0-min 70 \
  --f0-max 450 \
  --workers 4

# 只验证将要执行的命令，不做训练
.venv/bin/phonlab train "$PIPELINE_ROOT/experiment" --dry-run
```

`experiment.json` 固定数据指纹与配置散列；训练前会再次核对指纹。400 steps
是流水线验收预算，不等于充分收敛，更不应作为跨模型质量比较的默认预算。

## 5. GPU 训练必须提交 Slurm

不要在登录节点直接运行非 `--dry-run` 的 `phonlab train`。当前 ALICE
示例用专门脚本完成训练、指标、重建、操控和最终验收：

```bash
source scripts/project_env.sh
sbatch --parsable slurm/cmu_arctic_slt_pipeline.slurm
```

记下 `sbatch` 返回的数字。已验收运行返回 `4558227`；新运行必须使用自己的
ID，不能把这个历史 ID 当成刚提交的作业：

```bash
PIPELINE_JOB_ID=4558227  # 仅用于查询本文记录的已完成运行
squeue -j "$PIPELINE_JOB_ID" -o "%.18i %.12P %.24j %.10T %.10M %.20R"
sacct -j "$PIPELINE_JOB_ID" \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList
tail -f "$PIPELINE_ROOT/experiment/slurm-$PIPELINE_JOB_ID.log"
```

`squeue` 中作业消失不等于成功；以 `sacct` 的终态、日志、checkpoint 和
`training_job.json` 四者交叉核对。作业脚本当前针对 ALICE：
`partition=testing`、`gpu:l4:1`、`ALICE/default` 与 `CUDA/12.4.0`。换集群
时只按当地文档调整 `#SBATCH` 和 module 行，并保留项目环境、数据指纹检查
及 provenance 输出。

作业成功后可再次运行机器验收：

```bash
.venv/bin/python tools/check_complete_pipeline.py "$PIPELINE_ROOT"
```

只有输出 `PHONLAB_COMPLETE_PIPELINE_OK` 才表示约定的文件和数值检查全部通过；
它仍不代表感知质量或研究假设已经验证。

## 6. Loss、checkpoint、重建与 manipulation

完整 Slurm 脚本已经顺序执行本节命令。这里单独列出，便于理解和构建其他
Slurm 作业；这些推理命令同样应在 GPU 节点上运行。

### 6.1 Loss 与学习率

```bash
.venv/bin/phonlab metrics \
  "$PIPELINE_ROOT/experiment" \
  --output "$PIPELINE_ROOT/metrics.html"
```

打开 `metrics.html` 可查看 train loss、validation loss、learning rate 和
其他 Lightning CSV 序列。发现 NaN/Inf 时命令返回非零状态；不能因为作业
仍生成了 checkpoint 就忽略非有限值。原始表位于
`experiment/runs/metrics/version_*/metrics.csv`。

### 6.2 基线重建和原音对照

```bash
CHECKPOINT="$PIPELINE_ROOT/experiment/runs/checkpoints/last.ckpt"

.venv/bin/phonlab synthesize \
  "$PIPELINE_ROOT/experiment" \
  "$CHECKPOINT" \
  "$PIPELINE_ROOT/reconstruction"

.venv/bin/phonlab compare \
  "$PIPELINE_ROOT/dataset" \
  "$PIPELINE_ROOT/reconstruction" \
  --output "$PIPELINE_ROOT/comparison.html"
```

### 6.3 有声 F0 条件的 ±4 半音操控

```bash
.venv/bin/phonlab manipulate \
  "$PIPELINE_ROOT/experiment" \
  "$CHECKPOINT" \
  "$PIPELINE_ROOT/manipulations" \
  --semitones -4 4 \
  --baseline "$PIPELINE_ROOT/reconstruction" \
  --report "$PIPELINE_ROOT/manipulation.html"
```

`manipulation.html` 把 held-out 原音、零偏移重建、−4 和 +4 半音结果并排。
`manipulations/manipulation.json` 记录 checkpoint SHA-256、数据指纹、半音值、
F0 比例和输出目录。输出目录已存在时工具会停止，以防混合两次运行。

若只想在登录节点准备一个独立的 GPU 后处理作业包，而不直接推理：

```bash
.venv/bin/phonlab init-postprocess \
  "$PIPELINE_ROOT/experiment" \
  "$CHECKPOINT" \
  "$PIPELINE_ROOT/postprocess" \
  --semitones -4 4 \
  --partition gpu-short \
  --gres gpu:l4:1 \
  --time 00:30:00 \
  --cpus 4 \
  --memory 24G
```

命令会打印作业包路径；将其中的 `train.slurm` 提交给 `sbatch`。也可在 GUI
卡片 7 生成作业包，再在卡片 6 勾选确认并提交。

## 7. 产物目录与试听

已验收运行的完整目录大致如下：

```text
artifacts/cmu_arctic_slt_demo/
├── corpus/
│   ├── cmu_us_slt_arctic-0.95-release.tar.bz2
│   ├── corpus.json
│   ├── continuous.wav
│   ├── extracted/cmu_us_slt_arctic/COPYING
│   └── selected/                         # 606 WAV
├── segments/
│   ├── audio/                            # 633 WAV
│   ├── segments.csv
│   └── split.json
├── dataset/
│   ├── audio/                            # 633 WAV
│   ├── f0/                               # 633 F0 tracks
│   ├── manifest.csv
│   ├── parameters.csv
│   ├── dataset.json
│   └── report.html
├── experiment/
│   ├── config.yaml
│   ├── decoder.yaml
│   ├── experiment.json
│   ├── train.sh
│   ├── train.slurm
│   ├── slurm-4558227.log
│   └── runs/
│       ├── metrics/version_0/metrics.csv
│       └── checkpoints/last.ckpt
├── training_job.json
├── metrics.html
├── reconstruction/                       # 63 WAV
├── comparison.html
├── manipulations/
│   ├── manipulation.json
│   ├── pitch_minus_4st/                   # 63 WAV
│   └── pitch_plus_4st/                    # 63 WAV
└── manipulation.html
```

最直接的四个入口是：

- `dataset/report.html`：数据质检与逐条原始切片试听；
- `comparison.html`：held-out 原音与重建并排试听；
- `manipulation.html`：原音、重建及多个 F0 条件并排试听；
- `metrics.html`：loss、validation 和学习率。

HTML 内使用相对音频路径。复制到自己的电脑时，应保留相关 HTML、WAV 和目录
层级；只复制一个 HTML 文件会导致播放器找不到音频。

## 8. 常见故障

| 现象 | 检查与处理 |
|---|---|
| 下载中断 | 保留同目录 `.part`，原命令可续传。不要手工把 `.part` 改名为正式归档。 |
| 大小或 SHA-256 不符 | 停止使用该文件，记录来源并移到单独隔离目录；不要关闭校验。 |
| `output already exists` | 工具把输出视为不可变运行；改用新目录或先把旧目录完整归档，避免混写。 |
| `pyworld` 缺失 | 本示例不需要它；明确使用 `--f0-method autocorr`。 |
| 大量无声 F0 或跳八度 | 在 `report.html` 试听并检查 70–450 Hz 是否适合说话人；调整范围时建立新数据集并记录新指纹。 |
| dataset fingerprint changed | 数据或 manifest 已改变；不要绕过检查，应从该数据重新建立一个新实验。 |
| `sbatch`/`squeue` 不可用 | 当前主机不是带 Slurm 客户端的集群入口，或 module/path 未配置；不要改为在登录节点直接训练。 |
| 作业长期 `PENDING` | 用 `squeue` 查看 Reason，核对 partition、GRES、时限和节点限制。 |
| CUDA、NVVM 或 `torchlpc` 错误 | 查看作业日志，确认作业获得 GPU 且已加载 `CUDA/12.4.0`；不要在无 GPU 登录进程中调试训练。 |
| GPU OOM | 建立新的实验并减小 batch size，例如 8 改为 4；保留旧实验以便比较。 |
| 没有 `val_loss` | 检查训练是否完成至少一个 epoch、验证集是否非空及 CSV 日志；本配置每个 epoch 验证一次。 |
| metrics 报 NaN/Inf | 把运行视为失败候选，先定位最早异常 step，再检查音频/F0、学习率和梯度；不要只挑正常曲线展示。 |
| 没有 checkpoint | 先读 `training_job.json` 的 `state/stage/exit_code`，再查 `sacct` 和 `slurm-JOB_ID.log`。 |
| 远端报告打不开 | 使用本地 GUI + SSH 隧道，或连同引用的音频目录一起复制到本机。 |

## 9. 科研解释边界

本工具的 pitch manipulation 把**有声帧的 F0 条件**乘以
`2^(semitones/12)`，无声帧保持 0。因而 +4 半音约为原 F0 的 1.2599 倍，
−4 半音约为 0.7937 倍。这定义了输入给模型的条件变化，不保证输出波形中的
实际 F0 恰好按同一比例变化。

必须保留以下边界：

- `autocorr` F0 是算法估计，不是声带振动的无误差真值；错误、倍频和半频要
  通过试听及其他测量交叉核对。
- 改变 F0 条件后，神经合成器可能同时改变音色、能量、噪声或滤波特征。
  因此结果不能直接解释为隔离的声带或声道生理干预。
- 重建 loss 是优化目标，不是自然度、可懂度、身份保持、语音学对立或感知
  效应的充分证据。
- SLT 是一个英语单说话人示例；即使流水线成功，也不能推出对其他说话人、
  语言、音域或录音条件的泛化。
- 400-step 运行用于工程闭环和故障发现，不是收敛性研究。科研报告应另设
  训练预算、重复种子、独立测试和听辨/声学验证。

换成自己的录音时，可以从 `split` 开始复用同一流程，但应先取得合适的使用
授权，保存原始音频与采集说明，并把语言、说话人、设备、环境、纳排规则和
任何预处理写入研究 provenance。
