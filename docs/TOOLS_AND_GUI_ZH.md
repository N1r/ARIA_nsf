# 音频切分工具与本地 GUI

## 切分 CLI

`phonlab split`（完整名称 `phonlab split-audio`）接收单个音频文件或递归目录，
把结果写入一个全新的输出目录。已有输出不会被覆盖；任一输入失败时，临时目录
会被清理，不留下看似完整的半成品。

### 静音边界模式

```bash
phonlab split recordings/ segments/silence \
  --mode silence \
  --silence-threshold-db -40 \
  --min-silence-seconds 0.30 \
  --padding-seconds 0.05 \
  --min-duration-seconds 0.25 \
  --max-duration-seconds 15
```

- `--silence-threshold-db`：20 ms 窗口的 RMS dBFS 阈值；录音底噪较高时需要提高。
- `--min-silence-seconds`：只有达到此长度的静音才分开相邻话语。
- `--padding-seconds`：在检测到的有声范围两端保留上下文。
- `--max-duration-seconds`：连续活动过长时再等长切开，避免异常长样本。

静音检测是数据整理工具，不是语音活动检测研究模型。先对少量代表性录音试听，
再固定阈值处理完整语料。

### 固定窗口模式

```bash
phonlab split recordings/ segments/fixed \
  --mode fixed \
  --segment-seconds 2 \
  --overlap-seconds 0.5 \
  --min-duration-seconds 0.25
```

默认保留满足最短时长的末尾部分；加 `--discard-tail` 可只保留完整窗口。
重叠窗口会让同一语音内容进入多个片段。划分 train/validation/test 前应考虑
说话人和原始录音泄漏问题，不能把高度重叠片段随机分到不同集合。

### 同步切分 F0

```bash
phonlab split recordings/ segments/with_f0 \
  --mode silence \
  --split-f0-sidecars \
  --f0-hop-seconds 0.005

phonlab prepare segments/with_f0/audio data/my_voice --f0-method sidecar
```

启用后，每个源音频必须有同名 `.pv`。工具验证 F0 非负、有限且总时长与音频
相差不超过 100 ms，然后依据片段时间同步写出 `.pv`。`segments.csv` 的
`f0_path` 字段记录对应轨迹，并保存原始 `.pv` 的 SHA-256。

### 输出契约

```text
segments/
├── audio/
│   ├── source-id__0001__000000000-000002000.wav
│   └── source-id__0001__000000000-000002000.pv  # 可选
├── segments.csv
└── split.json
```

文件名包含稳定的源路径摘要、片段序号和毫秒起止时间。CSV 还保存源文件
SHA-256、精确秒数、采样率和样本数；JSON 保存全部算法参数和被跳过的纯静音文件。

## 本地浏览器工作台

```bash
source scripts/project_env.sh
.venv/bin/phonlab gui
```

默认地址为 `http://127.0.0.1:8765/`。页面按顺序提供：

0. 官方 CMU ARCTIC 示例语料的固定下载、校验和 30–60 分钟子集；
1. 固定窗口或静音边界切分，包括 `.pv` 同步切分；
2. 数据准备和 F0 方法选择；
3. 声学参数导出、数据质检、可视化和逐条试听；
4. GOLF、DDSP 或 ARIA-GOLF 实验及可配置 Slurm 资源；
5. train/validation loss、学习率和 NaN/Inf 检查；
6. 需要显式确认的 Slurm 提交、状态、日志和取消；
7. checkpoint 重建、F0 manipulation 和试听报告的 GPU 作业包。

GUI 和 CLI 调用同一组 Python 函数，所以输出格式、校验和 provenance 相同。
完成一步后，工作台会把路径自动填入下一步的空字段。网页请求本身不在登录
节点运行神经网络；卡片 6 只有在使用者勾选确认后才调用 `sbatch`，并可用
Job ID 查询终态和有界日志。取消作业必须输入 `CANCEL`。

完整的可复现示例及每张卡片的对应 CLI 见
[CMU ARCTIC 完整流程](CMU_ARCTIC_PIPELINE_ZH.md)。

### 远程集群

GUI 有本地文件读写能力，因此拒绝绑定公网地址。在工作站建立 SSH 隧道：

```bash
ssh -L 8765:127.0.0.1:8765 USER@CLUSTER
```

然后在集群登录 shell 运行：

```bash
source scripts/project_env.sh
.venv/bin/phonlab gui --no-browser
```

工作站浏览器打开 `http://127.0.0.1:8765/`。关闭终端中的 GUI 使用
`Ctrl+C`。
