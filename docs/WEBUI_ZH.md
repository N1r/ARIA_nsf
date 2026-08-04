# PhonLab-DDSP WebUI 使用指南

[KNOWN] 这个 WebUI 面向不希望编写命令行参数的语音学使用者；它覆盖已有项目扫描、图形化 manipulation 条件设计、Slurm 作业生成与提交、自动状态和日志、结果试听及导出。它与 `.venv/bin/phonlab` 共用同一套校验和作业后端，不是另一个训练实现，也不需要另外安装 Streamlit。

[KNOWN] 浏览器页面本身不运行神经网络。训练、checkpoint 重建和 manipulation 推理必须通过 Slurm 进入 GPU 计算节点；登录节点只运行轻量页面、读取元数据、生成作业包和查询调度状态。

## 1. 使用前准备

[KNOWN] WebUI 从一个已经建立的 experiment 开始时，至少需要：

- [KNOWN] 包含 `experiment.json` 的实验目录；
- [KNOWN] 与该实验匹配的 checkpoint，例如实验内 `runs/checkpoints/last.ckpt`；
- [KNOWN] 一个尚不存在的新输出目录；系统拒绝把新运行混写进旧结果；
- [KNOWN] 能使用 `sbatch`、`squeue` 和 `sacct` 的集群登录环境。

[KNOWN] 不要仅凭 checkpoint 文件名判断模型或数据。`experiment.json` 提供模型声明和 dataset fingerprint，作业包还会记录 checkpoint SHA-256。

### 用 uv 建立仓库内环境

[KNOWN] 在仓库根目录运行 `scripts/setup_project_env.sh`，会由 uv 建立 `.venv/`，并把 Python、下载和编译缓存放在本仓库的 `.cache/`；它不会要求激活虚拟环境。入口机需要先有一个可执行的 uv，若 uv 不在 `PATH`，可通过 `UV_BIN=/absolute/path/to/uv` 指定。

```bash
./scripts/setup_project_env.sh
.venv/bin/phonlab doctor
```

[KNOWN] 此后所有 PhonLab 和 Python 命令都应显式使用 `.venv/bin/phonlab` 或 `.venv/bin/python`，不要依赖 `source .venv/bin/activate` 后的隐式环境。

## 2. 启动与安全连接

### 在有桌面的本机启动

```bash
.venv/bin/phonlab webui \
  --workspace /absolute/path/to/glof_ddsp \
  --host 127.0.0.1 --port 8765
```

[KNOWN] `webui` 是 `gui` 的易记别名；两者启动同一个工作台。`--workspace` 明确规定项目发现、结果读取和服务端导出的目录边界；省略时使用启动命令所在目录。默认页面地址是 `http://127.0.0.1:8765/`。终端保持运行；结束服务时在该终端按 `Ctrl+C`。

### 在远程集群启动

[KNOWN] 先在集群登录节点的仓库根目录启动 WebUI，禁止尝试绑定公网或局域网地址：

```bash
.venv/bin/phonlab webui \
  --workspace /absolute/path/to/glof_ddsp \
  --host 127.0.0.1 --port 8765 --no-browser
```

[KNOWN] 再从自己的电脑建立 SSH tunnel：

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@CLUSTER
```

[KNOWN] 最后在自己的浏览器打开 `http://127.0.0.1:8765/`。浏览器中填写的 experiment、checkpoint 和输出路径都是**集群上的路径**，不是个人电脑上的路径。

### 安全边界

- [KNOWN] 服务只允许绑定 `127.0.0.1`、`localhost` 或 `::1`；`0.0.0.0` 会被拒绝。
- [KNOWN] WebUI 没有面向公网的账号、权限隔离或 TLS 层，却具有当前 Unix 用户的本地文件读写能力，并可在明确确认后提交 Slurm 作业。
- [KNOWN] SSH tunnel 只把本机端口转发到登录节点；不要把本机转发端口再次共享，也不要在公共反向代理中发布这个页面。
- [KNOWN] 页面不会自动扫描整台机器。只选择自己有权处理的 experiment、checkpoint、作业包和输出目录。
- [KNOWN] 结果读取、服务端另存和 ZIP 缓存受 `--workspace` 边界限制；把 workspace 设为仓库根目录可以把这些服务器端读写保持在当前项目内。
- [INFERRED] 在共享账号或不可信目录中运行会放大误读文件、覆盖权限和作业归属风险，因此应使用个人集群账号与个人项目目录。

## 3. 从 experiment 到 manipulation

### 第一步：选择实验与 checkpoint

[KNOWN] 在“项目与实验路径”中可直接填写路径，也可在“现有项目”区域按实验、checkpoint、结果目录或数据集筛选仓库内已发现的项目，然后点击“填入路径”。扫描只是列出识别到的元数据目录，不会修改实验。

[KNOWN] 填写实验目录、checkpoint 和新结果目录后，点击“读取实验与模型能力”。页面会读取实验的模型声明，并按当前模型动态显示允许的控制、单位、默认值、范围和滑杆；不支持的参数不会出现在 condition builder 中。

[KNOWN] 页面显示的是 experiment 级声明。真正载入 checkpoint 后，GPU 推理还会再次检查 decoder 是否具有相应执行路径；不支持的控制会失败并写入日志，不会被静默忽略。

### 第二步：建立命名条件

[KNOWN] 一个条件由“唯一名称 + 一个或多个已启用的控制”组成。使用滑杆或旁边的数字框设值，再点击“加入条件表”；条件可在表中编辑或删除。条件名称也会用于输出目录，因此应采用简短 ASCII 名称，例如 `pitch_up`、`less_noise`、`rd_high` 或 `vowel_shift`。

[KNOWN] 可以为同一 checkpoint 建立多个单参数条件，也可以建立联合条件。例如：

| 条件名 | 控制 | 用途 |
|---|---|---|
| `pitch_down` | `pitch_semitones=-4` | [KNOWN] 降低有声帧 F0 条件 |
| `less_noise` | `noise_gain_db=-6` | [KNOWN] 降低随机源分支增益 |
| `rd_high` | `glottal_rd_scale=1.2` | [KNOWN] 改变 GOLF 声门源的 `R_d` 表位置 |
| `quiet_source` | `output_gain_db=-3`、`noise_gain_db=-6` | [KNOWN] 同时应用两个控制 |

[KNOWN] 参数范围是软件接受范围，不是已验证的语音学效应范围。若研究目标是解释单个参数，至少保留 baseline、单参数条件和必要的联合条件；不要只渲染一个多参数组合后把差异归因给其中某一项。

### 第三步：生成而不是立即提交

[KNOWN] 填写新的输出目录和 Slurm 资源后，先执行“生成后处理作业”一类操作。此时系统只生成可审计的作业包，不启动 GPU。

[KNOWN] 提交前应在页面确认以下内容：

- [KNOWN] experiment、checkpoint 和新输出目录是否正确；
- [KNOWN] 条件名称、每个控制值及模型能力是否正确；
- [KNOWN] partition、GPU GRES、时间、CPU、内存和排除节点是否符合本集群规则；
- [KNOWN] 作业包中的 dataset fingerprint 与 checkpoint SHA-256 是否已记录；
- [KNOWN] 输出目录没有包含旧运行结果。

### 第四步：显式确认并提交 Slurm

[KNOWN] 将页面返回的作业包目录带到“Slurm 作业中心”，勾选提交确认后再提交。页面会返回数字 Job ID；没有勾选确认不会调用 `sbatch`。

[KNOWN] 后处理作业在 GPU 节点依次完成 baseline reconstruction、原音/重建对照页、所有 manipulation 条件、试听页和训练 metrics 页面。不要在登录节点手动运行 checkpoint 推理来绕过队列。

### 第五步：查看状态与日志

[KNOWN] 提交后页面默认每 5 秒刷新一次状态与日志；也可关闭自动刷新或点击“立即刷新”。常见状态包括排队、运行、完成、失败和取消，日志区域只显示有界长度的末尾内容。

[KNOWN] 排队期间还没有完整 WAV 是正常现象。只有调度状态为完成且退出码为零，才应进入结果验收；状态完成也不能替代 clipping、文件数量和 provenance 检查。

## 4. 模型与可用控制

[KNOWN] 当前公开能力矩阵如下；范围两端均包含在内：

| 参数 | 范围；默认值 | DDSP | GOLF | ARIA-GOLF |
|---|---|:---:|:---:|:---:|
| `pitch_semitones` | `-36..36`；`0` | ✓ | ✓ | ✓ |
| `output_gain_db` | `-24..12` dB；`0` | ✓ | ✓ | ✓ |
| `noise_gain_db` | `-24..24` dB；`0` | ✓ | ✓ | ✓ |
| `glottal_rd_scale` | `0.5..2.0`；`1.0` | — | ✓ | ✓ |
| `f1_scale` | `0.7..1.3`；`1.0` | — | — | ✓ |
| `f2_scale` | `0.7..1.3`；`1.0` | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25`；`0` | — | — | ✓ |

[KNOWN] 普通 GOLF 的声道滤波器输出 LPC/全极点参数，没有可被可靠标记为独立 F1 或 F2 的公共控制柄。因此 WebUI 不应为普通 GOLF 显示 F1/F2 manipulation；显式 F1/F2 与谱倾斜只属于 ARIA-GOLF。

[KNOWN] `output_gain_db` 是最终波形增益，不是声门强度或生理量；`noise_gain_db` 是模型随机源分支增益，不等于经过独立校准的 HNR；`glottal_rd_scale` 是模型内部声门源控制，不是 EGG 或声门几何测量。

### ARIA-GOLF 的当前证据边界

[COMPUTED] ARIA-GOLF 已完成真实 Slurm checkpoint 验收：F024 上训练 400 steps，选用最佳 `val_loss=5.05988` 的 checkpoint，分别渲染 F1 `0.9/1.1`、F2 `0.9/1.1` 和 tilt `-0.1/+0.1` 六个隔离条件。每组 up/down 在 4/4 个 held-out WAV 上均不同，28 个结果 WAV 的 clipping 总数为 0；专项验收返回 `PHONLAB_ARIA_MANIPULATION_OK`，WebUI 试听、Range、下载、ZIP 和临时另存返回 `PHONLAB_WEBUI_OK`。

[KNOWN] 这个闭环证明软件实际加载 ARIA checkpoint、调用了解析声道控制路径、写出了可区分且无削波的 WAV；它不等于独立声学测量已经证明每条语音的观测 F1/F2 恰好移动 10%。正式研究仍应使用独立 formant 方法复测方向和幅度，并人工检查重建质量。机器报告位于 `.cache/f024-aria-validated-acceptance.json`，可用 `make verify-aria` 重跑。

## 5. 结果试听与 WAV 保存

[KNOWN] 成功的后处理输出通常包含：

```text
OUTPUT/
├── reconstruction/                 # baseline WAV
├── reconstruction.html             # 原音/重建对照
├── manipulations/
│   ├── manipulation.json            # 条件级 provenance
│   ├── CONDITION/
│   │   ├── RECORD_ID.wav
│   │   └── _render.json             # 渲染与 clipping 审计
│   └── ...
├── manipulation.html                # 条件试听页
└── metrics.html                     # loss/学习率/异常值
```

[KNOWN] GPU 作业完成时，WAV 已经保存在集群的 `OUTPUT` 目录；关闭浏览器不会删除它们。在“结果试听与 WAV 保存”中填写结果目录并点击“加载结果目录”，页面会校验 baseline、所有 condition、WAV 集合、渲染审计和 provenance 后才建立试听索引。

[KNOWN] 可以从下拉框选择当前 condition 与语料条目，也可用左右箭头逐条移动。页面并排显示 Baseline 和当前 Manipulation；“下载 baseline WAV”和“浏览器下载当前 WAV”分别保存两边的单文件，当前 manipulation 还提供“服务端另存”。

[KNOWN] 页面中的“浏览器下载当前 WAV”会把当前文件复制到浏览器的下载目录，不会移动或重命名服务器原文件。浏览器下载只得到所点的 WAV；需要连同来源记录保存时，应使用服务端另存或 ZIP。

[KNOWN] 在结果区域选择一条记录时，应先并排试听 original、baseline reconstruction 和各 manipulation 条件。original 与 reconstruction 的差异是模型重建误差；reconstruction 与 manipulation 的差异才是本次控制条件下的总变化。

[KNOWN] 浏览器试听适合人工质检和材料预览，不提供随机化、盲法、播放设备校准或被试响应记录，不能直接替代正式知觉实验。

## 6. 服务端另存 condition、ZIP 与复现记录

[KNOWN] “服务端另存单 WAV”把当前语料条目的当前 condition WAV 复制到指定的新目录；“服务端另存 condition”把当前 condition 的全部 WAV 复制到新目录。另存目录必须位于当前仓库工作区内且尚不存在，避免与旧导出混写。

[KNOWN] 每次服务端另存都会建立 `audio/CONDITION/...` 和 `provenance.json`，后者记录来源结果、condition 名称与完整控制值、checkpoint/dataset 摘要、所选 item、每个导出文件的相对路径、字节数和 SHA-256。这里的“另存 condition”是导出当前条件的音频与来源记录，不是把某个模型的滑杆配置无条件套到另一个模型。

[KNOWN] “生成下载 ZIP”打包**当前 condition 的全部 WAV**和 `provenance.json`，再交给浏览器下载；它不会把所有 conditions 自动装进同一个压缩包。若需要多个条件，应逐个选择并分别生成 ZIP，这也能避免条件标签在离线分析时混淆。

[KNOWN] 最低复现集合包括：

- [KNOWN] `experiment.json` 与所用配置；
- [KNOWN] 作业包中的 `job.json` 和 `train.slurm`；
- [KNOWN] `manipulation.json`；
- [KNOWN] 每个条件的 `_render.json`；
- [KNOWN] checkpoint 的 SHA-256 和 dataset fingerprint；
- [KNOWN] WAV、metrics 页面、Slurm Job ID、终态和日志；
- [KNOWN] 后续独立声学测量脚本及其版本。

[KNOWN] ZIP 在仓库 `.cache/webui_exports/` 中以唯一名称原子生成后再提供下载，内部同样使用 `audio/CONDITION/...` 加 `provenance.json` 的结构。它是便捷传输副本，不应被当成唯一存档；集群输出目录和浏览器下载副本可能在不同时间被移动，归档时应保存哈希和元数据，以便发现文件漂移。

[KNOWN] WebUI 的防误操作上限是每个 condition 或一次导出最多 10,000 个 WAV、单 WAV 最多 512 MiB、一次导出未压缩总量最多 2 GiB；超过限制时应在集群文件系统中设计分批分析，而不是绕过页面校验。

## 7. Provenance 与 clipping 怎么看

[KNOWN] 加载结果后，页面顶部汇总语料条目数、condition 数、可用 WAV 数和 clipping；“削波、来源与报告”区域显示 clipping 汇总、provenance，以及 manipulation、reconstruction 和 loss/metrics 报告链接。

[KNOWN] `manipulation.json` 记录模型、实验路径、dataset fingerprint、checkpoint 路径与 SHA-256，以及每个条件的完整控制值和 F0 scale。`_render.json` 记录运行时能力、decoder hook 调用次数、写出文件数、峰值、总样本数和 clipping 统计。

[KNOWN] `clipped_samples > 0` 表示保存前有样本超过 `[-1, 1]`，随后已被硬限制；这会引入失真。`clipped_fraction` 是被截断样本占比，不能把非零值简单解释为“只是声音更响”。

[KNOWN] 正式条件比较应优先要求每个条件 `clipped_samples == 0`。若出现 clipping，应降低输出增益、使用新输出目录并重新渲染全部相关条件；不要只对个别 WAV 事后归一化。

[KNOWN] pitch 通过数据侧的有声 F0 条件应用，因此 pitch 的完整记录以 `manipulation.json` 为准；不要因为某个 `_render.json` 中没有 pitch decoder hook 就判断音高控制未执行。

## 8. 常见错误

| 页面或日志信息 | 原因与处理 |
|---|---|
| 找不到 `experiment.json` | [KNOWN] 选择的是实验的父目录、run 子目录或普通数据目录；改为包含 `experiment.json` 的实验根目录。 |
| checkpoint 不存在 | [KNOWN] 路径拼写错误、训练尚未产出 checkpoint，或浏览器中误填了个人电脑路径；填写集群服务器路径。 |
| 输出目录已存在 | [KNOWN] 系统拒绝混写；为本次参数组合选择新的、可辨识目录，不要先删除不确定来源的旧结果。 |
| 参数不支持或越界 | [KNOWN] 先刷新模型能力；不要给 DDSP 使用 `R_d`，也不要给普通 GOLF 使用 F1/F2/tilt。把数值调整到页面显示的闭区间。 |
| 条件名无效或重复 | [KNOWN] 使用以字母或数字开头、只含 ASCII 字母、数字、`_`、`-` 的唯一名称。 |
| 无法连接页面 | [KNOWN] 确认 WebUI 终端仍运行、SSH tunnel 未断开且本地端口与远程启动端口一致；不要把服务改绑 `0.0.0.0`。 |
| `sbatch`、`squeue` 或 `sacct` 不可用 | [KNOWN] 当前 shell 不是配置好的集群入口，或调度命令未加载；不要因此在登录节点直接跑 GPU 推理。 |
| 作业长时间排队 | [KNOWN] 查看状态中的 reason，并核对 partition、GRES、时间和资源需求；排队不等于失败。 |
| 日志尚不存在 | [KNOWN] 作业可能还未启动，Slurm 尚未创建日志；等待调度后再刷新。 |
| CUDA、NVVM 或 `torchlpc` 错误 | [KNOWN] GPU 节点环境或 CUDA module 不匹配；保留完整日志并检查生成的 Slurm 脚本是否加载项目要求的 CUDA 环境。 |
| checkpoint hash changed | [KNOWN] 生成作业包后 checkpoint 被覆盖或替换；不要忽略校验，固定 checkpoint 后生成新作业包。 |
| runtime capability 不匹配 | [KNOWN] experiment 声明与实际 checkpoint decoder 不一致；核对 checkpoint 来源，不要要求系统跳过控制。 |
| 结果目录无法加载 | [KNOWN] 作业可能失败、尚未完成、选择了错误输出目录，或某个 condition 的 WAV 集合/审计元数据与 baseline 不一致；先检查终态、退出码和日志，不要绕过结果校验。 |
| 服务端另存目录被拒绝 | [KNOWN] 目录已存在、位于仓库外、位于源结果目录内或经过不安全的符号链接；选择仓库内且结果树之外一个尚不存在的新目录。 |
| clipping 非零 | [KNOWN] 至少一个条件超过 PCM 保存范围；降低 `output_gain_db` 或重新设计条件，并完整重渲染。 |
| ZIP 很大、达到上限或下载中断 | [KNOWN] 页面限制每次最多 10,000 个 WAV、单 WAV 512 MiB、总量 2 GiB；减少 condition 文件数并分批处理，或把集群输出目录作为主存档。 |

## 9. 非工程使用者的最短检查表

1. [KNOWN] 用 `.venv/bin/phonlab doctor` 确认项目环境。
2. [KNOWN] 通过 loopback 或 SSH tunnel 打开页面，绝不公开绑定。
3. [KNOWN] 选择 experiment 与固定 checkpoint，先读取模型能力。
4. [KNOWN] 使用唯一条件名，并为每个可解释因素保留单参数条件。
5. [KNOWN] 选择新输出目录，生成作业包并检查条件、哈希和 Slurm 资源。
6. [KNOWN] 显式确认后提交；GPU 只在计算节点运行。
7. [KNOWN] 等待 `COMPLETED` 和零退出码，再检查日志、文件数量、hook 与 clipping。
8. [KNOWN] 并排试听 original、reconstruction 和 manipulation，必要时下载单个 WAV。
9. [KNOWN] 需要批量分析时服务端另存当前 condition；需要传输时逐条件下载含 `provenance.json` 的 ZIP，同时保留服务器上的完整 provenance。
10. [KNOWN] 用独立声学工具复测 F0、formant、噪声或声源相关指标后，才形成语音学解释。

[KNOWN] 更完整的参数含义和 CLI 对照见 [Manipulation 指南](MANIPULATION_ZH.md)，目录职责见 [仓库地图](REPOSITORY_MAP_ZH.md)。
