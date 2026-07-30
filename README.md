# PhonLab-DDSP

面向语音学家的可复现 DDSP / GOLF 分析合成工具包。它把 SLT 2026
冻结研究代码整理为一条可操作的工作流：**获取或导入录音、切分、提取
F0 与声学参数、质检、Slurm 训练、查看 loss、checkpoint 推理和可审计的
音高 manipulation**。

> 当前状态：研究预览版 `0.1.0`。旧 GOLF/ARIA 模型保持兼容，新数据与实验接口已有端到端测试。模型输出可用于分析合成和假设探索，但不应在未经独立验证时被直接解释为生理或构音测量。

## 从哪里开始

| 你要做什么 | 入口 |
|---|---|
| 第一次使用或不熟悉命令行 | `phonlab gui`，按 0–7 号卡片操作 |
| 准备自己的录音 | [中文快速开始](docs/QUICKSTART_ZH.md) |
| 复现 30 分钟互联网语料完整流程 | [CMU ARCTIC 完整流程](docs/CMU_ARCTIC_PIPELINE_ZH.md) |
| 了解数据哈希和归档要求 | [数据与复现契约](docs/DATA_AND_REPRODUCIBILITY.md) |
| 开发或理解新旧代码边界 | [架构说明](docs/ARCHITECTURE.md) |
| 查看科研背景材料 | [`docs/research/`](docs/research/) |

## 五分钟开始

```bash
source scripts/project_env.sh
./scripts/setup_project_env.sh
.venv/bin/phonlab doctor

.venv/bin/phonlab split recordings/ segments/ --mode silence
.venv/bin/phonlab prepare segments/audio data/my_voice --f0-method autocorr
.venv/bin/phonlab validate data/my_voice
.venv/bin/phonlab inspect data/my_voice

.venv/bin/phonlab init-experiment data/my_voice experiments/my_voice_golf --model golf
.venv/bin/phonlab train experiments/my_voice_golf --dry-run
```

不想使用命令行时，可启动仅绑定本机的浏览器工作台：

```bash
.venv/bin/phonlab gui
```

`autocorr` 是无需安装 `pyworld` 的默认可移植 F0 后端。已有 Praat/其他工具
生成的 `.pv` 轨迹也可用 `--f0-method sidecar` 复用。

`project_env.sh` 将 uv、Python、Torch、Numba、Matplotlib 和模型下载缓存全部
固定在本仓库的 `.venv/`、`.cache/` 和 `artifacts/`。GPU 训练与 checkpoint
推理应通过 Slurm 脚本提交，不应直接在登录节点运行。

集群上的短 F024 冒烟闭环：

```bash
source scripts/project_env.sh
./scripts/setup_project_env.sh
sbatch slurm/f024_e2e_smoke.slurm
```

该作业在 Slurm 短队列申请一张 GPU，执行一步训练、checkpoint
复制合成与并排试听报告。所有环境、下载缓存和输出均位于本仓库。

完整的公开语料验收场景使用官方 CMU ARCTIC `slt`：

```bash
source scripts/project_env.sh
.venv/bin/phonlab fetch-corpus artifacts/cmu_arctic_slt_demo/corpus
.venv/bin/phonlab split \
  artifacts/cmu_arctic_slt_demo/corpus/continuous.wav \
  artifacts/cmu_arctic_slt_demo/segments
# 其余可复制命令与验收方式见 docs/CMU_ARCTIC_PIPELINE_ZH.md
```

仓库内的验收运行已实际完成：Slurm Job `4558227` 在 `node887` 的 NVIDIA
L4 上训练 400 steps，并生成 63 条 held-out 重建及 −4/+4 半音各 63 条；
`train_loss` 由 297.72 降至 5.52，最终 `val_loss` 为 6.036，NaN/Inf 为 0。
机械验收命令输出 `PHONLAB_COMPLETE_PIPELINE_OK`。这些是工程复现证据，不是
模型充分收敛或语音学效度声明。

## CLI

```text
phonlab doctor             检查音频、F0、GPU 训练环境
phonlab fetch-corpus       获取并校验可复现的 CMU ARCTIC 示例语料
phonlab split              按静音边界或固定窗口切分连续录音
phonlab prepare            重采样、转单声道、提取 F0、确定性划分、生成哈希
phonlab parameters         导出逐条时长、能量、F0、清浊比例等参数表
phonlab validate           检查 manifest、路径和文件完整性
phonlab inspect            生成可离线浏览与试听的 HTML 质检报告
phonlab init-experiment    生成配置、provenance、Shell 与 Slurm 启动器
phonlab train              验证 dataset fingerprint 后启动 Lightning 训练
phonlab metrics            汇总 train/validation loss、学习率和异常数值
phonlab synthesize         用 checkpoint 重建 held-out 测试录音
phonlab manipulate         对有声 F0 做半音变换并记录 checkpoint 与参数
phonlab init-postprocess   生成重建、manipulation 和报告的 GPU 作业包
phonlab compare            生成原音/重建音并排试听页面
phonlab gui                启动本地浏览器工作台
```

每个命令都有 `--help`，例如：

```bash
phonlab prepare --help
```

## 代码结构

```text
src/phonlab_ddsp/  新的稳定工具层和 CLI
models/            GOLF / DDSP / ARIA 合成模型（冻结基线导入）
ltng/              Lightning 训练代码
loss/              频谱损失
cfg/               原论文与 ARIA 模型配置
tests/             工具层端到端测试和模型兼容测试
docs/              用户、架构和复现说明
provenance/        2026-07-08 冻结快照来源记录
slurm/             可直接提交的真实 GPU 验收作业
tools/             机械验收、仓库审计和引擎 checksum
```

`src/phonlab_ddsp/` 是供新代码调用的稳定共享库；顶层 `models/`、`ltng/`、
`loss/` 以及若干入口文件保留原导入路径，以兼容冻结 checkpoint 和配置。
新功能不应继续堆入这些兼容模块。

## 面向语音学研究的默认原则

- 不默认做逐文件响度/峰值归一化，以免悄悄抹去潜在研究变量。
- F0 上下界、算法、重采样率和划分种子进入 provenance。
- 数据路径存为相对路径，可把整个数据集复制到其他机器。
- 原始文件 SHA-256 和 dataset fingerprint 防止无声的数据漂移。
- HTML 报告允许逐条试听并暴露削波、无 F0 等异常。
- 训练日志使用本地 CSV，不要求注册第三方追踪平台。
- manipulation 记录数据 fingerprint、checkpoint SHA-256、半音与 F0 比例；
  它是受控合成条件，不等同于生理或构音因果测量。

## 开发

```bash
source scripts/project_env.sh
./scripts/setup_project_env.sh
make test
make lint
python tools/repo_audit.py --strict
python tools/engine_checksums.py
```

旧模型测试需要安装训练依赖及其 CUDA 扩展；轻量工作流测试只需 NumPy。

## 来源与引用

本仓库的研究引擎来自上级目录的
`golf_frozen_slt2026_20260708`。冻结状态、补丁和环境记录保存在
`provenance/`，大型 runs/checkpoints 没有复制进代码库。

底层 GOLF 方法：

- Chin-Yun Yu and György Fazekas, “Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis,” Interspeech 2024, DOI: `10.21437/Interspeech.2024-1187`.
- Chin-Yun Yu and György Fazekas, “Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables,” ISMIR 2023, DOI: `10.5281/zenodo.10265377`.

进一步发布时请补充本项目维护者、仓库 URL 和版本 DOI；机器可读引用信息见 `CITATION.cff`。

## License

MIT；详见 [LICENSE](LICENSE)。
