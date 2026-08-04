# 仓库地图与改动边界

这不是一个可以随意搬动旧模块的普通 Python 项目。仓库同时保存了新的稳定工作流和一套从冻结研究代码导入的兼容引擎。旧配置中的 `class_path`、Python 导入名以及 Lightning checkpoint 都可能引用精确的模块路径；文件仍能运行，并不代表改名后仍能载入旧实验。

最重要的原则是：**新工作流代码放进 `src/phonlab_ddsp/`；不要为了整理目录而移动兼容引擎。**

## 四层结构

| 层 | 路径 | 职责 | 改动原则 |
|---|---|---|---|
| 稳定库 | `src/phonlab_ddsp/` | 语料准备、切分、manifest、参数导出、实验、Slurm、报告、manipulation、CLI 与 GUI | 新功能的默认位置；保持公开 API、CLI 和元数据契约稳定 |
| checksum/checkpoint 兼容引擎 | `models/`、`ltng/`、`loss/`、`datasets/`、`cfg/`、`configs/`、`scripts/`，以及根目录旧 Python 模块 | 冻结的 GOLF/DDSP/ARIA 模型、训练代码、配置与历史入口 | **不得因重构或美化目录而移动、改名或改导入路径** |
| 用户与 QA 层 | `docs/`、`tests/`、`tools/`、`slurm/` | 使用说明、研究流程、回归测试、机械审计和集群验收任务 | 可以增补；改名时必须同步导航、CI、Makefile 和调用者 |
| 运行产物层 | `artifacts/`、`.cache/`、`.venv/` | 数据、实验输出、checkpoint、下载/构建缓存和本地环境 | 不是源码，不进入安装包；通常不提交 Git |

`provenance/` 是跨层的来源账本：它记录冻结代码从何处导入、冻结时状态和引擎 SHA-256。根目录的 `README.md`、`pyproject.toml`、`LICENSE` 等是项目导航与构建契约，不是放置新业务模块的地方。

## 1. 稳定工作流：`src/phonlab_ddsp/`

这里是研究者应直接使用、开发者应继续扩展的库：

- `corpus.py`、`segment.py`、`audio.py`、`manifest.py` 负责语料获取、切分、声学处理和可移动数据集契约；
- `parameters.py`、`report.py`、`metrics.py` 产生可检查的 CSV、JSON 和 HTML；
- `experiment.py`、`lightning.py` 把稳定工作流适配到旧训练引擎；
- `jobs.py`、`launchers.py` 负责安全的 Slurm 调度与后处理作业包；
- `controls/`、`manipulation.py` 提供经过能力检查的 checkpoint 控制；
- `cli.py`、`gui.py` 应保持为调用同一套库函数的薄接口。

数据准备、检查和报告代码应尽量维持轻依赖边界，不要因为增加一个 CPU 工作流就让导入 `phonlab_ddsp` 必须加载 Torch/CUDA。需要连接冻结引擎时，优先在这一层增加适配器，而不是把新工作流写回旧模型文件。

## 2. 冻结兼容引擎：路径就是接口

兼容引擎包括：

```text
models/  ltng/  loss/  datasets/  cfg/  configs/  scripts/
autoencode.py  main.py  biquads.py  harm_and_noise.py
```

这些名称至少承担三种兼容责任：

1. YAML 配置以字符串形式引用诸如 `models.*`、`ltng.*` 和 `loss.*`；
2. Python/Lightning checkpoint 可能在反序列化时导入原来的模块和类；
3. 训练与推理适配器仍以 `python -m autoencode` 等历史入口调用引擎。

`tools/engine_checksums.py` 还会按照 `provenance/ENGINE_FILES.sha256` 校验这些目录、根模块和 `scripts/`。日常检查可运行：

```bash
python tools/engine_checksums.py
```

检查同时拒绝清单内文件的缺失/变化，以及冻结目录中未登记的新文件；因此把新
业务模块误放进 `models/`、`ltng/` 等目录也会失败。

校验失败时先确认是否误改了冻结引擎。不要仅为了让检查变绿就执行 `--write`。真正需要修改引擎时，应把它当作一次兼容迁移：说明理由，保留旧导入 shim，验证旧 checkpoint 和配置，增加等价性/回归测试，审查 provenance 后才更新 checksum 清单。

`datasets/` 尤其容易产生误解：它是历史源代码命名空间，不是新数据工作流的位置，也不会作为公共 `datasets` 包安装，以免与 Hugging Face 的同名包冲突。新数据流程仍应进入 `src/phonlab_ddsp/`。

## 3. 用户、测试与机械验收

- `docs/` 面向使用者与维护者，保存流程、架构、复现和科学解释；
- `tests/` 同时覆盖轻依赖工作流与旧模型兼容性；
- `tools/` 放仓库审计、checksum 和验收工具，不放可复用业务逻辑；
- `slurm/` 保存真实集群的端到端验收任务；通用作业生成逻辑仍放在稳定库中。

测试能通过不等于旧 checkpoint 一定兼容。涉及兼容引擎路径的变更还必须运行 checksum 检查，并用代表性的旧配置/checkpoint 做载入或 dry-run 验证。

## 4. 运行环境与产物

`.venv/`、`.cache/` 和 `artifacts/` 可以存在于工作区，但它们不是仓库源码：

- `.venv/` 是项目本地 Python 环境；
- `.cache/` 保存 uv、下载和编译缓存；
- `artifacts/` 保存数据、训练 run、checkpoint、日志、重建音频和报告。

研究结果即使位于 `artifacts/`，也可能是论文证据；清理前仍应确认备份、哈希和复现记录。另一方面，不要把 checkpoint 或权重散落在仓库顶层。仓库审计明确禁止顶层 `build/`、`dist/`、`__pycache__/`、`*.egg-info`、`*.ckpt` 和 `*.pt`。

## 新代码放哪里

| 要增加的内容 | 推荐位置 | 说明 |
|---|---|---|
| 可复用的语料、实验、控制或报告逻辑 | `src/phonlab_ddsp/` | 先写库 API，再由 CLI/GUI 调用 |
| 新的公开控制规范与运行时适配 | `src/phonlab_ddsp/controls/` | 不修改 checkpoint 内部类路径即可扩展能力 |
| CLI 子命令 | `src/phonlab_ddsp/cli.py` + 独立库模块 | CLI 只解析参数和展示结果 |
| GUI 卡片/动作 | `src/phonlab_ddsp/gui.py` + 独立库模块 | GUI 与 CLI 应共享验证和输出契约 |
| 新工作流 preset | `src/phonlab_ddsp/presets/` | 与冻结的 `cfg/`、`configs/` 分开 |
| 回归或科研边界测试 | `tests/` | 对公开输出、失败模式和兼容路径断言 |
| 仓库机械检查 | `tools/` | 例如审计器、验收器，不承载用户库 API |
| 集群端到端验收脚本 | `slurm/` | 可复用的 Slurm 生成逻辑放 `launchers.py` |
| 使用与研究说明 | `docs/` | 同步更新 README 导航（如适用） |
| 数据、checkpoint、试听音频和 HTML 输出 | `artifacts/` 或受管外部存储 | 不作为源码提交 |

如果需求看似必须修改 `models/` 或 `ltng/`，先问能否在 `src/phonlab_ddsp/` 中用 wrapper、hook、callback 或 adapter 完成。只有模型数学本身确实需要改变时，才进入兼容迁移流程。

## 可移动与禁止改名表

| 对象 | 能否移动/改名 | 条件或原因 |
|---|---|---|
| `src/phonlab_ddsp/` 内部的私有新模块 | 可以，需谨慎 | 更新导入和测试；已公开的 Python 路径应保留 shim |
| 包名 `phonlab_ddsp`、CLI entry point | 不应随意改名 | 是用户脚本和安装元数据的稳定接口 |
| 元数据或生成命令中记录的 callback/class path | 不应直接改名 | 旧作业包和配置可能按字符串重新导入 |
| `models/`、`ltng/`、`loss/`、`datasets/` | **禁止整理式搬迁** | checkpoint、YAML 和历史导入依赖精确路径 |
| `cfg/`、`configs/` | **禁止整理式搬迁** | 是实验定义，也在 checksum 范围内 |
| `autoencode.py`、`main.py`、`biquads.py`、`harm_and_noise.py` | **禁止改名或搬入包内** | 历史模块名和入口仍被调用 |
| 现有 `scripts/` | 不应随意移动 | 被 checksum 覆盖，文档和 Slurm 脚本也直接引用 |
| `provenance/` 及 checksum 清单 | 不应改名 | 是冻结来源与审计链的一部分 |
| 一般 `docs/` 页面 | 可以 | 修正相对链接和 README 导航；关键审计文档路径需保留 |
| `tests/`、`tools/` 内文件 | 可以，需同步调用者 | 顶层目录名受 pytest、Makefile、CI 和文档约定 |
| `.venv/`、`.cache/` | 可删除后重建 | 不可把其中内容当作唯一研究证据 |
| `artifacts/` 内产物 | 可归档或迁移 | 迁移整套数据/实验时保留 metadata、相对路径、哈希与备份 |

判断不清时，先搜索字符串引用，再检查构建和 checksum：

```bash
rg '旧模块名或路径' .
python tools/engine_checksums.py
python tools/repo_audit.py --strict
```

目录整洁不是破坏复现性的理由。对这个仓库而言，旧路径本身就是历史实验接口。
