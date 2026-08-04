# Contributing to PhonLab-DDSP

[中文说明](#中文说明) | [English](#english)

## English

PhonLab-DDSP separates a stable user-facing library from a frozen,
checkpoint-compatible research engine. Preserve that boundary.

1. Put new workflow, CLI, control, report, or WebUI code under
   `src/phonlab_ddsp/`.
2. Do not edit `models/`, `ltng/`, `loss/`, `datasets/`, `cfg/`, `configs/`,
   `scripts/`, or legacy top-level modules unless the change explicitly updates
   frozen-engine provenance and compatibility.
3. Do not commit recordings, datasets, checkpoints, experiment runs, generated
   reports containing private paths, `.venv/`, or caches.
4. Keep GPU training and checkpoint inference in Slurm jobs. Ordinary CI tests
   must remain CPU-only.
5. Document control semantics and scientific limitations. A software control is
   not automatically a validated acoustic or physiological measurement.

Before a pull request, run:

```bash
source scripts/project_env.sh
make lint
make test
make verify-engine
make audit
git diff --check
```

Run `make verify-webui` and `make verify-aria` when the corresponding ignored
local artifacts are available. A pull request should state its user-visible
outcome, tests, provenance impact, and remaining scientific limitations.

## 中文说明

PhonLab-DDSP 将稳定用户库与冻结、兼容 checkpoint 的研究引擎分开。新工作流、
CLI、控制、报告和 WebUI 代码应放在 `src/phonlab_ddsp/`。除非同步更新冻结来源
和兼容性契约，否则不要修改 `models/`、`ltng/`、`loss/`、`datasets/`、`cfg/`、
`configs/`、`scripts/` 或旧顶层入口。

不要提交录音、数据集、checkpoint、实验输出、包含私人路径的生成报告、`.venv/`
或缓存。GPU 训练与 checkpoint 推理必须通过 Slurm；普通 CI 测试保持 CPU-only。
提交变更前运行上述验证命令，并在 PR 中说明结果、测试、provenance 影响和科学限制。
