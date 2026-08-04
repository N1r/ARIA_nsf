# GitHub 公开发布检查清单

这份清单用于第一次公开仓库以及后续版本发布，不替代机构的数据、伦理、版权或
开源审批。

## 1. 补齐身份和链接

- 将 `pyproject.toml` 中的通用贡献者名称替换为实际维护者或团队。
- 在 `pyproject.toml` 增加最终 repository、issues 和 documentation URL。
- 将 `CITATION.cff` 作者替换为实际软件作者；获得 DOI 后补 `doi`。
- 在 `SECURITY.md` 填入私人安全联系渠道，或启用 GitHub 私密漏洞报告。
- 核对 `LICENSE` 的 copyright、上游许可证和新增代码归属；不确定时不要猜测。

## 2. 检查将上传的文件

```bash
git status --short
git diff --check
git ls-files
git ls-files -z | xargs -0 du -h | sort -h | tail
```

确认没有录音、受限制语料、checkpoint、权重、个人信息、token、私钥、集群日志
或含私人路径的生成报告。`.gitignore` 已排除常见产物，不要用 `git add -f` 绕过。

冻结 `cfg/`、`configs/`、`scripts/` 和 `provenance/` 保留部分历史
`/zfsstore/...` 路径用于来源复现。公开前应确认它们不构成机构或参与者隐私问题；
稳定 `phonlab` 工作流不依赖这些路径。

## 3. 运行质量门

```bash
source scripts/project_env.sh
uv lock --check
make lint
make test
make verify-engine
make audit
make build
```

若本地保留被忽略的真实验收产物，再运行：

```bash
make verify-webui
make verify-aria
```

## 4. 用干净 clone 复查

提交后在新的临时 clone 中执行轻量安装和测试，防止当前未跟踪文件掩盖缺失资源：

```bash
git clone YOUR_FINAL_REPOSITORY_URL phonlab-ddsp-clean
cd phonlab-ddsp-clean
source scripts/project_env.sh
./scripts/setup_project_env.sh
make test-lightweight
make audit
make build
```

不要在确认 URL 前把示例占位符写入 `pyproject.toml`。

## 5. GitHub 设置

- 默认分支建议使用 `main`；当前本地若仍为 `master`，首次 push 前统一。
- 启用 branch protection，要求 GitHub Actions 通过后才能合并。
- 启用 secret scanning、Dependabot alerts 和 private vulnerability reporting。
- 在 About 中填写描述、许可证、文档链接和 topics。
- 首个公开版本使用带注释 tag，例如 `v0.1.0`，并列出验证和科学限制。
- 大型公开音频或 checkpoint 使用带 checksum 的独立归档，不写入 Git 历史。

## 6. 当前人工待办

当前代码库尚未配置 Git remote；`CITATION.cff` 与 `pyproject.toml` 的维护者仍是
通用文本。这些信息必须由仓库所有者提供，自动化工具不能猜测。
