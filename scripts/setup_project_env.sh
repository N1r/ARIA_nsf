#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/project_env.sh"

UV_BIN="${UV_BIN:-$(command -v uv)}"
"$UV_BIN" python install 3.11 --install-dir "$UV_PYTHON_INSTALL_DIR" --no-bin
"$UV_BIN" venv --managed-python --clear --python 3.11 "$UV_PROJECT_ENVIRONMENT"
"$UV_BIN" sync --managed-python --python 3.11 --extra audio --extra train --extra dev
"$ARIS_PROJECT_ROOT/.venv/bin/python" -m aris.cli doctor
