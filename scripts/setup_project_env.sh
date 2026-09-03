#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/project_env.sh"

echo "==> Synchronizing ARIS environment with uv..."
uv sync --all-extras
uv run aris doctor
