UV ?= uv
PYTHON := .venv/bin/python
PHONLAB := .venv/bin/phonlab

.PHONY: install test test-lightweight lint format doctor verify-engine verify-webui verify-aria audit verify build

install:
	./scripts/setup_project_env.sh

test:
	$(PYTHON) -m pytest

test-lightweight:
	$(PYTHON) -m pytest \
		--ignore=tests/test_glottal.py \
		--ignore=tests/test_prediction_writer.py \
		--ignore=tests/test_time_tensor.py \
		--ignore=tests/test_controls.py \
		--ignore=tests/test_control_writer.py

lint:
	$(PYTHON) -m ruff check src tests tools

format:
	$(PYTHON) -m ruff format src tests tools

doctor:
	$(PHONLAB) doctor

verify-engine:
	$(PYTHON) tools/engine_checksums.py

verify-webui:
	$(PYTHON) tools/check_webui.py \
		--check-export \
		--output .cache/webui-acceptance.json

verify-aria:
	$(PYTHON) tools/check_aria_manipulation.py \
		artifacts/f024_aria_validated/postprocess-best/reconstruction \
		artifacts/f024_aria_validated/postprocess-best/manipulations \
		--output .cache/f024-aria-validated-acceptance.json

audit:
	$(PYTHON) tools/repo_audit.py --strict

verify: test lint verify-engine audit

# Build outputs stay below .cache. setuptools may temporarily create ignored
# root build metadata; run `make audit` before build in CI/release workflows.
build:
	UV_CACHE_DIR=$(CURDIR)/.cache/uv \
	UV_PROJECT_ENVIRONMENT=$(CURDIR)/.venv \
	$(UV) build --out-dir .cache/dist
