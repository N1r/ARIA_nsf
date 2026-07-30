UV ?= uv
PYTHON := .venv/bin/python
PHONLAB := .venv/bin/phonlab

.PHONY: install test test-lightweight lint format doctor verify-engine audit verify build

install:
	./scripts/setup_project_env.sh

test:
	$(PYTHON) -m pytest

test-lightweight:
	$(PYTHON) -m pytest \
		--ignore=tests/test_glottal.py \
		--ignore=tests/test_prediction_writer.py \
		--ignore=tests/test_time_tensor.py

lint:
	$(PYTHON) -m ruff check src tests tools

format:
	$(PYTHON) -m ruff format src tests tools

doctor:
	$(PHONLAB) doctor

verify-engine:
	$(PYTHON) tools/engine_checksums.py

audit:
	$(PYTHON) tools/repo_audit.py --strict

verify: test lint verify-engine audit

# Build outputs stay below .cache. setuptools may temporarily create ignored
# root build metadata; run `make audit` before build in CI/release workflows.
build:
	UV_CACHE_DIR=$(CURDIR)/.cache/uv \
	UV_PROJECT_ENVIRONMENT=$(CURDIR)/.venv \
	$(UV) build --out-dir .cache/dist
