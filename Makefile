UV ?= uv
PYTHON := .venv/bin/python
ARIS := .venv/bin/aris

.PHONY: install test test-lightweight lint format doctor verify

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
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

doctor:
	$(ARIS) doctor

verify: test lint
