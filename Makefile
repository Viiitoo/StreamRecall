PYTHON ?= python3
API_VENV := apps/api/.venv
API_PYTHON := $(API_VENV)/bin/python
API_PIP := $(API_VENV)/bin/pip

.PHONY: setup demo test check-test-deps test-api test-core test-web

setup:
	$(PYTHON) -m venv $(API_VENV)
	$(API_PIP) install --upgrade pip
	$(API_PIP) install -r apps/api/requirements.txt -e 'streamtimelens[dev]'
	cd apps/web && npm ci

demo:
	./apps/run-local.sh

test: check-test-deps test-api test-core test-web

check-test-deps:
	@test -f third_party/TimeLens/timelens/utils.py || (echo "Initialize test dependencies with: git submodule update --init third_party/TimeLens" >&2; exit 1)

test-api:
	PYTHONPATH=apps/api $(API_PYTHON) -m pytest -q apps/api/tests

test-core:
	PYTHONPATH=third_party/TimeLens $(API_PYTHON) -m pytest -q streamtimelens/tests

test-web:
	cd apps/web && npm run build
