.PHONY: install test unit integration lint format type build smoke clean

install:
	pip install -e ".[test,lint]"

test: unit

unit:
	pytest tests/unit -q

integration:
	pytest tests/integration -q

lint:
	ruff check .

format:
	ruff format .

type:
	mypy llama_index

build:
	python3 -m pip install --quiet build
	python3 -m build

smoke: build
	rm -rf .smoke-venv
	python3 -m venv .smoke-venv
	.smoke-venv/bin/pip install --quiet "$$(ls dist/*.whl)" pytest pytest-asyncio
	.smoke-venv/bin/pytest tests/smoke -q
	rm -rf .smoke-venv

clean:
	rm -rf dist build .smoke-venv *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
