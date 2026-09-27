# Targets are added only once they work. If it is listed here, it runs.
.DEFAULT_GOAL := check
.PHONY: setup test lint types check clean

setup:                ## install the project and dev dependencies
	uv sync

test:                 ## run the test suite (no network, no API keys)
	uv run pytest -q

lint:                 ## style and correctness lints
	uv run ruff check .
	uv run ruff format --check .

types:                ## strict type checking
	uv run mypy

check: lint types test ## everything CI runs

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
