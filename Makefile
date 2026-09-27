# Targets are added only once they work. If it is listed here, it runs.
.DEFAULT_GOAL := check
.PHONY: setup test lint types eval demo study record-live check clean

setup:        ## install the project and dev dependencies
	uv sync

test:         ## run the test suite (no network, no API keys)
	uv run pytest -q

lint:         ## style and correctness lints
	uv run ruff check .
	uv run ruff format --check .

types:        ## strict type checking
	uv run mypy

eval:         ## verify the committed report reproduces from committed cassettes
	uv run python -m evals.study --check

demo:         ## narrated walkthrough of one fault experiment, offline
	uv run python -m evals.demo

study:        ## re-run the study against the scripted stand-in model
	uv run python -m evals.study

record-live:  ## record against a real OpenAI-compatible endpoint
              ## usage: make record-live UPSTREAM=... MODEL=... API_KEY=...
	@test -n "$(UPSTREAM)" || (echo "set UPSTREAM=https://host/v1" && exit 1)
	@test -n "$(MODEL)" || (echo "set MODEL=<model id>" && exit 1)
	uv run python -m evals.study \
		--upstream "$(UPSTREAM)" \
		--model "$(MODEL)" \
		--api-key "$(API_KEY)" \
		--variant baseline --variant distrust \
		--store evals/cassettes-live \
		--report evals/report-live.json

check: lint types test eval ## everything CI runs

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
