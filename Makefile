PYTHON ?= python
OUT ?= outputs
.SHELLFLAGS := -c
.DEFAULT_GOAL := check

.PHONY: check lint typecheck test coverage sample demo benchmark

check: lint typecheck test

lint:
	"$(PYTHON)" -m ruff check .
	"$(PYTHON)" -m ruff format . --check

typecheck:
	"$(PYTHON)" -m mypy geodiscover scripts

$(OUT):
	"$(PYTHON)" -c "from pathlib import Path; Path('$(OUT)').mkdir(parents=True, exist_ok=True)"

test: | $(OUT)
	"$(PYTHON)" -m pytest --cov=geodiscover --cov-report=term-missing --cov-report="json:$(OUT)/coverage.json" -q

coverage: test

sample: | $(OUT)
	"$(PYTHON)" scripts/generate_sample_data.py --output "$(OUT)/samples" --models 18

demo: | $(OUT)
	"$(PYTHON)" -m geodiscover.demo --trials 4 --budget 2 --simulations 16 --output "$(OUT)/demo.json"

benchmark: | $(OUT)
	"$(PYTHON)" -m geodiscover.benchmark --budgets 1,2 --trials 4 --simulations 16 --output "$(OUT)/efficiency.json"
