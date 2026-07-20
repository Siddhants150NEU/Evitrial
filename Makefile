# EVI-TRIAL — run everything from here.
# The src package sits at the repo root, so it imports directly — no PYTHONPATH needed.
PY ?= python

.DEFAULT_GOAL := help
.PHONY: help install check check-fast test eval demo clean

help: ## Show this help
	@echo "EVI-TRIAL make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime deps + editable package
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

ingest-qdrant:
	$(PY) -m src.indexQdrant
	
check: ## Full data sanity check (downloads TREC CT 2021 corpus on first run — slow)
	$(PY) -m src.checkIngest

check-fast: ## Data check, skipping the slow 375k full-corpus count
	$(PY) -m src.checkIngest --fast

test: ## Run the contract invariants (no data / heavy deps needed)
	$(PY) tests/testContracts.py

eval: ## Run the eval harness -> reports/runs/<runId>/
	$(PY) -m src.eval

demo: ## Launch the minimal Streamlit demo
	streamlit run src/demo.py

clean: ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
