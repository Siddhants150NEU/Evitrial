# EVI-TRIAL — run everything from here.
# The src package sits at the repo root, so it imports directly — no PYTHONPATH needed.
PY ?= python

.DEFAULT_GOAL := help
# .PHONY: help install check check-fast test eval demo clean
PORT ?= 8777
RUNG ?= generative
K    ?= 10
.PHONY: help install check check-fast test eval demo webapp webapp-live webapp-bg webapp-stop cache clean

help: ## Show this help
	@echo "EVI-TRIAL make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime deps + editable package
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

ingest-qdrant:
	$(PY) -m src.indexQdrant

train-lora:
	$(PY) -m src.trainLora

indexing:
	$(PY) -m src.i

check: ## Full data sanity check (downloads TREC CT 2021 corpus on first run — slow)
	$(PY) -m src.checkIngest

check-fast: ## Data check, skipping the slow 375k full-corpus count
	$(PY) -m src.checkIngest --fast

test: ## Run the contract invariants (no data / heavy deps needed)
	$(PY) tests/testContracts.py

eval: ## Run the eval harness -> reports/runs/<runId>/
	$(PY) -m src.eval

# demo: ## Launch the minimal Streamlit demo
# 	streamlit run src/demo.py
webapp: ## Serve the demo (cached runs only). PORT=8777 by default
	$(PY) -m webapp.server.serveDemo --port $(PORT)

webapp-live: ## Same, but visitors can run their own notes. Slow — ~10 min a run
	$(PY) -m webapp.server.serveDemo --port $(PORT) --live

webapp-bg: ## Start it in the background, wait until it answers, open a browser
	@nohup $(PY) -m webapp.server.serveDemo --port $(PORT) > .webapp.log 2>&1 & echo $$! > .webapp.pid
	@until curl -sf http://localhost:$(PORT)/api/matchers >/dev/null; do sleep 0.2; done
	@open http://localhost:$(PORT)
	@echo "serving http://localhost:$(PORT) (pid $$(cat .webapp.pid)) — make webapp-stop to kill"

webapp-stop: ## Kill the background demo server
	@kill $$(cat .webapp.pid) 2>/dev/null && rm -f .webapp.pid && echo stopped || echo "not running"

cache: ## Pre-build a demo run: make cache PATIENT=sigir-20142 K=10
	$(PY) -m webapp.server.runCache --patient $(PATIENT) --rung $(RUNG) --k $(K)

clean: ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
