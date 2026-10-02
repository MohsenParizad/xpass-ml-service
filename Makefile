# End-to-end pipeline. Each target is one step of the BPMN process (docs/xpass_process.bpmn).
PY = PYTHONPATH=src python3

.PHONY: all data features train test serve drift docker explain
all: data features train test

data:        ## Step 1: download StatsBomb open data
	$(PY) -m xpass.ingest euro2022

features:    ## Steps 2-3: clean + feature engineering
	$(PY) -m xpass.features euro2022

train:       ## Steps 4-9: split, experiments, calibration, evaluation gate, player scores
	$(PY) -m xpass.train --version $${VERSION:-1.0.0}

test:        ## Step 11: unit, model and API tests
	python3 -m pytest -q

serve:       ## Step 10: run the API locally -> http://localhost:8000/docs
	$(PY) -m uvicorn xpass.api:app --reload --port 8000

drift:       ## Step 15: drift check against a new tournament (TAG=wwc2023 or euro2025)
	$(PY) -m xpass.ingest $${TAG:-wwc2023} && $(PY) -m xpass.drift $${TAG:-wwc2023}

docker:      ## Step 12: build and run the serving image
	docker build -t xpass-api:local . && docker run --rm -p 8080:8080 xpass-api:local

explain:     ## Explainability: SHAP vs built-in vs permutation importance -> reports/explainability.md
	$(PY) scripts/explain.py
