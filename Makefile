# End-to-end pipeline. Each target is one step of the BPMN process (docs/xpass_process.bpmn).
PY = PYTHONPATH=src python3

.PHONY: all data features train test serve drift docker explain spark-features mlflow-train spark-score parity spark-all mlflow-ui
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

# ---------- PySpark + MLflow (same pipeline, Databricks-ready; pip install -r requirements-spark.txt, Java 17+)
spark-features:  ## Steps 2-3 in PySpark -> data/processed/euro2022_passes_spark.parquet
	$(PY) -m xpass.spark_features euro2022

mlflow-train:    ## Steps 4-8 tracked in MLflow; best model registered as "xpass", alias champion if gate passes
	$(PY) -m xpass.mlflow_train --features data/processed/euro2022_passes_spark.parquet

spark-score:     ## Step 9 with the registered model as Spark UDF -> reports/player_scores_spark.csv
	$(PY) -m xpass.spark_score

parity:          ## Spark/MLflow results vs. pandas reference -> reports/spark_parity.md
	$(PY) -m xpass.parity

spark-all: spark-features mlflow-train spark-score parity

mlflow-ui:       ## MLflow UI on http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
