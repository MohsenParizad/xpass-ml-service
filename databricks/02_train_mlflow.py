# Databricks notebook source
# MAGIC %md
# MAGIC # xPass on Databricks · 02 Training with MLflow tracking and Unity Catalog registry
# MAGIC
# MAGIC Four experiments ({logistic regression, CatBoost} × {SMOTE, none}, isotonic calibration, split by
# MAGIC match) as nested MLflow runs with parameters, metrics and calibration plots. The best model (lowest
# MAGIC Brier score) is registered as `<catalog>.<schema>.xpass`; the alias `champion` is set only if the
# MAGIC evaluation gate passes (Brier ≤ 0.13, log loss ≤ 0.42, ROC AUC ≥ 0.84).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "xpass")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
MODEL_NAME = f"{CATALOG}.{SCHEMA}.xpass"

import os, sys
sys.path.insert(0, os.path.abspath("../src"))
from xpass.mlflow_train import run, setup_tracking

user = spark.sql("SELECT current_user()").first()[0]
setup_tracking(experiment=f"/Users/{user}/xpass")

# COMMAND ----------

# The feature table is small (~23k rows), so training runs on the driver with scikit-learn/CatBoost.
# orderBy(source_row) restores the source order -> identical CV folds and metrics to the local run.
df = spark.table(f"{CATALOG}.{SCHEMA}.passes").orderBy("source_row").drop("source_row").toPandas()
summary = run(df, model_name=MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC Expected (same as `reports/experiments.json`): CatBoost without resampling wins with
# MAGIC Brier 0.1111, log loss 0.347, ROC AUC 0.882; gate passed → alias `champion`.
# MAGIC Open **Experiments** in the left sidebar to compare the four runs, and **Catalog → Models** for the
# MAGIC registered versions.
