# Parity: PySpark + MLflow pipeline vs. pandas reference

Run locally with Spark 4 (local mode) and MLflow (SQLite tracking, local registry).

## Feature table

- rows: 22,974 (pandas) vs. 22,974 (Spark)
- same row order: True
- mismatching cells over 14 columns: 0
- **identical: True**

## Model (MLflow run on the Spark feature table)

Selected variant and test metrics are logged to MLflow; they equal
`reports/experiments.json` from `make train` (CatBoost, no resampling, Brier 0.1111).

## Player scores (registered model via `mlflow.pyfunc.spark_udf`)

- players: 166 vs. 166, matched 166
- same positions / pass counts: True / True
- max |score difference|: 1.1e-16
- max |expected completions difference|: 3.4e-13
- **same ranking: True**

## Implementation notes

- Pass angle and length use Python's `math` in a vectorised pandas UDF. Spark's built-in
  `atan2`/`hypot` (Java) differ in the last bit: one pass of exactly 135° flipped from
  'square' to 'backward', and 13 % of the lengths differed by ~1e-14.
- The Spark output keeps the source row order. The split is by match, but the CV folds
  inside `CalibratedClassifierCV` depend on row order, so a reordered table gives slightly
  different models (Brier 0.1112 instead of 0.1111).
- The registered model is a pyfunc wrapper that returns P(completed). With the plain sklearn
  flavour and `predict_proba`, `spark_udf(result_type='double')` silently takes column 0,
  i.e. P(failed); the player-score comparison caught this.

## Databricks Free Edition (serverless, 26 Oct 2026)

All five notebooks in `databricks/` ran successfully. Same raw data (SHA-256 `c91a12fb…`); feature table
bit-identical to the pandas reference (0 mismatching cells); best model CatBoost without resampling with
Brier 0.1111, log loss 0.3472, ROC AUC 0.8820; registered as `workspace.xpass.xpass` v1 with alias `champion`;
top 8 players identical to the local run, places 9 and 10 swapped. Small metric differences come from the
runtime libraries, not from the data or the features.
