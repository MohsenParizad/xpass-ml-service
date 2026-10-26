# Databricks notebook source
# MAGIC %md
# MAGIC # xPass on Databricks · 03 Batch scoring with the registered model
# MAGIC
# MAGIC Loads `models:/<catalog>.<schema>.xpass@champion` as a Spark UDF (`mlflow.pyfunc.spark_udf`),
# MAGIC scores every pass in parallel and computes the thesis player score (Algorithm 1: mean of actual
# MAGIC outcome minus xPass, players with ≥ 51 passes). Results: Delta tables `pass_scores`, `player_scores`.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "xpass")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.xpass@champion"

import os, sys
sys.path.insert(0, os.path.abspath("../src"))
import mlflow
from xpass.spark_score import player_scores, score

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

scored = score(spark, spark.table(f"{CATALOG}.{SCHEMA}.passes"), MODEL_URI)
scored.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.pass_scores")

players = player_scores(spark.table(f"{CATALOG}.{SCHEMA}.pass_scores"))
players.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.player_scores")
display(players.limit(10))
# expected: 166 players, same ranking as reports/player_scores.csv

# COMMAND ----------

# MAGIC %sql
# MAGIC -- the same table, ready for a dashboard (Databricks SQL / Power BI)
# MAGIC SELECT position, COUNT(*) AS players, ROUND(AVG(score), 4) AS avg_score
# MAGIC FROM workspace.xpass.player_scores GROUP BY position ORDER BY avg_score DESC
