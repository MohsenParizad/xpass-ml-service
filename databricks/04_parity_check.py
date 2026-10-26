# Databricks notebook source
# MAGIC %md
# MAGIC # xPass on Databricks · 04 Parity check: Spark pipeline vs. pandas reference
# MAGIC
# MAGIC Runs the original pandas feature pipeline (`xpass.features`, also used by the API) on the driver
# MAGIC and compares it with the Spark Delta table: same rows, same order, bit-identical values.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "xpass")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

import os, sys
sys.path.insert(0, os.path.abspath("../src"))
import tempfile
from pathlib import Path
from xpass.parity import compare_feature_tables
from xpass import features

# read the same raw files as the Spark pipeline (works whether notebook 00 downloaded them or you uploaded them)
VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
features.RAW, features.PROCESSED = Path(VOLUME), Path(tempfile.mkdtemp())
reference = features.build("euro2022")
spark_df = spark.table(f"{CATALOG}.{SCHEMA}.passes").orderBy("source_row").drop("source_row").toPandas()
report = compare_feature_tables(reference, spark_df)
print(report)
assert report["identical"], report
