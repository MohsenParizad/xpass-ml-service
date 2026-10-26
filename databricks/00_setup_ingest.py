# Databricks notebook source
# MAGIC %md
# MAGIC # xPass on Databricks · 00 Setup and ingest
# MAGIC
# MAGIC Creates the Unity Catalog schema and volume, installs the pinned libraries, downloads the
# MAGIC StatsBomb open data (UEFA Women's Euro 2022, events + 360 frames, 31 matches) and stores the raw
# MAGIC parquet files in the volume.
# MAGIC
# MAGIC Run this notebook from a **Git folder** of the repository (Workspace → Create → Git folder),
# MAGIC so that `../src` contains the `xpass` package. Target: Databricks Free Edition (serverless).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "xpass")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw")

# COMMAND ----------

import os, shutil, sys
sys.path.insert(0, os.path.abspath("../src"))
from xpass.config import RAW
from xpass.ingest import ingest

manifest = ingest("euro2022")  # downloads from raw.githubusercontent.com/statsbomb/open-data
shutil.copytree(RAW / "euro2022", f"{VOLUME}/euro2022", dirs_exist_ok=True)
shutil.copy(os.path.abspath("../data/reference/power_index_euro2022.csv"), f"{VOLUME}/power_index_euro2022.csv")
print(manifest)
display(dbutils.fs.ls(f"{VOLUME}/euro2022"))

# COMMAND ----------

# MAGIC %md
# MAGIC If the cluster has no internet access, run `make data` locally and upload
# MAGIC `data/raw/euro2022/*.parquet` and `data/reference/power_index_euro2022.csv` into the volume
# MAGIC (Catalog → workspace → xpass → raw → Upload).
