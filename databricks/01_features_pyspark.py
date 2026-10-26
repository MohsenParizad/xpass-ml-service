# Databricks notebook source
# MAGIC %md
# MAGIC # xPass on Databricks · 01 Feature engineering in PySpark
# MAGIC
# MAGIC Cleaning and the 13 thesis features as Spark transformations (`xpass.spark_features`):
# MAGIC in-play passes only, goal difference at pass time (join on earlier goals), outplayed opponents
# MAGIC from the 360 freeze frames (join + aggregation), opponent power index (broadcast join).
# MAGIC Result: Delta table `<catalog>.<schema>.passes`.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "xpass")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

import os, sys
sys.path.insert(0, os.path.abspath("../src"))
from pyspark.sql import functions as F
from xpass.spark_features import build_features, read_inputs
from xpass.config import TARGET

# COMMAND ----------

events, frames, power_index = read_inputs(spark, f"{VOLUME}/euro2022", f"{VOLUME}/power_index_euro2022.csv")
print(f"events: {events.count():,}  frame rows: {frames.count():,}")

passes = build_features(events, frames, power_index, keep_row_id=True)
passes.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.passes")

# COMMAND ----------

t = spark.table(f"{CATALOG}.{SCHEMA}.passes")
display(t.agg(F.count("*").alias("passes"), F.sum(TARGET).alias("successful"),
              F.round(F.avg(TARGET), 4).alias("completion_rate")))
# expected: 22,974 passes, 17,728 successful (identical to the pandas pipeline)

# COMMAND ----------

display(t.groupBy("team").agg(F.count("*").alias("passes"), F.round(F.avg(TARGET), 3).alias("completion"),
                              F.round(F.avg("outplayed_players"), 2).alias("avg_outplayed"))
         .orderBy(F.desc("completion")))
