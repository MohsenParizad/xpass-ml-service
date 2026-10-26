"""Batch scoring in Spark with the registered MLflow model (step 9 at scale).

Loads the model by alias (models:/xpass@champion) as a Spark UDF, so scoring runs in
parallel on the cluster workers, then computes the thesis player score (Algorithm 1:
mean(actual - xPass) for players with >= 51 passes) as a Spark aggregation.

Usage
  local:       python -m xpass.spark_score            (after python -m xpass.mlflow_train)
  Databricks:  see databricks/03_score_registered.py
"""
import argparse
import sys

import mlflow
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .config import FEATURES, MIN_PASSES_PER_PLAYER, PROCESSED, REPORTS, TARGET, TRAIN_TAG
from .mlflow_train import MODEL_NAME, setup_tracking
from .spark_features import get_spark


def score(spark, passes: DataFrame, model_uri: str) -> DataFrame:
    """Add column 'xpass' = calibrated completion probability from the registered model."""
    udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double", env_manager="local")
    return passes.withColumn("xpass", udf(F.struct(*[F.col(c) for c in FEATURES])))


def player_scores(scored: DataFrame) -> DataFrame:
    """Thesis Algorithm 1 in Spark; same columns as reports/player_scores.csv."""
    # most frequent position per player (ties -> alphabetical, like pandas Series.mode)
    pos = (scored.groupBy("player", "team", "position").count()
           .withColumn("rk", F.row_number().over(
               Window.partitionBy("player", "team").orderBy(F.desc("count"), "position")))
           .filter("rk = 1").select("player", "team", "position"))
    agg = (scored.groupBy("player", "team")
           .agg(F.count("*").alias("passes"), F.sum(TARGET).alias("completed"),
                F.sum("xpass").alias("expected"), F.avg(TARGET).alias("completion_rate"),
                F.avg(F.col(TARGET) - F.col("xpass")).alias("score")))
    return (agg.join(pos, on=["player", "team"])
            .filter(F.col("passes") >= MIN_PASSES_PER_PLAYER)
            .select("player", "team", "position", "passes", "completed", "expected",
                    "completion_rate", "score")
            .orderBy(F.desc("score")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(PROCESSED / f"{TRAIN_TAG}_passes_spark.parquet"))
    ap.add_argument("--model-uri", default=f"models:/{MODEL_NAME}@champion")
    args = ap.parse_args()
    setup_tracking()
    spark = get_spark("xpass-scoring")
    scored = score(spark, spark.read.parquet(args.features), args.model_uri)
    scores = player_scores(scored).toPandas()
    out = REPORTS / "player_scores_spark.csv"
    scores.to_csv(out, index=False)
    print(f"{out}: {len(scores)} players scored with {args.model_uri}", file=sys.stderr)
    print(scores.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
