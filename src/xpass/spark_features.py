"""Steps 2 + 3 in PySpark: the same cleaning and feature engineering as xpass.features,
written for Apache Spark so it runs on Databricks (or any Spark cluster).

The pandas version (xpass.features) stays the reference implementation and is still used
by the API, so that training and serving share one code path. This module reproduces it
row by row; tests/test_spark_features.py and databricks/04_parity_check.py check that both
produce the same feature table.

Usage (local Spark):  python -m xpass.spark_features euro2022
"""
import math
import sys

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import (EXCLUDED_OUTCOMES, EXCLUDED_PASS_TYPES, FEATURES, PROCESSED, RAW,
                     REFERENCE, TARGET)

HEIGHTS = {"Ground Pass": "ground", "Low Pass": "low", "High Pass": "high"}
META_COLS = ["match_id", "event_id", "team", "opponent", "player", "position"]


def get_spark(app_name: str = "xpass") -> SparkSession:
    """Return the active session on Databricks, or a small local session elsewhere."""
    return (SparkSession.builder.appName(app_name)
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def _not_in(col: str, values) -> F.Column:
    """NULL-safe 'not in': in pandas, NaN is never part of the exclusion set."""
    return F.col(col).isNull() | ~F.col(col).isin(list(values))


def clean(events: DataFrame) -> DataFrame:
    """Thesis 3.1.1: in-play passes only; drop offside/unknown/injury clearance."""
    return (events
            .filter(F.col("type") == "Pass")
            .filter(_not_in("pass_type", EXCLUDED_PASS_TYPES))
            .filter(_not_in("outcome", EXCLUDED_OUTCOMES))
            .filter(F.col("height").isin(list(HEIGHTS)))
            .withColumn(TARGET, F.col("outcome").isNull().cast("long")))  # no outcome = completed


def add_goal_diff(passes: DataFrame, events: DataFrame) -> DataFrame:
    """Goals for minus goals against for the passing team at the moment of the pass.
    A goal counts if it happened strictly before the pass (event index), as in the pandas
    version (np.searchsorted with side='left')."""
    goals = (events.filter(F.col("is_goal"))
             .select("match_id", F.col("index").alias("g_index"), F.col("team").alias("g_team")))
    diff = (passes.select("event_id", "match_id", "index", "team")
            .join(goals, on="match_id", how="inner")
            .filter(F.col("g_index") < F.col("index"))
            .groupBy("event_id")
            .agg(F.sum(F.when(F.col("g_team") == F.col("team"), 1).otherwise(-1)).alias("goal_diff")))
    return (passes.join(diff, on="event_id", how="left")
            .withColumn("goal_diff", F.coalesce(F.col("goal_diff"), F.lit(0)).cast("long")))


def add_outplayed(passes: DataFrame, frames: DataFrame) -> DataFrame:
    """Thesis 3.1.1: opponents in the 360 frame at pass start whose x lies between pass
    start and end (StatsBomb always attacks left->right). Backward passes -> 0.
    Passes without a 360 frame -> NULL (dropped as missing, as in the thesis)."""
    with_frame = frames.select("event_id").distinct().withColumn("has_frame", F.lit(True))
    opp = frames.filter(~F.col("teammate")).select("event_id", F.col("x").alias("opp_x"))
    lo = F.least("start_x", "end_x")
    hi = F.greatest("start_x", "end_x")
    bypassed = ((F.col("end_x") > F.col("start_x")) & (F.col("opp_x") > lo)
                & (F.col("opp_x") < hi)).cast("long")
    counts = (passes.select("event_id", "start_x", "end_x")
              .join(opp, on="event_id", how="inner")
              .groupBy("event_id")
              .agg(F.sum(bypassed).alias("bypassed")))
    return (passes.join(with_frame, on="event_id", how="left")
            .join(counts, on="event_id", how="left")
            .withColumn("outplayed_players",
                        F.when(F.col("has_frame"), F.coalesce(F.col("bypassed"), F.lit(0)))
                        .cast("double"))
            .drop("has_frame", "bypassed"))


def add_power_index(passes: DataFrame, power_index: DataFrame) -> DataFrame:
    """Opponent strength from the reference table (small, so it is broadcast)."""
    pi = power_index.select(F.col("team").alias("opponent"), F.col("power_index").cast("double"))
    return passes.join(F.broadcast(pi), on="opponent", how="left")


def add_model_features(passes: DataFrame) -> DataFrame:
    """The 13 model features, identical to xpass.features.feature_row."""
    dx = F.col("end_x") - F.col("start_x")
    dy = F.col("end_y") - F.col("start_y")
    # Angle and length use Python's math module in a vectorized pandas UDF, bit-identical to
    # xpass.features. Spark's built-in atan2/hypot (Java) differ in the last bit: one pass of
    # exactly 135 degrees flipped from 'square' to 'backward' and 13% of lengths differed by
    # 1e-14 in the parity check. The UDFs are defined locally so cloudpickle ships them by
    # value: the Spark workers then need only math + pandas, not the xpass package.
    def abs_angle(dy: pd.Series, dx: pd.Series) -> pd.Series:
        return pd.Series([abs(math.atan2(b, a)) for b, a in zip(dy, dx)], dtype="float64")

    def length(dx: pd.Series, dy: pd.Series) -> pd.Series:
        return pd.Series([math.hypot(a, b) for a, b in zip(dx, dy)], dtype="float64")

    angle = F.pandas_udf(abs_angle, "double")(dy, dx)
    direction = (F.when(angle <= math.pi / 4, "forward")
                 .when(angle >= 3 * math.pi / 4, "backward")
                 .otherwise("square"))
    zone = (F.when(F.col("start_x") < 40, "defensive")
            .when(F.col("start_x") < 80, "midfield")
            .otherwise("attacking"))
    height = F.col("height")
    return (passes
            .withColumn("ground_pass", (height == "Ground Pass").cast("long"))
            .withColumn("low_pass", (height == "Low Pass").cast("long"))
            .withColumn("high_pass", (height == "High Pass").cast("long"))
            .withColumn("defensive_third", (zone == "defensive").cast("long"))
            .withColumn("midfield", (zone == "midfield").cast("long"))
            .withColumn("attacking_third", (zone == "attacking").cast("long"))
            .withColumn("backward_pass", (direction == "backward").cast("long"))
            .withColumn("square_pass", (direction == "square").cast("long"))
            .withColumn("forward_pass", (direction == "forward").cast("long"))
            .withColumn("pass_length", F.pandas_udf(length, "double")(dx, dy)))


def build_features(events: DataFrame, frames: DataFrame, power_index: DataFrame,
                   keep_row_id: bool = False) -> DataFrame:
    """Full Spark pipeline: clean -> goal diff -> outplayed players -> power index -> features.
    Rows with a missing feature are dropped, exactly as in the thesis and the pandas version.

    The output keeps the row order of the source file (like pandas). This matters for
    reproducibility: the train/test split is by match, but the CV folds inside
    CalibratedClassifierCV depend on row order, so a reordered table gives slightly
    different (equally valid) models. monotonically_increasing_id() follows file order
    when the input is read from parquet files. Set keep_row_id=True when the result is
    written to a Delta table (which does not guarantee row order): the column
    'source_row' then lets training restore the order with orderBy('source_row')."""
    events = events.withColumn("_row", F.monotonically_increasing_id())
    p = clean(events)
    p = add_goal_diff(p, events)
    p = add_outplayed(p, frames)
    p = add_power_index(p, power_index)
    p = add_model_features(p)
    extra = [F.col("_row").alias("source_row")] if keep_row_id else []
    return (p.dropna(subset=FEATURES).orderBy("_row")
            .select(*META_COLS, *FEATURES, TARGET, *extra))


def read_inputs(spark: SparkSession, raw_dir: str, reference_csv: str):
    events = spark.read.parquet(f"{raw_dir}/events.parquet")
    frames = spark.read.parquet(f"{raw_dir}/frames.parquet")
    power_index = spark.read.csv(reference_csv, header=True, inferSchema=True)
    return events, frames, power_index


def main(tag: str = "euro2022") -> None:
    spark = get_spark()
    events, frames, power_index = read_inputs(
        spark, str(RAW / tag), str(REFERENCE / "power_index_euro2022.csv"))
    df = build_features(events, frames, power_index)
    out = PROCESSED / f"{tag}_passes_spark.parquet"
    df.toPandas().to_parquet(out, index=False)  # small result (~23k rows), one local file
    n, pos = df.count(), df.agg(F.sum(TARGET)).first()[0]
    print(f"{out}: {n} passes, {pos} successful, {n - pos} unsuccessful")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "euro2022")
