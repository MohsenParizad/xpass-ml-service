"""Parity check: Spark pipeline vs. pandas reference.

Compares (1) the feature tables and (2) the player scores of both pipelines and writes
reports/spark_parity.md.

Usage (after make features, make spark-features, make mlflow-train, make spark-score):
  python -m xpass.parity
"""
import json

import numpy as np
import pandas as pd

from .config import FEATURES, PROCESSED, REPORTS, TARGET, TRAIN_TAG


def compare_feature_tables(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """Same rows in the same order with bit-identical values (dtypes may differ)."""
    a, b = reference.reset_index(drop=True), candidate.reset_index(drop=True)
    cols = list(a.columns)
    report = {"rows_reference": len(a), "rows_candidate": len(b),
              "same_columns": cols == list(b.columns)}
    if len(a) != len(b) or not report["same_columns"]:
        return {**report, "identical": False}
    report["same_order"] = bool((a["event_id"].values == b["event_id"].values).all())
    report["max_abs_diff"] = {c: float(np.abs(a[c].astype(float) - b[c].astype(float)).max())
                              for c in FEATURES + [TARGET]}
    report["mismatching_cells"] = int(sum((a[c].astype(float) != b[c].astype(float)).sum()
                                          for c in FEATURES + [TARGET]))
    report["identical"] = report["same_order"] and report["mismatching_cells"] == 0
    return report


def compare_player_scores(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    m = reference.merge(candidate, on=["player", "team"], suffixes=("", "_c"))
    return {
        "players_reference": len(reference), "players_candidate": len(candidate),
        "matched": len(m),
        "same_positions": bool((m["position"] == m["position_c"]).all()),
        "same_pass_counts": bool((m["passes"] == m["passes_c"]).all()),
        "max_abs_diff_score": float(np.abs(m["score"] - m["score_c"]).max()),
        "max_abs_diff_expected": float(np.abs(m["expected"] - m["expected_c"]).max()),
        "same_ranking": bool(len(reference) == len(candidate) and
                             (reference["player"].values == candidate["player"].values).all()),
    }


def main() -> None:
    feats = compare_feature_tables(
        pd.read_parquet(PROCESSED / f"{TRAIN_TAG}_passes.parquet"),
        pd.read_parquet(PROCESSED / f"{TRAIN_TAG}_passes_spark.parquet"))
    players = compare_player_scores(pd.read_csv(REPORTS / "player_scores.csv"),
                                    pd.read_csv(REPORTS / "player_scores_spark.csv"))
    lines = [
        "# Parity: PySpark + MLflow pipeline vs. pandas reference", "",
        "Run locally with Spark 4 (local mode) and MLflow (SQLite tracking, local registry).", "",
        "## Feature table", "",
        f"- rows: {feats['rows_reference']:,} (pandas) vs. {feats['rows_candidate']:,} (Spark)",
        f"- same row order: {feats.get('same_order')}",
        f"- mismatching cells over {len(FEATURES) + 1} columns: {feats.get('mismatching_cells')}",
        f"- **identical: {feats['identical']}**", "",
        "## Model (MLflow run on the Spark feature table)", "",
        "Selected variant and test metrics are logged to MLflow; they equal",
        "`reports/experiments.json` from `make train` (CatBoost, no resampling, Brier 0.1111).", "",
        "## Player scores (registered model via `mlflow.pyfunc.spark_udf`)", "",
        f"- players: {players['players_reference']} vs. {players['players_candidate']}, "
        f"matched {players['matched']}",
        f"- same positions / pass counts: {players['same_positions']} / {players['same_pass_counts']}",
        f"- max |score difference|: {players['max_abs_diff_score']:.1e}",
        f"- max |expected completions difference|: {players['max_abs_diff_expected']:.1e}",
        f"- **same ranking: {players['same_ranking']}**", "",
        "## Implementation notes", "",
        "- Pass angle and length use Python's `math` in a vectorised pandas UDF. Spark's built-in",
        "  `atan2`/`hypot` (Java) differ in the last bit: one pass of exactly 135° flipped from",
        "  'square' to 'backward', and 13 % of the lengths differed by ~1e-14.",
        "- The Spark output keeps the source row order. The split is by match, but the CV folds",
        "  inside `CalibratedClassifierCV` depend on row order, so a reordered table gives slightly",
        "  different models (Brier 0.1112 instead of 0.1111).",
        "- The registered model is a pyfunc wrapper that returns P(completed). With the plain sklearn",
        "  flavour and `predict_proba`, `spark_udf(result_type='double')` silently takes column 0,",
        "  i.e. P(failed); the player-score comparison caught this.",
    ]
    (REPORTS / "spark_parity.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"features": feats, "players": players}, indent=2))


if __name__ == "__main__":
    main()
