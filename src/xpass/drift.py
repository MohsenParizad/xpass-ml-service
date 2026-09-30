"""Step 15: Drift monitoring.

Compares new data (e.g. Women's World Cup 2023) with the training reference stored in
models/metadata.json:
  - data drift:        PSI per feature (continuous: quantile bins, binary: share of 1s)
  - prediction drift:  PSI of the xPass distribution
  - coverage:          share of passes whose opponent has no power index (training-serving gap)
  - performance:       Brier score on the new data (possible here because outcomes are known)

PSI rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 significant drift.

Output: reports/drift_<tag>.md and reports/drift_<tag>.json
Exit code 1 if retraining is recommended (used by the GitHub Action to open an issue).

Usage: python -m xpass.drift wwc2023
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .config import FEATURES, MODELS, PROCESSED, REPORTS, TARGET
from .features import build

PSI_WARN, PSI_ALERT = 0.1, 0.25
MAX_BRIER_INCREASE = 0.02
MAX_UNKNOWN_OPPONENT_SHARE = 0.05


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e, a = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(ref: dict, values: pd.Series) -> float:
    if ref["type"] == "binary":
        e = np.array([1 - ref["share_1"], ref["share_1"]])
        m = values.mean()
        return psi(e, np.array([1 - m, m]))
    edges = np.array(ref["edges"], dtype=float)
    clipped = values.clip(edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    return psi(np.array(ref["shares"]), counts / max(counts.sum(), 1))


def status(v: float) -> str:
    return "ALERT" if v > PSI_ALERT else "watch" if v > PSI_WARN else "ok"


def run(tag: str) -> bool:
    meta = json.loads((MODELS / "metadata.json").read_text())
    ref = meta["reference_stats"]
    model = joblib.load(MODELS / "xpass_model.joblib")

    # Rebuild features for the new data. Keep rows with unknown opponents to measure the gap.
    build(tag)
    summary = json.loads((PROCESSED / f"{tag}_summary.json").read_text())
    new = pd.read_parquet(PROCESSED / f"{tag}_passes.parquet")
    unknown_share = summary["missing_power_index_rows"] / max(summary["in_play_passes"], 1)

    rows = []
    for col in FEATURES:
        v = feature_psi(ref[col], new[col])
        rows.append({"feature": col, "psi": round(v, 4), "status": status(v)})
    p_new = model.predict_proba(new[FEATURES])[:, 1]
    pred_psi = feature_psi(ref["prediction"], pd.Series(p_new))
    brier_new = brier_score_loss(new[TARGET], p_new)
    brier_ref = meta["metrics_test"]["brier"]

    reasons = []
    if any(r["status"] == "ALERT" for r in rows):
        reasons.append("feature drift (PSI > 0.25)")
    if pred_psi > PSI_ALERT:
        reasons.append("prediction drift")
    if brier_new - brier_ref > MAX_BRIER_INCREASE:
        reasons.append(f"performance drop (Brier {brier_ref:.4f} -> {brier_new:.4f})")
    if unknown_share > MAX_UNKNOWN_OPPONENT_SHARE:
        reasons.append(f"{unknown_share:.0%} of passes against opponents without power index")
    retrain = bool(reasons)

    result = {"tag": tag, "model_version": meta["model_version"], "n_passes_scored": len(new),
              "unknown_opponent_share": round(unknown_share, 4), "prediction_psi": round(pred_psi, 4),
              "brier_reference": brier_ref, "brier_new": round(brier_new, 4),
              "features": rows, "retrain_recommended": retrain, "reasons": reasons}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"drift_{tag}.json").write_text(json.dumps(result, indent=2))
    md = [f"# Drift report: {tag} vs. training data (model v{meta['model_version']})", "",
          f"- Passes scored: {len(new)}",
          f"- Passes against opponents without power index: {unknown_share:.1%}",
          f"- Prediction PSI: {pred_psi:.4f} ({status(pred_psi)})",
          f"- Brier score: reference {brier_ref:.4f} -> new {brier_new:.4f}", "",
          "## Feature drift (PSI)", "", pd.DataFrame(rows).to_markdown(index=False), "",
          f"## Decision: {'RETRAINING RECOMMENDED' if retrain else 'no action needed'}",
          *[f"- {r}" for r in reasons]]
    (REPORTS / f"drift_{tag}.md").write_text("\n".join(md))
    print("\n".join(md))
    return retrain


if __name__ == "__main__":
    sys.exit(1 if run(sys.argv[1] if len(sys.argv) > 1 else "wwc2023") else 0)
