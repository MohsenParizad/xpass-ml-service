"""Methodology checks: split, feature and coverage ablations.

1. Random vs. grouped 5-fold CV (is a random split optimistic?)       -> reports/ablation_cv.json
2. Grouped CV without power_index (does the feature add signal?)      -> reports/ablation_cv.json
3. World Cup 2023: production model vs. model without power_index,
   on known opponents and on all passes (coverage + selection bias)   -> reports/ablation_wwc2023_no_power.json

Needs processed euro2022 data and raw wwc2023 data (make features; python -m xpass.ingest wwc2023).
Usage: PYTHONPATH=src python scripts/ablations.py        (about 5 minutes on 4 cores)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from xpass.config import FEATURES, MODELS, PROCESSED, RANDOM_STATE, REPORTS, TARGET
from xpass.features import build
from xpass.train import fit, metrics, split

NO_POWER = [f for f in FEATURES if f != "power_index"]
KEYS = ["brier", "log_loss", "roc_auc", "mean_predicted", "observed_rate"]


def cv_comparison(df):
    X, y, g = df[FEATURES], df[TARGET], df["match_id"]
    setups = [("stratified_kfold", StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), FEATURES),
              ("group_kfold", GroupKFold(5), FEATURES),
              ("group_kfold_no_power", GroupKFold(5), NO_POWER)]
    res = {}
    for name, cv, feats in setups:
        folds = []
        for tr, te in cv.split(X, y, g):
            m = fit("catboost", "none", X.iloc[tr][feats], y.iloc[tr])
            folds.append(metrics(y.iloc[te], m.predict_proba(X.iloc[te][feats])[:, 1]))
        res[name] = {k: {"mean": round(float(np.mean([f[k] for f in folds])), 4),
                         "sd": round(float(np.std([f[k] for f in folds])), 4)}
                     for k in ["brier", "log_loss", "roc_auc"]}
        print(name, res[name], flush=True)
    return res


class _Zero(dict):
    """Power-index lookup that returns 0 for every team, so no pass is dropped."""
    def __missing__(self, key):
        return 0.0


def wwc_no_power(euro):
    known = pd.read_parquet(PROCESSED / "wwc2023_passes.parquet")
    backup = {p: p.read_bytes() for p in PROCESSED.glob("wwc2023_*")}
    try:
        everything = build("wwc2023", power_index=_Zero({"__dummy__": 0.0}))
    finally:  # restore the regular processed files used by drift.py
        for p, b in backup.items():
            p.write_bytes(b)
    train, _ = split(euro)
    no_power = fit("catboost", "none", train[NO_POWER], train[TARGET])
    prod = joblib.load(MODELS / "xpass_model.joblib")
    pick = lambda d: {k: d[k] for k in KEYS}
    return {"passes_known": len(known), "passes_all": len(everything),
            "production_on_known": pick(metrics(known[TARGET], prod.predict_proba(known[FEATURES])[:, 1])),
            "no_power_on_known": pick(metrics(known[TARGET], no_power.predict_proba(known[NO_POWER])[:, 1])),
            "no_power_on_all": pick(metrics(everything[TARGET], no_power.predict_proba(everything[NO_POWER])[:, 1]))}


if __name__ == "__main__":
    euro = pd.read_parquet(PROCESSED / "euro2022_passes.parquet").reset_index(drop=True)
    (REPORTS / "ablation_cv.json").write_text(json.dumps(cv_comparison(euro), indent=2))
    wwc = wwc_no_power(euro)
    print(json.dumps(wwc, indent=2))
    (REPORTS / "ablation_wwc2023_no_power.json").write_text(json.dumps(wwc, indent=2))
