"""Steps 4-9: split, imbalance experiment, calibrated training, evaluation gate,
risk clusters, player scores, versioned artifacts.

Outputs:
  models/xpass_model.joblib      calibrated classifier (the deployable artifact)
  models/risk_gmm.joblib         Gaussian mixture for risk categories
  models/metadata.json           version, data hash, params, metrics, drift reference
  reports/experiments.json       comparison of all experiment variants
  reports/evaluation.md          human-readable evaluation + gate result
  reports/player_scores.csv      thesis Algorithm 1 (players with >= 51 passes)

Usage: python -m xpass.train [--version 1.0.0]
"""
import argparse
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss, matthews_corrcoef,
                             roc_auc_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit

from .config import (FEATURES, GATE, MIN_PASSES_PER_PLAYER, MODELS, PROCESSED, RANDOM_STATE,
                     RAW, REPORTS, TARGET, TEST_SIZE, TRAIN_TAG)

CONTINUOUS = ["pass_length", "outplayed_players", "goal_diff", "power_index"]


def split(df: pd.DataFrame):
    """Step 4: split BY MATCH so passes from one game never sit in train and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(gss.split(df, groups=df["match_id"]))
    return df.iloc[tr], df.iloc[te]


def base_model(kind: str):
    if kind == "catboost":
        return CatBoostClassifier(learning_rate=0.01, iterations=800, depth=6, verbose=0,
                                  random_seed=RANDOM_STATE, thread_count=4)
    return LogisticRegression(max_iter=1000)


def fit(kind: str, resampling: str, X: pd.DataFrame, y: pd.Series):
    """Thesis setup: base model inside CalibratedClassifierCV(isotonic, cv=5).
    resampling='smote' reproduces the thesis (SMOTE to 1:1 before calibration);
    resampling='none' keeps the original class distribution."""
    if resampling == "smote":
        X, y = SMOTE(sampling_strategy=1.0, random_state=RANDOM_STATE).fit_resample(X, y)
    model = CalibratedClassifierCV(base_model(kind), method="isotonic", cv=5)
    return model.fit(X, y)


def metrics(y, p) -> dict:
    return {
        "brier": round(brier_score_loss(y, p), 4),
        "log_loss": round(log_loss(y, p), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "accuracy": round(accuracy_score(y, p >= 0.5), 4),
        "mcc": round(matthews_corrcoef(y, p >= 0.5), 4),
        "mean_predicted": round(float(np.mean(p)), 4),
        "observed_rate": round(float(np.mean(y)), 4),
    }


def calibration_table(y, p, bins=10) -> list[dict]:
    frac, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return [{"mean_predicted": round(m, 3), "observed": round(f, 3)} for m, f in zip(mean_pred, frac)]


def reference_stats(X: pd.DataFrame, p: np.ndarray) -> dict:
    """Drift reference: quantile bin edges + bin shares per continuous feature,
    shares for binary features, and the prediction distribution."""
    ref = {}
    for col in FEATURES:
        if col in CONTINUOUS:
            edges = np.unique(np.quantile(X[col], np.linspace(0, 1, 11)))
            counts, _ = np.histogram(X[col], bins=edges)
            ref[col] = {"type": "continuous", "edges": edges.round(4).tolist(),
                        "shares": (counts / counts.sum()).round(5).tolist()}
        else:
            ref[col] = {"type": "binary", "share_1": round(float(X[col].mean()), 5)}
    edges = np.linspace(0, 1, 11)
    counts, _ = np.histogram(p, bins=edges)
    ref["prediction"] = {"type": "continuous", "edges": edges.round(2).tolist(),
                         "shares": (counts / counts.sum()).round(5).tolist()}
    return ref


def fit_risk_gmm(p: np.ndarray) -> tuple[GaussianMixture, dict]:
    """Thesis 3.5: 3 risk clusters on predicted success probabilities."""
    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE).fit(p.reshape(-1, 1))
    order = np.argsort(gmm.means_.ravel())  # lowest mean success prob = risky
    labels = {int(order[0]): "risky", int(order[1]): "medium-risk", int(order[2]): "non-risky"}
    return gmm, labels


def player_scores(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Thesis Algorithm 1: score = mean(actual outcome - xPass) for players with >= 51 passes."""
    d = df.assign(xpass=p, residual=df[TARGET] - p)
    g = d.groupby(["player", "team"]).agg(
        position=("position", lambda s: s.mode().iloc[0]),  # most frequent position
        passes=(TARGET, "size"), completed=(TARGET, "sum"), expected=("xpass", "sum"),
        completion_rate=(TARGET, "mean"), score=("residual", "mean")).reset_index()
    return g[g["passes"] >= MIN_PASSES_PER_PLAYER].sort_values("score", ascending=False)


def main(version: str):
    df = pd.read_parquet(PROCESSED / f"{TRAIN_TAG}_passes.parquet")
    train, test = split(df)
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]

    # Step 5-6: experiments (thesis setup vs. alternatives)
    experiments, models = [], {}
    for kind in ["logreg", "catboost"]:
        for resampling in ["smote", "none"]:
            m = fit(kind, resampling, Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            experiments.append({"model": kind, "resampling": resampling, **metrics(yte, p)})
            models[(kind, resampling)] = (m, p)
            print(f"{kind:9s} {resampling:6s} {experiments[-1]}", file=sys.stderr)

    # Step 7: choose the best calibrated model (lowest Brier score)
    best = min(experiments, key=lambda e: e["brier"])
    model, p_test = models[(best["model"], best["resampling"])]
    final_metrics = metrics(yte, p_test)

    # Step 8: evaluation gate
    gate_checks = {
        "brier": final_metrics["brier"] <= GATE["max_brier"],
        "log_loss": final_metrics["log_loss"] <= GATE["max_log_loss"],
        "roc_auc": final_metrics["roc_auc"] >= GATE["min_roc_auc"],
    }
    gate_passed = all(gate_checks.values())

    # Step 9: risk clusters + player scores on all passes (batch scoring)
    p_train = model.predict_proba(Xtr)[:, 1]
    gmm, risk_labels = fit_risk_gmm(p_train)
    p_all = model.predict_proba(df[FEATURES])[:, 1]
    scores = player_scores(df, p_all)

    # Artifacts
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "xpass_model.joblib")
    joblib.dump(gmm, MODELS / "risk_gmm.joblib")
    manifest = json.loads((RAW / TRAIN_TAG / "manifest.json").read_text())
    metadata = {
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_data": {"tag": TRAIN_TAG, "events_sha256": manifest["events_sha256"],
                          "n_train": len(train), "n_test": len(test),
                          "split": "GroupShuffleSplit by match_id"},
        "features": FEATURES,
        "model": {"base": best["model"], "resampling": best["resampling"],
                  "calibration": "isotonic, cv=5",
                  "catboost_params": {"learning_rate": 0.01, "iterations": 800, "depth": 6}},
        "metrics_test": final_metrics,
        "gate": {"thresholds": GATE, "checks": gate_checks, "passed": gate_passed},
        "risk_labels": risk_labels,
        "reference_stats": reference_stats(Xtr, p_train),
    }
    (MODELS / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (REPORTS / "experiments.json").write_text(json.dumps(experiments, indent=2))
    scores.to_csv(REPORTS / "player_scores.csv", index=False)
    write_report(experiments, best, final_metrics, calibration_table(yte, p_test),
                 gate_checks, gate_passed, scores, version)
    print(json.dumps({"best": best, "gate_passed": gate_passed}, indent=2))
    sys.exit(0 if gate_passed else 1)


def write_report(experiments, best, m, calib, checks, passed, scores, version):
    exp = pd.DataFrame(experiments)
    lines = [
        f"# xPass model evaluation (v{version})", "",
        "## Experiments (test set, split by match)", "", exp.to_markdown(index=False), "",
        f"**Selected:** {best['model']} with resampling = `{best['resampling']}` (lowest Brier score)", "",
        "## Calibration (quantile bins)", "", pd.DataFrame(calib).to_markdown(index=False), "",
        "## Evaluation gate", "",
        *[f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()],
        f"\n**Gate result: {'PASSED - model may be deployed' if passed else 'FAILED - deployment blocked'}**", "",
        f"## Top 10 players (xPass score, >= {MIN_PASSES_PER_PLAYER} passes)", "",
        scores.head(10).round(4).to_markdown(index=False),
    ]
    (REPORTS / "evaluation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    main(ap.parse_args().version)
