"""Explainability for the production model: SHAP, compared with the built-in importance and permutation importance.

Questions this answers
  1. Which features drive xPass overall, and in which direction?         (global SHAP)
  2. Why did the model rate one specific pass as easy or hard?             (local SHAP)
  3. Do different explanation methods agree?                              (SHAP vs CatBoost built-in vs permutation)
  4. What about one-hot encoded features (height, zone, direction)?       (grouped SHAP + grouped permutation)

Two technical points worth knowing
  * The deployed model is CalibratedClassifierCV: 5 CatBoost models, each followed by an isotonic
    calibrator, averaged. SHAP explains each CatBoost model's raw output (log-odds) exactly; the
    calibration step is a monotone transformation applied afterwards. So SHAP says *why the score is
    high or low*, but the contributions do not add up to the final calibrated probability.
    We average the SHAP values of the 5 models (SHAP is additive, so the average is again consistent).
  * One-hot features must be judged as groups: permuting 'low_pass' alone creates impossible passes
    (low AND ground at the same time), and a single dummy's SHAP value depends on which category
    was left as the reference. Summing SHAP values within a group is allowed because SHAP is additive.

Output: reports/explainability.md, reports/explainability.json, docs/shap_*.png
Usage:  PYTHONPATH=src python scripts/explain.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from xpass.config import FEATURES, MODELS, PROCESSED, RANDOM_STATE, REPORTS, TARGET
from xpass.train import split

DOCS = Path(__file__).resolve().parents[1] / "docs"
GROUPS = {
    "pass height": ["ground_pass", "low_pass", "high_pass"],
    "pitch zone": ["defensive_third", "midfield", "attacking_third"],
    "direction": ["backward_pass", "square_pass", "forward_pass"],
    "pass length": ["pass_length"],
    "outplayed players": ["outplayed_players"],
    "goal difference": ["goal_diff"],
    "power index": ["power_index"],
}
# The two example passes discussed with LIME in the thesis (Table 3.8). The thesis only recorded
# whether opponents were outplayed (yes/no); for the second pass we assume one outplayed opponent.
EXAMPLES = {
    "A: low forward pass, midfield, vs Netherlands": dict(
        ground_pass=0, low_pass=1, high_pass=0, defensive_third=0, midfield=1, attacking_third=0,
        backward_pass=0, square_pass=0, forward_pass=1, pass_length=1.118, outplayed_players=0,
        goal_diff=1, power_index=7.5),
    "B: ground backward pass, midfield, vs Norway": dict(
        ground_pass=1, low_pass=0, high_pass=0, defensive_third=0, midfield=1, attacking_third=0,
        backward_pass=1, square_pass=0, forward_pass=0, pass_length=19.3, outplayed_players=1,
        goal_diff=4, power_index=43.33),
}


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Average SHAP values (log-odds) and base value over the 5 CatBoost models inside the calibrator."""
    values, bases = [], []
    for cc in model.calibrated_classifiers_:
        explainer = shap.TreeExplainer(cc.estimator)
        values.append(explainer.shap_values(X))
        bases.append(float(np.ravel(explainer.expected_value)[0]))
    return np.mean(values, axis=0), float(np.mean(bases))


def raw_logodds(model, X: pd.DataFrame) -> np.ndarray:
    """Mean raw output of the 5 CatBoost models: what SHAP explains (before calibration)."""
    return np.mean([cc.estimator.predict(X, prediction_type="RawFormulaVal")
                    for cc in model.calibrated_classifiers_], axis=0)


def builtin_importance(model) -> pd.Series:
    """CatBoost's own importance (PredictionValuesChange), as reported in thesis Table 3.10."""
    imp = np.mean([cc.estimator.get_feature_importance() for cc in model.calibrated_classifiers_], axis=0)
    return pd.Series(imp, index=FEATURES)


def grouped_permutation_importance(model, X: pd.DataFrame, y: pd.Series, repeats: int = 5) -> pd.Series:
    """Increase in Brier score when a whole feature group is shuffled together (keeps one-hot rows valid)."""
    rng = np.random.default_rng(RANDOM_STATE)
    base = np.mean((model.predict_proba(X)[:, 1] - y) ** 2)
    out = {}
    for name, cols in GROUPS.items():
        losses = []
        for _ in range(repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[idx]
            losses.append(np.mean((model.predict_proba(Xp)[:, 1] - y) ** 2) - base)
        out[name] = float(np.mean(losses))
    return pd.Series(out)


def direction_table(X: pd.DataFrame, sv: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, f in enumerate(FEATURES):
        col = X[f].to_numpy()
        if set(np.unique(col)) <= {0, 1}:
            on, off = sv[col == 1, j].mean(), sv[col == 0, j].mean()
            rows.append({"feature": f, "type": "binary", "mean SHAP if 1": round(on, 3),
                         "mean SHAP if 0": round(off, 3),
                         "effect": "raises xPass" if on > off else "lowers xPass"})
        else:
            rho = spearmanr(col, sv[:, j]).statistic
            rows.append({"feature": f, "type": "numeric", "mean SHAP if 1": None, "mean SHAP if 0": None,
                         "effect": f"{'higher value raises' if rho > 0 else 'higher value lowers'} xPass "
                                   f"(Spearman {rho:+.2f})"})
    return pd.DataFrame(rows)


def main():
    model = joblib.load(MODELS / "xpass_model.joblib")
    df = pd.read_parquet(PROCESSED / "euro2022_passes.parquet")
    _, test = split(df)
    X, y = test[FEATURES].reset_index(drop=True), test[TARGET].reset_index(drop=True)

    sv, base = shap_values(model, X)
    additivity_error = float(np.max(np.abs(base + sv.sum(axis=1) - raw_logodds(model, X))))

    shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES)
    builtin = builtin_importance(model)
    grouped_shap = pd.Series({g: float(np.abs(sv[:, [FEATURES.index(c) for c in cols]].sum(axis=1)).mean())
                              for g, cols in GROUPS.items()})
    grouped_perm = grouped_permutation_importance(model, X, y)
    rho_features = spearmanr(shap_imp, builtin).statistic
    rho_groups = spearmanr(grouped_shap, grouped_perm[grouped_shap.index]).statistic
    directions = direction_table(X, sv)

    # Local explanations
    ex = pd.DataFrame(list(EXAMPLES.values()))[FEATURES]
    ex_sv, _ = shap_values(model, ex)
    ex_raw = raw_logodds(model, ex)
    ex_cal = model.predict_proba(ex)[:, 1]
    local = {}
    for i, name in enumerate(EXAMPLES):
        contrib = pd.Series(ex_sv[i], index=FEATURES).sort_values(key=np.abs, ascending=False)
        local[name] = {"base_logodds": round(base, 3), "raw_logodds": round(float(ex_raw[i]), 3),
                       "uncalibrated_probability": round(float(1 / (1 + np.exp(-ex_raw[i]))), 3),
                       "calibrated_xpass": round(float(ex_cal[i]), 3),
                       "top_contributions": {k: round(float(v), 3) for k, v in contrib.head(6).items()}}

    # Plots
    DOCS.mkdir(exist_ok=True)
    shap.summary_plot(sv, X, show=False, max_display=13)
    plt.title("SHAP values per pass (log-odds of success), test matches")
    plt.tight_layout(); plt.savefig(DOCS / "shap_summary.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    grouped_shap.sort_values().plot.barh(ax=ax[0], color="#1F6F78")
    ax[0].set_title("Grouped SHAP: mean |contribution| (log-odds)")
    grouped_perm.sort_values().plot.barh(ax=ax[1], color="#9C6B30")
    ax[1].set_title("Grouped permutation: Brier increase")
    plt.tight_layout(); plt.savefig(DOCS / "shap_grouped.png", dpi=150); plt.close()

    shap.dependence_plot("pass_length", sv, X, interaction_index="forward_pass", show=False)
    plt.title("Pass length: effect on xPass, coloured by forward pass")
    plt.tight_layout(); plt.savefig(DOCS / "shap_dependence_pass_length.png", dpi=150); plt.close()

    for i, name in enumerate(EXAMPLES):
        e = shap.Explanation(values=ex_sv[i], base_values=base, data=ex.iloc[i].to_numpy(), feature_names=FEATURES)
        shap.plots.waterfall(e, max_display=8, show=False)
        plt.title(f"{name}\ncalibrated xPass = {ex_cal[i]:.2f}", fontsize=10)
        plt.tight_layout(); plt.savefig(DOCS / f"shap_local_{name[0]}.png", dpi=150); plt.close()

    # Report
    REPORTS.mkdir(exist_ok=True)
    result = {"n_test_passes": len(X), "base_logodds": base, "additivity_max_error": additivity_error,
              "shap_importance": shap_imp.round(4).to_dict(), "builtin_importance": builtin.round(3).to_dict(),
              "grouped_shap": grouped_shap.round(4).to_dict(), "grouped_permutation_brier": grouped_perm.round(5).to_dict(),
              "spearman_shap_vs_builtin": round(float(rho_features), 3),
              "spearman_grouped_shap_vs_permutation": round(float(rho_groups), 3),
              "directions": directions.to_dict("records"), "local": local}
    (REPORTS / "explainability.json").write_text(json.dumps(result, indent=2))

    comp = pd.DataFrame({"SHAP mean abs": shap_imp.round(3), "SHAP rank": shap_imp.rank(ascending=False).astype(int),
                         "built-in": builtin.round(2), "built-in rank": builtin.rank(ascending=False).astype(int)}
                        ).sort_values("SHAP rank")
    grp = pd.DataFrame({"grouped SHAP": grouped_shap.round(3), "SHAP rank": grouped_shap.rank(ascending=False).astype(int),
                        "permutation (Brier +)": grouped_perm.round(4),
                        "perm. rank": grouped_perm.rank(ascending=False).astype(int)}).sort_values("SHAP rank")
    md = ["# Explainability report", "",
          f"Production model, {len(X):,} passes from held-out test matches. SHAP values are in log-odds of "
          "pass success, averaged over the 5 CatBoost models inside the calibrator.", "",
          f"Additivity check: base value + sum of SHAP values reproduces the raw model output with a maximum "
          f"error of {additivity_error:.1e}.", "",
          "## 1. Individual features: SHAP vs CatBoost built-in importance (thesis Table 3.10)", "",
          comp.to_markdown(), "",
          f"Rank agreement (Spearman): {rho_features:.2f}.", "",
          "## 2. Feature groups: grouped SHAP vs grouped permutation importance", "",
          "One-hot features are evaluated as groups (summed SHAP; columns shuffled together).", "",
          grp.to_markdown(), "",
          f"Rank agreement (Spearman): {rho_groups:.2f}.", "",
          "## 3. Direction of each feature's effect", "", directions.to_markdown(index=False), "",
          "## 4. Two passes explained (the LIME examples from the thesis)", ""]
    for name, d in local.items():
        md += [f"**{name}**: base {d['base_logodds']:+.2f} → raw {d['raw_logodds']:+.2f} log-odds "
               f"(uncalibrated p = {d['uncalibrated_probability']:.2f}) → calibrated xPass = {d['calibrated_xpass']:.2f}", "",
               "| feature | SHAP |", "|---|---|"] + [f"| {k} | {v:+.3f} |" for k, v in d["top_contributions"].items()] + [""]
    # Numeric features can act non-monotonically; show SHAP and observed success by pass-length bin.
    bins = pd.cut(X["pass_length"], [0, 3, 5, 8, 12, 20, 30, 45, 130])
    jl = FEATURES.index("pass_length")
    lt = pd.DataFrame({"passes": y.groupby(bins, observed=True).size(),
                       "observed success": y.groupby(bins, observed=True).mean().round(3),
                       "mean SHAP of pass_length": pd.Series(sv[:, jl]).groupby(bins.to_numpy(), observed=True).mean().round(3)})
    result["pass_length_bins"] = {str(k): v for k, v in lt.to_dict("index").items()}
    (REPORTS / "explainability.json").write_text(json.dumps(result, indent=2))
    md += ["## 5. Pass length is not monotonic", "",
           "A single correlation hides the shape: very short 'passes' (often blocked or miscontrolled) mostly fail, "
           "medium passes are safest, long passes get harder again.", "", lt.to_markdown(), ""]
    md += ["## Plots", "", "![summary](../docs/shap_summary.png)", "", "![grouped](../docs/shap_grouped.png)", "",
           "![dependence](../docs/shap_dependence_pass_length.png)", "",
           "![A](../docs/shap_local_A.png) ![B](../docs/shap_local_B.png)", ""]
    (REPORTS / "explainability.md").write_text("\n".join(md))
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
