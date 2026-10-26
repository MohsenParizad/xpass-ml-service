"""MLflow version of steps 4-9: the same experiments as xpass.train, tracked and registered
with MLflow, so that runs, metrics and model versions are visible in the MLflow UI
(locally or in a Databricks workspace).

What is tracked
  parent run "xpass-training"      data hash, split, selected variant, evaluation gate
  4 nested runs                    logreg/catboost x smote/none: params + test metrics +
                                   calibration plot
  registered model "xpass"         the best variant (lowest Brier), with signature and
                                   input example; alias "champion" only if the gate passed

The training logic itself is imported from xpass.train (split, fit, metrics), so the
numbers are identical to reports/experiments.json.

Usage
  local:       python -m xpass.mlflow_train            (tracking in ./mlflow.db, UI: make mlflow-ui)
  Databricks:  see databricks/02_train_mlflow.py       (tracking + Unity Catalog registry)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.calibration import calibration_curve

from .config import FEATURES, GATE, PROCESSED, RANDOM_STATE, RAW, ROOT, TARGET, TRAIN_TAG
from .train import fit, metrics, split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ
EXPERIMENT = "/Shared/xpass" if IS_DATABRICKS else "xpass"  # notebooks pass /Users/<you>/xpass
MODEL_NAME = "xpass"  # on Databricks with Unity Catalog: "<catalog>.<schema>.xpass"
VARIANTS = [("logreg", "smote"), ("logreg", "none"), ("catboost", "smote"), ("catboost", "none")]


class XPassModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper that returns P(pass completed) as one column.

    A plain sklearn flavour with predict_proba returns both class columns, and
    mlflow.pyfunc.spark_udf(result_type="double") then silently takes column 0 = P(failed).
    The wrapper makes every consumer (Spark UDF, REST serving, pyfunc.load_model) get the
    same, correct number."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return self.model.predict_proba(model_input[FEATURES])[:, 1]


def setup_tracking(tracking_uri: str | None = None, experiment: str | None = None) -> None:
    """Local: SQLite file next to the repo. Databricks: workspace tracking server and the
    Unity Catalog model registry (model names are then <catalog>.<schema>.<model>)."""
    if IS_DATABRICKS:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    else:
        mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment or EXPERIMENT)


def calibration_plot(y, p, title: str) -> plt.Figure:
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    ax.plot(mean_pred, frac, "o-", label="model")
    ax.set(xlabel="mean predicted xPass", ylabel="observed completion rate", title=title,
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def run(df: pd.DataFrame, model_name: str = MODEL_NAME, data_hash: str = "") -> dict:
    train, test = split(df)
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]

    with mlflow.start_run(run_name="xpass-training") as parent:
        mlflow.set_tags({"project": "xpass", "train_tag": TRAIN_TAG,
                         "events_sha256": data_hash, "split": "GroupShuffleSplit by match_id"})
        mlflow.log_params({"n_train": len(train), "n_test": len(test),
                           "n_features": len(FEATURES), "random_state": RANDOM_STATE})

        results, fitted = [], {}
        for kind, resampling in VARIANTS:
            with mlflow.start_run(run_name=f"{kind}-{resampling}", nested=True) as child:
                params = {"model": kind, "resampling": resampling, "calibration": "isotonic",
                          "calibration_cv": 5}
                if kind == "catboost":
                    params.update({"learning_rate": 0.01, "iterations": 800, "depth": 6})
                mlflow.log_params(params)
                model = fit(kind, resampling, Xtr, ytr)
                p = model.predict_proba(Xte)[:, 1]
                m = metrics(yte, p)
                mlflow.log_metrics(m)
                mlflow.log_figure(calibration_plot(yte, p, f"{kind} / {resampling}"),
                                  "calibration.png")
                plt.close("all")
                results.append({"model": kind, "resampling": resampling, **m,
                                "run_id": child.info.run_id})
                fitted[child.info.run_id] = model
                print(f"{kind:9s} {resampling:6s} {m}", file=sys.stderr)

        best = min(results, key=lambda r: r["brier"])
        checks = {"brier": best["brier"] <= GATE["max_brier"],
                  "log_loss": best["log_loss"] <= GATE["max_log_loss"],
                  "roc_auc": best["roc_auc"] >= GATE["min_roc_auc"]}
        passed = all(checks.values())
        mlflow.set_tags({"best_variant": f"{best['model']}-{best['resampling']}",
                         "gate_passed": str(passed)})
        mlflow.log_metrics({f"best_{k}": best[k] for k in ("brier", "log_loss", "roc_auc")})
        mlflow.log_dict({"thresholds": GATE, "checks": checks, "passed": passed}, "gate.json")
        mlflow.log_dict(results, "experiments.json")

        # Log and register the selected model with its input/output signature.
        model = fitted[best["run_id"]]
        example = Xte.head(5)
        signature = infer_signature(example, model.predict_proba(example)[:, 1])
        info = mlflow.pyfunc.log_model(
            name="model", python_model=XPassModel(model), signature=signature,
            input_example=example, registered_model_name=model_name,
            code_paths=[str(Path(__file__).parent)],  # ship the xpass package with the model
            pip_requirements=["catboost==1.2.10", "scikit-learn==1.8.0", "pandas", "numpy"])
        version = info.registered_model_version
        client = MlflowClient()
        client.set_model_version_tag(model_name, version, "gate_passed", str(passed))
        client.set_model_version_tag(model_name, version, "variant",
                                     f"{best['model']}-{best['resampling']}")
        if passed:  # the evaluation gate decides what the scoring job picks up
            client.set_registered_model_alias(model_name, "champion", version)

    summary = {"parent_run_id": parent.info.run_id, "model_uri": info.model_uri,
               "registered": f"{model_name} v{version}", "champion": passed,
               "best": {k: v for k, v in best.items() if k != "run_id"}, "gate": checks}
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(PROCESSED / f"{TRAIN_TAG}_passes.parquet"),
                    help="feature table (pandas or Spark output, they are identical)")
    ap.add_argument("--tracking-uri", default=None)
    args = ap.parse_args()
    setup_tracking(args.tracking_uri)
    manifest = RAW / TRAIN_TAG / "manifest.json"
    data_hash = json.loads(manifest.read_text())["events_sha256"] if manifest.exists() else ""
    summary = run(pd.read_parquet(args.features), data_hash=data_hash)
    sys.exit(0 if summary["champion"] else 1)


if __name__ == "__main__":
    main()
