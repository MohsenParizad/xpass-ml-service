"""Model contract tests: the deployed artifact must match its metadata and pass the gate."""
import json

import joblib
import pandas as pd

from xpass.config import FEATURES, MODELS


def test_metadata_gate_passed():
    meta = json.loads((MODELS / "metadata.json").read_text())
    assert meta["gate"]["passed"] is True
    assert meta["features"] == FEATURES


def test_model_outputs_valid_probabilities():
    model = joblib.load(MODELS / "xpass_model.joblib")
    X = pd.DataFrame([{f: 0 for f in FEATURES} | {"ground_pass": 1, "midfield": 1,
                                                   "square_pass": 1, "pass_length": 10.0,
                                                   "power_index": 20.0}])[FEATURES]
    p = model.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
