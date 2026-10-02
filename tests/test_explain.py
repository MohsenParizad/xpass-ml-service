"""SHAP contract: explanations must add up exactly to what the model computes (before calibration)."""
import importlib.util

import joblib
import numpy as np
import pandas as pd
import pytest

from xpass.config import FEATURES, MODELS

pytestmark = pytest.mark.skipif(importlib.util.find_spec("shap") is None, reason="shap not installed")


@pytest.fixture(scope="module")
def setup():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import explain
    model = joblib.load(MODELS / "xpass_model.joblib")
    X = pd.DataFrame(list(explain.EXAMPLES.values()))[FEATURES]
    return explain, model, X


def test_shap_values_add_up_to_raw_model_output(setup):
    explain, model, X = setup
    sv, base = explain.shap_values(model, X)
    assert np.allclose(base + sv.sum(axis=1), explain.raw_logodds(model, X), atol=1e-6)


def test_easy_pass_is_explained_as_easy(setup):
    explain, model, X = setup
    sv, base = explain.shap_values(model, X)
    assert base + sv[1].sum() > 0 > base + sv[0].sum()   # B (backward ground pass) easy, A (1-yard pass) hard


def test_groups_cover_every_feature_once(setup):
    explain, _, _ = setup
    cols = [c for cols in explain.GROUPS.values() for c in cols]
    assert sorted(cols) == sorted(FEATURES)
