"""MLflow model contract: the logged pyfunc returns P(completed) as a single column.

Guards against the silent failure mode of the plain sklearn flavour with predict_proba,
where mlflow.pyfunc.spark_udf(result_type='double') takes column 0 = P(failed).
"""
import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from xpass.config import FEATURES  # noqa: E402
from xpass.mlflow_train import XPassModel  # noqa: E402


def test_pyfunc_returns_probability_of_success(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, len(FEATURES))), columns=FEATURES)
    y = (X["pass_length"] < 0.7).astype(int)  # short passes succeed
    sk = LogisticRegression().fit(X, y)

    path = tmp_path / "model"
    mlflow.pyfunc.save_model(str(path), python_model=XPassModel(sk), input_example=X.head(2))
    p = np.asarray(mlflow.pyfunc.load_model(str(path)).predict(X)).ravel()

    assert p.shape == (len(X),)
    np.testing.assert_allclose(p, sk.predict_proba(X)[:, 1])
    assert p[X["pass_length"] < 0.2].mean() > p[X["pass_length"] > 0.9].mean()
