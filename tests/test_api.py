import pytest
from fastapi.testclient import TestClient

from xpass.api import app

client = TestClient(app)

EASY = {"height": "ground", "start_x": 30, "start_y": 40, "end_x": 20, "end_y": 38,
        "outplayed_players": 0, "goal_diff": 2, "opponent": "Northern Ireland"}
HARD = {"height": "high", "start_x": 85, "start_y": 20, "end_x": 115, "end_y": 45,
        "outplayed_players": 6, "goal_diff": -1, "opponent": "Germany"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_probability_and_risk():
    r = client.post("/predict", json=EASY)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["xpass"] <= 1.0
    assert body["risk_category"] in {"risky", "medium-risk", "non-risky"}


def test_short_backward_ground_pass_is_easier_than_long_high_forward_pass():
    easy = client.post("/predict", json=EASY).json()["xpass"]
    hard = client.post("/predict", json=HARD).json()["xpass"]
    assert easy > 0.8
    assert hard < 0.5
    assert easy > hard


@pytest.mark.parametrize("patch", [{"start_x": -5}, {"height": "rocket"}, {"outplayed_players": 30}])
def test_invalid_input_is_rejected(patch):
    assert client.post("/predict", json={**EASY, **patch}).status_code == 422


def test_unknown_opponent_is_rejected_with_clear_message():
    r = client.post("/predict", json={**EASY, "opponent": "USA"})
    assert r.status_code == 422
    assert "Unknown opponent" in r.json()["detail"]


def test_top_players():
    r = client.get("/players/top", params={"n": 5})
    assert r.status_code == 200
    assert len(r.json()["players"]) == 5
