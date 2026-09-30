"""Step 10: Online serving with FastAPI.

Endpoints:
  GET  /health            status + model version
  POST /predict           one pass -> xPass (success probability) + risk category
  GET  /players/top?n=10  batch-scored player ranking (thesis Algorithm 1)

Every prediction is logged as one JSON line to stdout. On Cloud Run these lines land in
Cloud Logging and are the raw material for drift monitoring.

Run locally: uvicorn xpass.api:app --reload   (then open http://localhost:8000/docs)
"""
import json
import logging
import sys
import time
from datetime import datetime, timezone
from enum import Enum

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import FEATURES, MODELS, REPORTS
from .features import feature_row, load_power_index

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger("xpass")

MODEL = joblib.load(MODELS / "xpass_model.joblib")
GMM = joblib.load(MODELS / "risk_gmm.joblib")
META = json.loads((MODELS / "metadata.json").read_text())
RISK = {int(k): v for k, v in META["risk_labels"].items()}
POWER_INDEX = load_power_index()
SCORES_PATH = REPORTS / "player_scores.csv"

app = FastAPI(title="xPass Service",
              description="Expected pass success probability (xPass), based on a master's thesis "
                          "on UEFA Women's Euro 2022 StatsBomb data.",
              version=META["model_version"])


class Height(str, Enum):
    ground = "ground"
    low = "low"
    high = "high"


class PassIn(BaseModel):
    """One pass in StatsBomb coordinates (pitch 120 x 80 yards, attacking left to right)."""
    height: Height
    start_x: float = Field(ge=0, le=120)
    start_y: float = Field(ge=0, le=80)
    end_x: float = Field(ge=0, le=120)
    end_y: float = Field(ge=0, le=80)
    outplayed_players: int = Field(ge=0, le=11, description="Opponents bypassed by the pass")
    goal_diff: int = Field(ge=-15, le=15, description="Goals for minus goals against at the moment of the pass")
    opponent: str = Field(description="Opponent team, e.g. 'Norway'")

    model_config = {"json_schema_extra": {"examples": [{
        "height": "ground", "start_x": 60, "start_y": 40, "end_x": 45, "end_y": 38,
        "outplayed_players": 0, "goal_diff": 1, "opponent": "Norway"}]}}


class PassOut(BaseModel):
    xpass: float
    risk_category: str
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": META["model_version"]}


@app.post("/predict", response_model=PassOut)
def predict(p: PassIn):
    if p.opponent not in POWER_INDEX:
        raise HTTPException(422, f"Unknown opponent '{p.opponent}'. Known: {sorted(POWER_INDEX)}")
    row = feature_row(p.height.value, p.start_x, p.start_y, p.end_x, p.end_y,
                      p.outplayed_players, p.goal_diff, POWER_INDEX[p.opponent])
    X = pd.DataFrame([row])[FEATURES]
    t0 = time.perf_counter()
    prob = float(MODEL.predict_proba(X)[0, 1])
    risk = RISK[int(GMM.predict([[prob]])[0])]
    log.info(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(), "event": "prediction",
        "model_version": META["model_version"], "features": row,
        "xpass": round(prob, 5), "risk": risk,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}))
    return PassOut(xpass=round(prob, 4), risk_category=risk, model_version=META["model_version"])


@app.get("/players/top")
def top_players(n: int = Query(10, ge=1, le=100)):
    df = pd.read_csv(SCORES_PATH).head(n)
    return {"model_version": META["model_version"], "players": df.round(4).to_dict("records")}
