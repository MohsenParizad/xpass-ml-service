"""Step 2 + 3: Cleaning and feature engineering (thesis section 3.1, Table 3.2).

The SAME functions are used for training and in the API to avoid training-serving skew.

Output: data/processed/<tag>_passes.parquet  (one row per in-play pass, 13 features + target)
Usage:  python -m xpass.features euro2022
"""
import json
import math
import sys

import numpy as np
import pandas as pd

from .config import (EXCLUDED_OUTCOMES, EXCLUDED_PASS_TYPES, FEATURES, PROCESSED, RAW,
                     REFERENCE, TARGET)

HEIGHTS = {"Ground Pass": "ground", "Low Pass": "low", "High Pass": "high"}


# ---------- single-pass feature logic (shared with the API) ----------
def pass_geometry(start_x: float, start_y: float, end_x: float, end_y: float) -> tuple[float, float]:
    """Length in yards and StatsBomb angle in radians (0 = straight forward)."""
    dx, dy = end_x - start_x, end_y - start_y
    return math.hypot(dx, dy), math.atan2(dy, dx)


def direction(angle: float) -> str:
    """Thesis Fig. 3.2: forward within ±45°, backward beyond ±135°, otherwise square."""
    a = abs(angle)
    if a <= math.pi / 4:
        return "forward"
    if a >= 3 * math.pi / 4:
        return "backward"
    return "square"


def zone(start_x: float) -> str:
    """Pitch thirds by pass origin (StatsBomb pitch length 120)."""
    if start_x < 40:
        return "defensive"
    if start_x < 80:
        return "midfield"
    return "attacking"


def load_power_index(path=REFERENCE / "power_index_euro2022.csv") -> dict[str, float]:
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["power_index"]))


def feature_row(height: str, start_x: float, start_y: float, end_x: float, end_y: float,
                outplayed_players: int, goal_diff: int, power_index: float) -> dict:
    """Build the 13 model features for one pass. height in {ground, low, high}."""
    length, angle = pass_geometry(start_x, start_y, end_x, end_y)
    d, z = direction(angle), zone(start_x)
    return {
        "ground_pass": int(height == "ground"), "low_pass": int(height == "low"),
        "high_pass": int(height == "high"),
        "defensive_third": int(z == "defensive"), "midfield": int(z == "midfield"),
        "attacking_third": int(z == "attacking"),
        "backward_pass": int(d == "backward"), "square_pass": int(d == "square"),
        "forward_pass": int(d == "forward"),
        "pass_length": float(length), "outplayed_players": int(outplayed_players),
        "goal_diff": int(goal_diff), "power_index": float(power_index),
    }


# ---------- dataset-level steps ----------
def clean(events: pd.DataFrame) -> pd.DataFrame:
    """Thesis 3.1.1: in-play passes only; drop offside/unknown/injury clearance."""
    p = events[events["type"] == "Pass"].copy()
    p = p[~p["pass_type"].isin(EXCLUDED_PASS_TYPES)]
    p = p[~p["outcome"].isin(EXCLUDED_OUTCOMES)]
    p = p[p["height"].isin(HEIGHTS)]
    p[TARGET] = p["outcome"].isna().astype(int)  # no outcome = completed pass
    return p


def add_goal_diff(passes: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Goals for minus goals against for the passing team at the moment of the pass."""
    goals = events[events["is_goal"]][["match_id", "index", "team"]]
    diffs = pd.Series(0, index=passes.index, dtype=int)
    for (mid, team), grp in passes.groupby(["match_id", "team"]):
        g = goals[goals["match_id"] == mid]
        own = np.sort(g.loc[g["team"] == team, "index"].to_numpy())
        other = np.sort(g.loc[g["team"] != team, "index"].to_numpy())
        idx = grp["index"].to_numpy()
        diffs.loc[grp.index] = np.searchsorted(own, idx) - np.searchsorted(other, idx)
    return diffs


def add_outplayed(passes: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Thesis 3.1.1: opponents in the 360 frame at pass start whose x lies between pass
    start and end (StatsBomb always attacks left->right). Backward passes -> 0.
    Passes without a 360 frame -> NaN (dropped as missing, as in the thesis)."""
    opp = frames[~frames["teammate"]][["event_id", "x"]]
    has_frame = set(frames["event_id"])
    merged = passes[["event_id", "start_x", "end_x"]].merge(opp, on="event_id", how="left")
    lo = merged[["start_x", "end_x"]].min(axis=1)
    hi = merged[["start_x", "end_x"]].max(axis=1)
    forward = merged["end_x"] > merged["start_x"]
    merged["bypassed"] = (forward & (merged["x"] > lo) & (merged["x"] < hi)).astype(int)
    counts = merged.groupby("event_id")["bypassed"].sum()
    result = passes["event_id"].map(counts).astype(float)
    result[~passes["event_id"].isin(has_frame)] = np.nan
    return result


def build(tag: str, power_index: dict | None = None) -> pd.DataFrame:
    events = pd.read_parquet(RAW / tag / "events.parquet")
    frames = pd.read_parquet(RAW / tag / "frames.parquet")
    power_index = power_index or load_power_index()

    p = clean(events)
    p["goal_diff"] = add_goal_diff(p, events)
    p["outplayed_players"] = add_outplayed(p, frames)
    p["power_index"] = p["opponent"].map(power_index)
    p["height_cat"] = p["height"].map(HEIGHTS)

    feats = pd.DataFrame([
        feature_row(r.height_cat, r.start_x, r.start_y, r.end_x, r.end_y,
                    0 if pd.isna(r.outplayed_players) else r.outplayed_players,
                    r.goal_diff, 0 if pd.isna(r.power_index) else r.power_index)
        for r in p.itertuples()
    ], index=p.index)
    feats.loc[p["outplayed_players"].isna(), "outplayed_players"] = np.nan
    feats.loc[p["power_index"].isna(), "power_index"] = np.nan

    meta_cols = ["match_id", "event_id", "team", "opponent", "player", "position"]
    df = pd.concat([p[meta_cols], feats, p[[TARGET]]], axis=1)
    n_before = len(df)
    missing_power = int(df["power_index"].isna().sum())
    df = df.dropna(subset=FEATURES)  # thesis: entries with missing values excluded

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / f"{tag}_passes.parquet", index=False)
    summary = {
        "tag": tag, "in_play_passes": n_before, "after_dropping_missing": len(df),
        "successful": int(df[TARGET].sum()), "unsuccessful": int((1 - df[TARGET]).sum()),
        "missing_power_index_rows": missing_power,
        "players": int(df["player"].nunique()),
    }
    (PROCESSED / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    return df


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "euro2022"
    build(tag)
    print((PROCESSED / f"{tag}_summary.json").read_text())
