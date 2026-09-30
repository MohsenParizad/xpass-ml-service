"""Step 1: Download StatsBomb open data (events + 360 frames) for one competition.

Output:
  data/raw/<tag>/events.parquet   one row per relevant event (passes + goals)
  data/raw/<tag>/frames.parquet   360 freeze-frame players for pass events
  data/raw/<tag>/manifest.json    source, match count, content hash (data versioning)

Usage: python -m xpass.ingest euro2022
"""
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from .config import COMPETITIONS, RAW, STATSBOMB_BASE


def _get(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _team(name: str) -> str:
    """Normalise StatsBomb women's team names ('England Women's', 'WNT Finland', ...)."""
    n = name.replace(" Women's", "").replace("WNT ", "").strip()
    return n[:-2] if n.endswith(" W") else n


def _match_rows(match: dict):
    mid = match["match_id"]
    events = _get(f"{STATSBOMB_BASE}/events/{mid}.json")
    try:
        frames = _get(f"{STATSBOMB_BASE}/three-sixty/{mid}.json")
    except Exception:
        frames = []
    home, away = _team(match["home_team"]["home_team_name"]), _team(match["away_team"]["away_team_name"])

    ev_rows, pass_ids = [], set()
    for e in events:
        etype = e["type"]["name"]
        is_goal = (etype == "Shot" and e.get("shot", {}).get("outcome", {}).get("name") == "Goal") \
            or etype == "Own Goal For"
        if etype != "Pass" and not is_goal:
            continue
        team = _team(e["team"]["name"])
        row = {
            "match_id": mid, "event_id": e["id"], "index": e["index"], "period": e["period"],
            "minute": e["minute"], "second": e["second"], "type": etype, "team": team,
            "opponent": away if team == home else home,
            "player": e.get("player", {}).get("name"),
            "position": e.get("position", {}).get("name"),
            "is_goal": bool(is_goal),
        }
        if etype == "Pass":
            p = e["pass"]
            row.update({
                "start_x": e["location"][0], "start_y": e["location"][1],
                "end_x": p["end_location"][0], "end_y": p["end_location"][1],
                "sb_length": p.get("length"), "sb_angle": p.get("angle"),
                "height": p.get("height", {}).get("name"),
                "pass_type": p.get("type", {}).get("name"),
                "outcome": p.get("outcome", {}).get("name"),  # None = completed
            })
            pass_ids.add(e["id"])
        ev_rows.append(row)

    fr_rows = []
    for f in frames:
        if f["event_uuid"] not in pass_ids:
            continue
        for pl in f.get("freeze_frame", []):
            fr_rows.append({
                "event_id": f["event_uuid"], "x": pl["location"][0], "y": pl["location"][1],
                "teammate": pl["teammate"], "actor": pl["actor"], "keeper": pl["keeper"],
            })
    return ev_rows, fr_rows


def ingest(tag: str) -> dict:
    comp = COMPETITIONS[tag]
    matches = _get(f"{STATSBOMB_BASE}/matches/{comp['competition_id']}/{comp['season_id']}.json")
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_match_rows, matches))
    events = pd.DataFrame([r for ev, _ in results for r in ev])
    frames = pd.DataFrame([r for _, fr in results for r in fr])

    out = RAW / tag
    out.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out / "events.parquet", index=False)
    frames.to_parquet(out / "frames.parquet", index=False)

    digest = hashlib.sha256(pd.util.hash_pandas_object(events, index=False).values.tobytes()).hexdigest()
    manifest = {
        "tag": tag, **comp, "source": STATSBOMB_BASE,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matches": len(matches), "events": len(events), "frame_rows": len(frames),
        "events_sha256": digest,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    print(json.dumps(ingest(sys.argv[1] if len(sys.argv) > 1 else "euro2022"), indent=2))
