"""PySpark pipeline: same features as the pandas reference implementation (xpass.features).

Small synthetic matches cover the edge cases: excluded set pieces and outcomes, NULL
outcome (= completed), goals before/after the pass, a pass without a 360 frame (dropped),
a frame without opponents in between (0), a backward pass, and an exact 135 degree pass
(where Spark's Java atan2 would disagree with Python's math.atan2).
"""
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from xpass import features as pdf  # noqa: E402
from xpass.config import FEATURES, TARGET  # noqa: E402
from xpass.parity import compare_feature_tables  # noqa: E402
from xpass.spark_features import build_features, get_spark  # noqa: E402

EV_COLS = ["match_id", "event_id", "index", "period", "minute", "second", "type", "team",
           "opponent", "player", "position", "is_goal", "start_x", "start_y", "end_x", "end_y",
           "sb_length", "sb_angle", "height", "pass_type", "outcome"]


def _pass(mid, eid, idx, team, opp, sx, sy, ex, ey, height="Ground Pass", ptype=None, outcome=None):
    return dict(match_id=mid, event_id=eid, index=idx, period=1, minute=idx, second=0, type="Pass",
                team=team, opponent=opp, player=f"{team}-p", position="Center Back", is_goal=False,
                start_x=sx, start_y=sy, end_x=ex, end_y=ey, sb_length=None, sb_angle=None,
                height=height, pass_type=ptype, outcome=outcome)


def _goal(mid, eid, idx, team, opp):
    return dict(match_id=mid, event_id=eid, index=idx, period=1, minute=idx, second=0, type="Shot",
                team=team, opponent=opp, player=f"{team}-s", position="Center Forward", is_goal=True,
                start_x=None, start_y=None, end_x=None, end_y=None, sb_length=None, sb_angle=None,
                height=None, pass_type=None, outcome=None)


@pytest.fixture(scope="module")
def spark():
    return get_spark("xpass-tests")


@pytest.fixture(scope="module")
def data():
    ev = [
        _pass(1, "a", 1, "Spain", "Italy", 30, 40, 50, 40),                    # forward, before goals
        _goal(1, "g1", 2, "Spain", "Italy"),
        _pass(1, "b", 3, "Italy", "Spain", 60, 40, 50, 30, outcome="Incomplete"),  # backward
        _pass(1, "c", 4, "Spain", "Italy", 73.3, 37.5, 63.7, 27.9),            # exactly 135 deg
        _pass(1, "d", 5, "Spain", "Italy", 50, 40, 90, 40, height="High Pass"),
        _pass(1, "e", 6, "Spain", "Italy", 50, 40, 60, 40, ptype="Corner"),    # excluded set piece
        _pass(1, "f", 7, "Spain", "Italy", 50, 40, 60, 40, outcome="Pass Offside"),  # excluded
        _goal(1, "g2", 8, "Italy", "Spain"),
        _goal(1, "g3", 9, "Italy", "Spain"),
        _pass(1, "h", 10, "Spain", "Italy", 85, 10, 100, 30, height="Low Pass"),
        _pass(2, "i", 1, "Norway", "Italy", 20, 40, 30, 70, outcome="Out"),    # square
        _pass(2, "j", 2, "Norway", "Italy", 20, 40, 30, 40),                   # no 360 frame
    ]
    events = pd.DataFrame(ev, columns=EV_COLS)
    fr = []
    for eid in ["a", "b", "c", "d", "h", "i"]:
        fr.append(dict(event_id=eid, x=10.0, y=40.0, teammate=True, actor=True, keeper=False))
    fr += [dict(event_id="a", x=x, y=40.0, teammate=False, actor=False, keeper=False)
           for x in (35, 45, 55)]                                               # 2 bypassed
    fr += [dict(event_id="d", x=x, y=40.0, teammate=False, actor=False, keeper=False)
           for x in (60, 70, 89.9, 90, 95)]                                    # 3 bypassed
    fr += [dict(event_id="b", x=55.0, y=40.0, teammate=False, actor=False, keeper=False)]  # backward -> 0
    frames = pd.DataFrame(fr)
    power = pd.DataFrame({"team": ["Spain", "Italy", "Norway"], "power_index": [20.0, 15.0, 12.0]})
    return events, frames, power


@pytest.fixture(scope="module")
def both(spark, data, tmp_path_factory, monkeypatch_module):
    events, frames, power = data
    ref_path = tmp_path_factory.mktemp("ref") / "pi.csv"
    power.to_csv(ref_path, index=False)
    raw = tmp_path_factory.mktemp("raw")
    (raw / "t").mkdir()
    events.to_parquet(raw / "t" / "events.parquet", index=False)
    frames.to_parquet(raw / "t" / "frames.parquet", index=False)
    monkeypatch_module.setattr(pdf, "RAW", raw)
    monkeypatch_module.setattr(pdf, "PROCESSED", tmp_path_factory.mktemp("processed"))
    reference = pdf.build("t", power_index=dict(zip(power.team, power.power_index)))
    sp = build_features(spark.read.parquet(str(raw / "t" / "events.parquet")),
                        spark.read.parquet(str(raw / "t" / "frames.parquet")),
                        spark.createDataFrame(power)).toPandas()
    return reference, sp


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_spark_matches_pandas_exactly(both):
    reference, sp = both
    report = compare_feature_tables(reference, sp)
    assert report["identical"], report


def test_edge_cases(both):
    _, sp = both
    r = sp.set_index("event_id")
    assert set(r.index) == {"a", "b", "c", "d", "h", "i"}  # set piece, offside, no-frame dropped
    assert r.loc["a", "goal_diff"] == 0 and r.loc["c", "goal_diff"] == 1
    assert r.loc["h", "goal_diff"] == -1 and r.loc["b", "goal_diff"] == -1
    assert r.loc["a", "outplayed_players"] == 2 and r.loc["d", "outplayed_players"] == 3
    assert r.loc["b", "outplayed_players"] == 0 and r.loc["c", "outplayed_players"] == 0
    assert r.loc["c", "square_pass"] == 1  # exact 135 deg, as in math.atan2
    assert r.loc["b", "backward_pass"] == 1 and r.loc["i", "square_pass"] == 1
    assert r.loc["b", TARGET] == 0 and r.loc["a", TARGET] == 1
    assert list(sp.columns[6:-1]) == FEATURES


def test_keep_row_id_restores_order(spark, data):
    events, frames, power = data
    out = build_features(spark.createDataFrame(events), spark.createDataFrame(frames),
                         spark.createDataFrame(power), keep_row_id=True).toPandas()
    assert "source_row" in out and out["source_row"].is_monotonic_increasing
