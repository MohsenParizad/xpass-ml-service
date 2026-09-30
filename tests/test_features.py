import math

import pytest

from xpass.config import FEATURES
from xpass.features import direction, feature_row, pass_geometry, zone


@pytest.mark.parametrize("angle,expected", [
    (0.0, "forward"), (0.7, "forward"), (-0.7, "forward"),
    (math.pi / 2, "square"), (-math.pi / 2, "square"),
    (math.pi, "backward"), (-2.5, "backward"),
])
def test_direction_follows_thesis_figure_3_2(angle, expected):
    assert direction(angle) == expected


@pytest.mark.parametrize("x,expected", [(0, "defensive"), (39.9, "defensive"), (40, "midfield"),
                                        (79.9, "midfield"), (80, "attacking"), (120, "attacking")])
def test_zone_thirds(x, expected):
    assert zone(x) == expected


def test_geometry_matches_statsbomb_convention():
    length, angle = pass_geometry(60, 40, 70, 40)
    assert length == pytest.approx(10)
    assert angle == pytest.approx(0)  # straight forward


def test_feature_row_has_all_features_and_one_hot_groups_sum_to_one():
    row = feature_row("low", 60, 40, 75, 30, outplayed_players=2, goal_diff=1, power_index=15.0)
    assert list(row) == FEATURES
    assert row["ground_pass"] + row["low_pass"] + row["high_pass"] == 1
    assert row["defensive_third"] + row["midfield"] + row["attacking_third"] == 1
    assert row["backward_pass"] + row["square_pass"] + row["forward_pass"] == 1
    assert row["low_pass"] == 1 and row["midfield"] == 1 and row["forward_pass"] == 1
