"""Central configuration: paths, data sources, model settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REFERENCE = DATA / "reference"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# StatsBomb competition/season IDs (check with competitions.json)
COMPETITIONS = {
    "euro2022": {"competition_id": 53, "season_id": 106},  # training data (thesis)
    "wwc2023": {"competition_id": 72, "season_id": 107},   # new data for drift check
    "euro2025": {"competition_id": 53, "season_id": 315},  # new data for drift check
}
TRAIN_TAG = "euro2022"

# Set pieces excluded as in thesis section 3.1.1
EXCLUDED_PASS_TYPES = {"Free Kick", "Kick Off", "Throw-in", "Goal Kick", "Corner"}
# Outcomes excluded (offside, unknown, deliberate injury clearance)
EXCLUDED_OUTCOMES = {"Pass Offside", "Unknown", "Injury Clearance"}

FEATURES = [
    "ground_pass", "low_pass", "high_pass",
    "defensive_third", "midfield", "attacking_third",
    "backward_pass", "square_pass", "forward_pass",
    "pass_length", "outplayed_players", "goal_diff", "power_index",
]
TARGET = "pass_success"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_PASSES_PER_PLAYER = 51  # median threshold from thesis

# Evaluation gate: CI fails and nothing is deployed if the model is worse
GATE = {"max_brier": 0.13, "max_log_loss": 0.42, "min_roc_auc": 0.84}
