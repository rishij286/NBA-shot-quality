"""Shared paths, schema, and training defaults for V1."""

from pathlib import Path

# Project root: .../nba-shot-quality (three levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "shot_logs.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "shot_quality_v1.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "metrics_v1.json"

REQUIRED_COLUMNS = [
    "GAME_ID",
    "MATCHUP",
    "LOCATION",
    "W",
    "FINAL_MARGIN",
    "SHOT_NUMBER",
    "PERIOD",
    "GAME_CLOCK",
    "SHOT_CLOCK",
    "DRIBBLES",
    "TOUCH_TIME",
    "SHOT_DIST",
    "PTS_TYPE",
    "SHOT_RESULT",
    "CLOSEST_DEFENDER",
    "CLOSEST_DEFENDER_PLAYER_ID",
    "CLOSE_DEF_DIST",
    "FGM",
    "PTS",
    "player_name",
    "player_id",
]

TARGET_COLUMN = "FGM"

# Numeric columns coerced after load
NUMERIC_COLUMNS = [
    "FINAL_MARGIN",
    "SHOT_NUMBER",
    "PERIOD",
    "SHOT_CLOCK",
    "DRIBBLES",
    "TOUCH_TIME",
    "SHOT_DIST",
    "PTS_TYPE",
    "CLOSEST_DEFENDER_PLAYER_ID",
    "CLOSE_DEF_DIST",
    "FGM",
    "PTS",
    "player_id",
]

# Pre-shot features used by the model (after feature engineering)
MODEL_FEATURE_COLUMNS = [
    "SHOT_DIST",
    "CLOSE_DEF_DIST",
    "SHOT_CLOCK",
    "TOUCH_TIME",
    "DRIBBLES",
    "PERIOD",
    "PTS_TYPE",
    "is_home",
    "game_clock_seconds",
    "is_three",
    "shot_clock_missing",
]

# Columns that must never be used as inputs (outcome or post-game leakage)
LEAKAGE_COLUMNS = ["SHOT_RESULT", "FGM", "PTS", "W", "FINAL_MARGIN"]

# Kept for later player/team analysis; not model inputs in V1
ANALYSIS_COLUMNS = [
    "GAME_ID",
    "MATCHUP",
    "player_name",
    "player_id",
    "CLOSEST_DEFENDER",
    "CLOSEST_DEFENDER_PLAYER_ID",
]

RANDOM_SEED = 42
TEST_SIZE = 0.2
