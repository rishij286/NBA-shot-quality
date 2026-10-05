"""Lightweight tests for cleaning, features, and metrics."""

from __future__ import annotations

import pandas as pd

from nba_shot_quality.data import clean_shot_logs, game_level_train_test_split
from nba_shot_quality.evaluate import evaluate_predictions, select_best_model
from nba_shot_quality.features import get_feature_matrix, get_target


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": [1, 1, 2, 2, 3, 3],
            "MATCHUP": ["OCT 01, 2014 - BOS @ NYK"] * 6,
            "LOCATION": ["A", "A", "H", "H", "A", "H"],
            "W": ["W", "W", "L", "L", "W", "L"],
            "FINAL_MARGIN": [5, 5, -3, -3, 10, -2],
            "SHOT_NUMBER": [1, 2, 1, 2, 1, 2],
            "PERIOD": [1, 1, 2, 2, 1, 4],
            "GAME_CLOCK": ["10:00", "0:05", "5:30", "1:00", "8:15", "0:00"],
            "SHOT_CLOCK": [12.0, None, 8.0, 3.0, None, 1.0],
            "DRIBBLES": [0, 2, 1, 5, 0, 3],
            "TOUCH_TIME": [0.8, -1.5, 2.0, 4.0, 0.5, 3.0],
            "SHOT_DIST": [5.0, 24.0, 12.0, 18.0, 3.0, 26.0],
            "PTS_TYPE": [2, 3, 2, 2, 2, 3],
            "SHOT_RESULT": ["made", "missed", "made", "missed", "made", "missed"],
            "CLOSEST_DEFENDER": ["A"] * 6,
            "CLOSEST_DEFENDER_PLAYER_ID": [1, 2, 3, 4, 5, 6],
            "CLOSE_DEF_DIST": [2.0, 5.0, 1.5, 4.0, 0.5, 6.0],
            "FGM": [1, 0, 1, 0, 1, 0],
            "PTS": [2, 0, 2, 0, 2, 0],
            "player_name": ["player a"] * 6,
            "player_id": [101] * 6,
        }
    )


def test_clean_clips_touch_time_and_flags_shot_clock() -> None:
    cleaned = clean_shot_logs(_sample_frame())
    assert (cleaned["TOUCH_TIME"] >= 0).all()
    assert cleaned["shot_clock_missing"].sum() == 2
    assert "game_clock_seconds" in cleaned.columns
    assert set(cleaned["is_home"].unique()).issubset({0, 1})
    assert set(cleaned["is_three"].unique()).issubset({0, 1})


def test_feature_matrix_shape() -> None:
    cleaned = clean_shot_logs(_sample_frame())
    X = get_feature_matrix(cleaned)
    y = get_target(cleaned)
    assert len(X) == len(y) == 6
    assert X.isna().sum().sum() == 2  # two missing SHOT_CLOCK values retained


def test_game_split_keeps_games_together() -> None:
    cleaned = clean_shot_logs(_sample_frame())
    train_df, test_df = game_level_train_test_split(cleaned, test_size=0.34)
    overlap = set(train_df["GAME_ID"]) & set(test_df["GAME_ID"])
    assert not overlap


def test_metrics_and_selection() -> None:
    metrics = evaluate_predictions([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["log_loss"] > 0
    best = select_best_model(
        {
            "a": {"log_loss": 0.5, "brier_score": 0.2},
            "b": {"log_loss": 0.4, "brier_score": 0.25},
        }
    )
    assert best == "b"
