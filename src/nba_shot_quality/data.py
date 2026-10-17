"""Load, validate, clean, and split the shot-log CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_shot_quality.config import (
    DEFAULT_DATA_PATH,
    NUMERIC_COLUMNS,
    RANDOM_SEED,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_shot_logs(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw shot-log CSV."""
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Shot-log CSV not found: {data_path}")
    return pd.read_csv(data_path)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise if required columns are missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _parse_game_clock_seconds(value: object) -> float:
    """Convert GAME_CLOCK strings like '1:09' to seconds remaining."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or ":" not in text:
        return np.nan
    try:
        minutes, seconds = text.split(":", 1)
        return float(int(minutes) * 60 + int(seconds))
    except (TypeError, ValueError):
        return np.nan


def clean_shot_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply V1 cleaning rules.

    - Coerce numeric columns
    - Clip negative TOUCH_TIME to 0
    - Flag missing SHOT_CLOCK (do not drop)
    - Parse GAME_CLOCK to seconds
    - Derive is_home
    - Drop rows missing the target
    """
    out = df.copy()
    validate_schema(out)

    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Target must be present and binary. Validate before astype(int) so
    # non-binary floats (e.g. 0.7) raise instead of silently truncating.
    out = out.dropna(subset=[TARGET_COLUMN])
    if not out[TARGET_COLUMN].isin([0, 1]).all():
        bad = sorted(out.loc[~out[TARGET_COLUMN].isin([0, 1]), TARGET_COLUMN].unique().tolist())
        raise ValueError(f"FGM must be binary 0/1; found non-binary values: {bad}")
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)

    # Known data quality issue: negative touch times (sensor noise)
    out["TOUCH_TIME"] = out["TOUCH_TIME"].clip(lower=0)

    # Missing shot clock: keep rows, flag for the model
    out["shot_clock_missing"] = out["SHOT_CLOCK"].isna().astype(int)

    out["game_clock_seconds"] = out["GAME_CLOCK"].map(_parse_game_clock_seconds)
    out["is_home"] = (out["LOCATION"].astype(str).str.upper() == "H").astype(int)

    return out.reset_index(drop=True)


def prepare_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load + validate + clean in one call."""
    return clean_shot_logs(load_shot_logs(path))


def game_level_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by GAME_ID so shots from the same game stay together.

    Returns (train_df, test_df).
    """
    if "GAME_ID" not in df.columns:
        raise ValueError("GAME_ID is required for the game-level split")

    game_ids = df["GAME_ID"].dropna().unique()
    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(game_ids)
    n_test = max(1, int(round(len(shuffled) * test_size)))
    test_ids = set(shuffled[:n_test])

    test_mask = df["GAME_ID"].isin(test_ids)
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Game-level split produced an empty train or test set")

    return train_df, test_df
