"""Feature matrices and sklearn preprocessing for V1."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_shot_quality.config import (
    LEAKAGE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

# Columns that benefit from median imputation when missing
IMPUTE_COLUMNS = ["SHOT_CLOCK", "TOUCH_TIME", "game_clock_seconds"]

# Columns that should be complete after cleaning; pass through as-is
PASSTHROUGH_COLUMNS = [
    c for c in MODEL_FEATURE_COLUMNS if c not in IMPUTE_COLUMNS
]


def assert_no_feature_leakage() -> None:
    """Raise if model features include outcome / post-game leakage columns."""
    overlap = set(MODEL_FEATURE_COLUMNS) & set(LEAKAGE_COLUMNS)
    if overlap:
        raise ValueError(
            f"Model features overlap leakage columns: {sorted(overlap)}"
        )


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select model feature columns and check they exist."""
    assert_no_feature_leakage()
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered feature columns: {missing}")
    return df[MODEL_FEATURE_COLUMNS].copy()


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return the binary FGM target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")
    return df[TARGET_COLUMN].astype(int)


def build_preprocessor(*, scale_numeric: bool = False) -> ColumnTransformer:
    """
    Build preprocessing for model pipelines.

    - Median-impute SHOT_CLOCK (and a couple of other sparse numerics)
    - Optionally standardize for logistic regression
    - Leave remaining features as-is
    """
    impute_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        impute_steps.append(("scaler", StandardScaler()))

    transformers = [
        (
            "impute",
            Pipeline(impute_steps),
            IMPUTE_COLUMNS,
        ),
        (
            "pass",
            (
                Pipeline([("scaler", StandardScaler())])
                if scale_numeric
                else "passthrough"
            ),
            PASSTHROUGH_COLUMNS,
        ),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop")
