"""Model pipelines for V1 baselines."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from nba_shot_quality.config import RANDOM_SEED
from nba_shot_quality.features import build_preprocessor


def build_logistic_regression() -> Pipeline:
    """Scaled logistic regression baseline."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting() -> Pipeline:
    """HistGradientBoosting — handles missing values; no scaling needed."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=False)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.1,
                    max_iter=200,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_candidate_models() -> dict[str, Pipeline]:
    """Return named candidate pipelines for training/comparison."""
    return {
        "logistic_regression": build_logistic_regression(),
        "hist_gradient_boosting": build_hist_gradient_boosting(),
    }
