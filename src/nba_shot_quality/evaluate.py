"""Probability and classification metrics for V1."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute log loss, Brier score, ROC-AUC, and accuracy."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    # Clip for numerical stability in log_loss
    y_prob_clipped = np.clip(y_prob_arr, 1e-7, 1 - 1e-7)
    y_pred = (y_prob_clipped >= threshold).astype(int)

    return {
        "log_loss": float(log_loss(y_true_arr, y_prob_clipped)),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_clipped)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_clipped)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
    }


def select_best_model(
    metrics_by_model: dict[str, dict[str, float]],
    *,
    primary_metric: str = "log_loss",
    lower_is_better: bool = True,
) -> str:
    """Pick the best model name by primary metric (default: lowest log loss)."""
    if not metrics_by_model:
        raise ValueError("No model metrics provided")

    def sort_key(item: tuple[str, dict[str, Any]]) -> float:
        name, metrics = item
        value = metrics[primary_metric]
        return value if lower_is_better else -value

    return min(metrics_by_model.items(), key=sort_key)[0]
