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
    """Compute log loss, Brier score, ROC-AUC, and accuracy.

    ROC-AUC is NaN when y_true has only one class (undefined otherwise).
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    # Clip for numerical stability in log_loss
    y_prob_clipped = np.clip(y_prob_arr, 1e-7, 1 - 1e-7)
    y_pred = (y_prob_clipped >= threshold).astype(int)

    # labels=[0, 1] keeps log_loss well-defined if a tiny split is one-class
    metrics: dict[str, float] = {
        "log_loss": float(log_loss(y_true_arr, y_prob_clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_clipped)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
    }
    if len(np.unique(y_true_arr)) < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_clipped))
    return metrics


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
        if primary_metric not in metrics:
            raise KeyError(
                f"Metric '{primary_metric}' missing for model '{name}'"
            )
        value = metrics[primary_metric]
        return value if lower_is_better else -value

    return min(metrics_by_model.items(), key=sort_key)[0]
