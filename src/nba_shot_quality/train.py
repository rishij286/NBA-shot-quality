"""Train V1 shot-quality models and save the selected pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from nba_shot_quality.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    MODEL_FEATURE_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from nba_shot_quality.data import game_level_train_test_split, prepare_dataset
from nba_shot_quality.evaluate import evaluate_predictions, select_best_model
from nba_shot_quality.features import get_feature_matrix, get_target
from nba_shot_quality.modeling import build_candidate_models


def train(
    data_path: Path | str | None = None,
    model_path: Path | str | None = None,
    metrics_path: Path | str | None = None,
) -> dict:
    """Load data, train candidates, evaluate, save best pipeline + metrics."""
    data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    metrics_path = Path(metrics_path) if metrics_path else DEFAULT_METRICS_PATH

    print(f"Loading data from {data_path}")
    df = prepare_dataset(data_path)
    print(f"Rows after cleaning: {len(df):,}")

    train_df, test_df = game_level_train_test_split(df)
    print(
        f"Train: {len(train_df):,} shots / {train_df['GAME_ID'].nunique()} games | "
        f"Test: {len(test_df):,} shots / {test_df['GAME_ID'].nunique()} games"
    )

    X_train = get_feature_matrix(train_df)
    y_train = get_target(train_df)
    X_test = get_feature_matrix(test_df)
    y_test = get_target(test_df)

    candidates = build_candidate_models()
    metrics_by_model: dict[str, dict[str, float]] = {}
    fitted: dict[str, object] = {}

    for name, pipeline in candidates.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, y_prob)
        metrics_by_model[name] = metrics
        fitted[name] = pipeline
        print(
            f"  log_loss={metrics['log_loss']:.4f}  "
            f"brier={metrics['brier_score']:.4f}  "
            f"roc_auc={metrics['roc_auc']:.4f}  "
            f"accuracy={metrics['accuracy']:.4f}"
        )

    best_name = select_best_model(metrics_by_model)
    best_pipeline = fitted[best_name]
    print(f"\nSelected model: {best_name} (lowest log loss)")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "random_seed": RANDOM_SEED,
        "metrics": metrics_by_model[best_name],
        "all_metrics": metrics_by_model,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    joblib.dump(artifact, model_path)
    print(f"Saved model artifact to {model_path}")

    metrics_payload = {
        "selected_model": best_name,
        "metrics_by_model": metrics_by_model,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "data_path": str(data_path),
        "trained_at": artifact["trained_at"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"Saved metrics to {metrics_path}")

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NBA shot-quality V1 models"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to shot_logs.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to write the joblib artifact",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to write metrics JSON",
    )
    args = parser.parse_args()
    train(
        data_path=args.data_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
