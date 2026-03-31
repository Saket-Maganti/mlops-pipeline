"""
MLOps Pipeline - Model Training with MLflow Tracking
Trains tree-based classifiers on a classification dataset with reusable helpers.
"""

import os
import json
import logging
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_path: str = None):
    """Load wine dataset or from provided CSV path."""
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        X = df.drop("target", axis=1)
        y = df["target"]
        feature_names = list(X.columns)
    else:
        logger.info("Loading Wine dataset from sklearn")
        wine = load_wine()
        X = pd.DataFrame(wine.data, columns=wine.feature_names)
        y = pd.Series(wine.target)
        feature_names = list(wine.feature_names)

    logger.info(f"Dataset shape: {X.shape}, Classes: {y.nunique()}")
    return X, y, feature_names


def train_model(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=10,
    random_state=42,
    model_type: str = "random_forest",
):
    """Train a supported tree-based classifier."""
    common_kwargs = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if model_type == "random_forest":
        model = RandomForestClassifier(**common_kwargs)
    elif model_type == "extra_trees":
        model = ExtraTreesClassifier(**common_kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_samples": len(y_test),
    }
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    logger.info(f"Metrics: {metrics}")
    return metrics, report


def fit_training_bundle(
    df: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    model_type: str = "random_forest",
) -> Dict[str, object]:
    """Fit a model bundle that includes model, scaler, and metadata."""
    X = df.drop("target", axis=1)
    y = df["target"]
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = train_model(
        X_scaled,
        y,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        model_type=model_type,
    )
    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "classes": [int(cls) for cls in model.classes_],
        "params": {
            "model_type": model_type,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        },
    }


def score_model_bundle(bundle: Dict[str, object], df: pd.DataFrame) -> Dict[str, float]:
    """Score a trained bundle against a labeled dataframe."""
    X = df[bundle["feature_names"]]
    y = df["target"]
    X_scaled = bundle["scaler"].transform(X)
    metrics, _ = evaluate_model(bundle["model"], X_scaled, y)
    return metrics


def main(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(getattr(args, "experiment_name", "wine-classifier-mlops"))

    with mlflow.start_run(run_name=f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # Load data
        X, y, feature_names = load_data(args.data_path)
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Log params
        params = {
            "model_type": getattr(args, "model_type", "random_forest"),
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "test_size": 0.2,
            "random_state": 42,
            "dataset": "wine",
            "n_features": len(feature_names),
            "train_samples": len(X_train),
        }
        mlflow.log_params(params)

        # Train
        logger.info("Training model...")
        model = train_model(
            X_train_scaled,
            y_train,
            args.n_estimators,
            args.max_depth,
            42,
            getattr(args, "model_type", "random_forest"),
        )

        # Evaluate
        metrics, report = evaluate_model(model, X_test_scaled, y_test)
        mlflow.log_metrics(metrics)

        # Log feature importances
        importances = dict(
            zip(feature_names, model.feature_importances_.tolist())
        )
        mlflow.log_dict(importances, "feature_importances.json")
        mlflow.log_dict(report, "classification_report.json")

        # Save artifacts
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, "model.pkl")
        scaler_path = os.path.join(args.model_dir, "scaler.pkl")
        meta_path = os.path.join(args.model_dir, "metadata.json")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        metadata = {
            "feature_names": feature_names,
            "classes": [int(c) for c in model.classes_],
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics,
            "params": params,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifacts(args.model_dir)
        mlflow.sklearn.log_model(model, "sklearn-model")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Training complete. Run ID: {run_id}")
        logger.info(f"F1 Score: {metrics['f1_weighted']:.4f}")

        return run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="/tmp/model_artifacts")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--model-type", type=str, default="random_forest")
    parser.add_argument("--experiment-name", type=str, default="wine-classifier-mlops")
    args = parser.parse_args()
    main(args)
