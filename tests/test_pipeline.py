"""
Unit Tests for MLOps Pipeline
Tests training, drift detection, and API endpoints.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Training Tests ──────────────────────────────────────────────────────────

class TestTraining:
    def test_load_data(self):
        from src.training.train import load_data
        X, y, feature_names = load_data(None)
        assert X.shape[0] > 0
        assert len(feature_names) == X.shape[1]
        assert len(y) == len(X)

    def test_train_model(self):
        from src.training.train import load_data, train_model
        X, y, _ = load_data(None)
        model = train_model(X.values, y.values, n_estimators=10, max_depth=5)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_evaluate_model(self):
        from src.training.train import load_data, train_model, evaluate_model
        from sklearn.model_selection import train_test_split
        X, y, _ = load_data(None)
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)
        model = train_model(X_train, y_train, n_estimators=10, max_depth=5)
        metrics, _ = evaluate_model(model, X_test, y_test)
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_weighted"] <= 1

    def test_model_saves_artifacts(self):
        from src.training.train import load_data, train_model
        from sklearn.preprocessing import StandardScaler
        import joblib
        X, y, feature_names = load_data(None)
        model = train_model(X.values, y.values, n_estimators=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            joblib.dump(model, os.path.join(tmpdir, "model.pkl"))
            assert os.path.exists(os.path.join(tmpdir, "model.pkl"))


# ── Drift Detection Tests ────────────────────────────────────────────────────

class TestDriftDetection:
    @pytest.fixture
    def reference_df(self):
        from sklearn.datasets import load_wine
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df["target"] = wine.target
        return df

    def test_no_drift_same_data(self, reference_df):
        from src.monitoring.drift_monitor import fallback_drift_detection
        result = fallback_drift_detection(reference_df, reference_df.copy(), threshold=0.3)
        assert "drift_detected" in result
        # Same data should not drift significantly
        assert result["drift_share"] < 0.3

    def test_drift_detected_with_shifted_data(self, reference_df):
        from src.monitoring.drift_monitor import fallback_drift_detection
        drifted = reference_df.copy()
        feature_cols = [c for c in drifted.columns if c != "target"]
        for col in feature_cols:
            drifted[col] = drifted[col] * 5 + 100  # Massive shift
        result = fallback_drift_detection(drifted, reference_df.copy(), threshold=0.3)
        assert result["drift_detected"] == True

    def test_drift_result_structure(self, reference_df):
        from src.monitoring.drift_monitor import fallback_drift_detection
        result = fallback_drift_detection(reference_df, reference_df.copy())
        required_keys = ["drift_detected", "drift_share", "drifted_features", "method"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


# ── Drift Injection Tests ─────────────────────────────────────────────────────

class TestDriftInjection:
    @pytest.fixture
    def base_df(self):
        from sklearn.datasets import load_wine
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df["target"] = wine.target
        return df

    def test_covariate_shift_changes_features(self, base_df):
        from src.training.inject_drift import inject_covariate_shift
        drifted = inject_covariate_shift(base_df, drift_magnitude=0.5)
        assert not base_df.equals(drifted)
        assert len(drifted) == len(base_df)

    def test_label_shift_increases_samples(self, base_df):
        from src.training.inject_drift import inject_label_shift
        drifted = inject_label_shift(base_df, target_class=0, multiplier=2.0)
        assert len(drifted) > len(base_df)

    def test_production_stream_generation(self, tmp_path):
        from src.training.inject_drift import generate_production_stream
        results = generate_production_stream(
            n_batches=3,
            batch_size=20,
            drift_start_batch=2,
            output_dir=str(tmp_path),
        )
        assert len(results) == 3
        batch_files = list(tmp_path.glob("batch_*.csv"))
        assert len(batch_files) == 3


# ── API Tests ─────────────────────────────────────────────────────────────────

class TestAPIEndpoints:
    @pytest.fixture
    def client(self, tmp_path):
        """Create test client with mock model."""
        import joblib
        import json
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import load_wine

        wine = load_wine()
        X, y = wine.data, wine.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_scaled, y)

        model_dir = str(tmp_path)
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
        metadata = {
            "feature_names": list(wine.feature_names),
            "classes": [0, 1, 2],
            "trained_at": "2025-01-01T00:00:00",
            "metrics": {"f1_weighted": 0.99},
            "params": {},
        }
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        os.environ["MODEL_DIR"] = model_dir
        os.environ["PREDICTIONS_LOG"] = str(tmp_path / "predictions.jsonl")

        from fastapi.testclient import TestClient
        from src.api.app import app, load_model
        load_model()
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_valid_input(self, client):
        from sklearn.datasets import load_wine
        wine = load_wine()
        features = wine.data[0].tolist()
        response = client.post("/predict", json={"features": features})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in [0, 1, 2]
        assert len(data["probability"]) == 3
        assert data["latency_ms"] < 1000

    def test_predict_wrong_feature_count(self, client):
        response = client.post("/predict", json={"features": [1.0, 2.0]})
        assert response.status_code == 400

    def test_predict_batch(self, client):
        from sklearn.datasets import load_wine
        wine = load_wine()
        instances = wine.data[:5].tolist()
        response = client.post("/predict/batch", json={"instances": instances})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 5

    def test_predict_batch_rejects_invalid_feature_count(self, client):
        response = client.post("/predict/batch", json={"instances": [[1.0, 2.0]]})
        assert response.status_code == 400

    def test_info_endpoint(self, client):
        response = client.get("/info")
        assert response.status_code == 200
        assert "feature_names" in response.json()


class TestBenchmarkPlatform:
    def test_create_benchmark_suite(self, tmp_path):
        from src.benchmark.scenarios import create_benchmark_suite

        scenarios = create_benchmark_suite(output_dir=str(tmp_path))
        assert len(scenarios) >= 7
        assert (tmp_path / "mild_covariate" / "scenario_manifest.csv").exists()
        assert (tmp_path / "hybrid_regime_shift" / "scenario_manifest.csv").exists()
        assert (tmp_path / "adaptive_recovery_window" / "scenario_manifest.csv").exists()
        assert (tmp_path / "mild_covariate" / "scenario_spec.json").exists()

    def test_run_benchmark(self, tmp_path):
        from src.benchmark.runner import run_benchmark

        result = run_benchmark(
            output_dir=str(tmp_path / "benchmark"),
            config_path="configs/benchmark_config.yaml",
        )
        leaderboard = result["summary"]["overall_leaderboard"]
        assert leaderboard
        assert (tmp_path / "benchmark" / "benchmark_report.md").exists()
        assert (tmp_path / "benchmark" / "dashboard.html").exists()
        assert (tmp_path / "benchmark" / "benchmark_results.csv").exists()
        assert all("policy" in row for row in leaderboard)

    def test_run_benchmark_with_alternate_dataset(self, tmp_path):
        from src.benchmark.runner import run_benchmark

        result = run_benchmark(
            output_dir=str(tmp_path / "benchmark_alt"),
            config_path="configs/benchmark_config.yaml",
            dataset_name="breast_cancer",
        )
        assert result["summary"]["dataset_name"] == "breast_cancer"

    def test_run_benchmark_with_digits_dataset(self, tmp_path):
        from src.benchmark.runner import run_benchmark

        result = run_benchmark(
            output_dir=str(tmp_path / "benchmark_digits"),
            config_path="configs/benchmark_config.yaml",
            dataset_name="digits",
        )
        assert result["summary"]["dataset_name"] == "digits"
        assert any(row["policy"] == "challenger" for row in result["summary"]["overall_leaderboard"])

    def test_portfolio_config_runs(self, tmp_path):
        from src.benchmark.runner import run_benchmark

        result = run_benchmark(
            output_dir=str(tmp_path / "benchmark_portfolio"),
            config_path="configs/benchmark_portfolio.yaml",
        )
        assert result["summary"]["dataset_name"] == "breast_cancer"

    def test_tuned_digits_config_runs(self, tmp_path):
        from src.benchmark.runner import run_benchmark

        result = run_benchmark(
            output_dir=str(tmp_path / "benchmark_digits_tuned"),
            config_path="configs/benchmark_digits_tuned.yaml",
        )
        assert result["summary"]["dataset_name"] == "digits"
