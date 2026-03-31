#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# MLOps Pipeline — One-command local setup
# Usage: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────────
set -e

echo "========================================="
echo "  Drift Benchmark Setup"
echo "========================================="

# 1. Python venv
echo "[1/6] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 2. Install deps
echo "[2/6] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements/training.txt -r requirements/api.txt -q
echo "      Done."

# 3. Create data dirs
echo "[3/6] Creating directories..."
mkdir -p data/production data/reference model_artifacts reports/drift reports/benchmark logs

# 4. Generate reference + production data
echo "[4/6] Generating dataset and drift batches..."
python src/training/inject_drift.py \
  --n-batches 10 \
  --batch-size 50 \
  --drift-start 5 \
  --drift-type covariate \
  --drift-magnitude 0.4 \
  --output-dir data/production
echo "      10 batches generated (drift from batch 5 onwards)"

# 5. Train initial model (local, no MLflow server needed)
echo "[5/6] Training initial model..."
MLFLOW_TRACKING_URI=sqlite:///mlruns.db python src/training/train.py \
  --model-dir model_artifacts \
  --model-type random_forest \
  --n-estimators 160 \
  --max-depth 14
echo "      Model saved to model_artifacts/"

# 6. Run benchmark suite
echo "[6/6] Running drift benchmark suite..."
python src/benchmark/runner.py \
  --output-dir reports/benchmark \
  --config configs/benchmark_config.yaml
echo "      Benchmark reports saved to reports/benchmark/"

echo ""
echo "========================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Run benchmark:     python src/benchmark/runner.py --output-dir reports/benchmark"
echo "  2. Read report:       reports/benchmark/benchmark_report.md"
echo "  3. Open dashboard:    reports/benchmark/dashboard.html"
echo "  4. Start full stack:  cd docker && docker compose up"
echo "  5. API docs:          http://localhost:8000/docs"
echo "  6. Run tests:         pytest tests/ -v"
echo "========================================="
