# Execution Guide

## Mac Setup

```bash
cd /Users/saketmaganti/projects/MLOPS/mlops-pipeline
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements/training.txt -r requirements/api.txt -r requirements/test.txt
```

Estimated time:
- environment creation: `under 1 min`
- dependency install: `3-10 min` on Mac, depending on first-time wheels

## Validation

```bash
pytest tests/test_pipeline.py -q
```

Estimated time:
- test suite: `1-2 min`

## Recommended Runs

### 1. Default Research Benchmark

```bash
python3 src/benchmark/runner.py \
  --output-dir reports/benchmark \
  --config configs/benchmark_config.yaml
```

Use this when:
- you want the hardest default benchmark
- you want to compare policies on `digits`

Estimated time:
- `45-120 sec`

### 2. Portfolio Benchmark

```bash
python3 src/benchmark/runner.py \
  --output-dir reports/benchmark-portfolio \
  --config configs/benchmark_portfolio.yaml
```

Use this when:
- you want presentation-ready results
- you want a benchmark where adaptive policies are more competitive
- you want the best storytelling dataset/config combination

Estimated time:
- `30-90 sec`

### 3. Tuned Digits Benchmark

```bash
python3 src/benchmark/runner.py \
  --output-dir reports/benchmark-digits-tuned \
  --config configs/benchmark_digits_tuned.yaml
```

Use this when:
- you specifically want retraining-friendly `digits` scenarios
- you want to stress challenger and adaptive policies

Estimated time:
- `45-120 sec`

### 4. Run Both Presentation Configs

```bash
bash scripts/run_portfolio_benchmarks.sh
```

Estimated time:
- `2-4 min`

## Where To Look

- default dashboard: `reports/benchmark/dashboard.html`
- portfolio dashboard: `reports/benchmark-portfolio/dashboard.html`
- tuned digits dashboard: `reports/benchmark-digits-tuned/dashboard.html`
- CSV outputs: `benchmark_results.csv` in each output folder
- Markdown reports: `benchmark_report.md` in each output folder

## Optional Local Model Training

```bash
MLFLOW_TRACKING_URI=sqlite:///mlruns.db python3 src/training/train.py \
  --model-dir model_artifacts \
  --model-type random_forest \
  --n-estimators 160 \
  --max-depth 14
```

Estimated time:
- `10-30 sec`

## Optional Docker Stack

```bash
cd docker
docker compose up mlflow api airflow
```

Estimated time:
- first run: `4-10 min`
- later runs: `1-3 min`
