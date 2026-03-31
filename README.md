# Drift Response Benchmark Platform

This project is now a Level 4 MLOps portfolio system: not just a single retraining pipeline, but a benchmark platform for comparing drift-response strategies across multiple scenarios and datasets.

Instead of asking "can we detect drift and retrain?", the project asks:

`Which retraining policy performs best under different types of drift, how quickly does it recover quality, and what runtime/cost tradeoff does it create?`

## What It Does

- generates multiple benchmark scenarios with covariate, label, and concept drift
- evaluates several retraining policies side by side
- measures batch performance, final recovery quality, retrain count, training runtime, and inference latency
- ranks policies per scenario and produces a benchmark report
- still keeps the original FastAPI, MLflow, Airflow, and drift-monitoring pieces for end-to-end MLOps context

## Policy Set

- `no_retrain`: baseline control
- `scheduled`: retrain on a fixed cadence
- `threshold`: retrain when drift or quality degradation crosses a threshold
- `adaptive_window`: retrain using rolling labeled data when drift is sustained or quality drops
- `challenger`: promote retraining only when expected recovery gain appears worth the cost

## Built-in Datasets

- `digits` is the default research benchmark because it creates a harder multi-class setting
- `wine`
- `breast_cancer`
- custom CSV with a `target` column

## Recommended Configs

- `configs/benchmark_config.yaml`: hardest default research benchmark
- `configs/benchmark_portfolio.yaml`: best presentation/demo benchmark
- `configs/benchmark_digits_tuned.yaml`: retraining-friendly `digits` stress benchmark

## Scenario Set

- `mild_covariate`
- `severe_covariate`
- `label_shift`
- `concept_drift`
- `hybrid_regime_shift`
- `adaptive_recovery_window`
- `late_breaking_regime_shift`

Each scenario includes:
- initial train/reference/holdout splits
- sequential production-like batches
- delayed label availability
- per-batch metadata for reproducibility
- a post-drift holdout set so policies are judged on recovery, not just on the original data split

## Project Structure

```text
mlops-pipeline/
├── src/
│   ├── api/
│   ├── benchmark/
│   │   ├── scenarios.py
│   │   ├── policies.py
│   │   └── runner.py
│   ├── monitoring/
│   ├── training/
│   └── utils/
├── configs/
│   ├── benchmark_config.yaml
│   └── pipeline_config.yaml
├── reports/
├── scripts/
│   ├── setup.sh
│   └── run_benchmark.sh
└── tests/
```

## Quick Start

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/training.txt -r requirements/api.txt -r requirements/test.txt
```

### 2. Run the Benchmark Platform

```bash
python src/benchmark/runner.py \
  --output-dir reports/benchmark \
  --config configs/benchmark_config.yaml
```

Outputs:

- `reports/benchmark/benchmark_summary.json`
- `reports/benchmark/benchmark_results.json`
- `reports/benchmark/benchmark_results.csv`
- `reports/benchmark/benchmark_report.md`
- `reports/benchmark/dashboard.html`
- `reports/benchmark/scenarios/...`

### 3. Switch Dataset

```bash
python src/benchmark/runner.py \
  --output-dir reports/benchmark-breast-cancer \
  --config configs/benchmark_config.yaml \
  --dataset-name breast_cancer
```

```bash
python src/benchmark/runner.py \
  --output-dir reports/benchmark-digits \
  --config configs/benchmark_config.yaml \
  --dataset-name digits
```

### 4. Run The Portfolio Config

```bash
python src/benchmark/runner.py \
  --output-dir reports/benchmark-portfolio \
  --config configs/benchmark_portfolio.yaml
```

### 5. Run The Tuned Digits Config

```bash
python src/benchmark/runner.py \
  --output-dir reports/benchmark-digits-tuned \
  --config configs/benchmark_digits_tuned.yaml
```

### 6. One-Command Setup

```bash
bash scripts/setup.sh
```

### 7. Run Tests

```bash
pytest tests/ -v
```

## Optional Monitoring Upgrade

`evidently` is now included in the training requirements. On Mac, the fallback PSI drift detector will still work if Evidently is unavailable, but when installation succeeds the monitoring path becomes richer automatically.

## Docker

Run the original local stack:

```bash
cd docker
docker compose up mlflow api airflow
```

Run the benchmark profile:

```bash
cd docker
docker compose --profile benchmark up benchmark
```

## Typical Runtime

On a laptop for the built-in wine benchmark:

- benchmark suite: `10-60 seconds`
- single retrain event: `under 1 second` to a few seconds
- full end-to-end local setup: `2-5 minutes`

With larger real datasets, the benchmark framework is intended to scale to:

- medium benchmark study: `5-30 minutes`
- multi-policy, multi-scenario benchmark: `20-90 minutes`

## Why This Is Level 4

This project is distinct because it is no longer a one-model demo. It is a reusable experiment framework for:

- evaluating retraining policies
- quantifying drift severity versus quality loss
- measuring recovery speed and runtime tradeoffs
- producing evidence-backed recommendations for production retraining strategy

That makes it closer to an MLOps research and platform engineering project than a standard "train-serve-monitor" tutorial.

## Hardening Notes

- benchmark scenarios and policies are config-driven
- the runner can switch between built-in datasets without code changes
- results are exported as JSON, CSV, Markdown, and a standalone HTML dashboard
- scenario specs are persisted for reproducibility
- default scenarios now include harder regime-shift cases where retraining policies can meaningfully outperform the baseline
- the default benchmark now uses a tougher multi-class dataset and includes a challenger policy intended to trade off recovery against retrain cost
- retraining is now guarded by candidate promotion instead of blindly replacing the incumbent
- separate portfolio and tuned benchmark configs are included for demos and storytelling
