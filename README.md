# Drift-Response Benchmark Platform

**An MLOps benchmark that evaluates *when retraining is actually worth promoting* under drift, not
just whether drift can be detected.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/Serving-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Airflow](https://img.shields.io/badge/Orchestration-Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white)
![Docker](https://img.shields.io/badge/Stack-Docker%20Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)

Most MLOps projects stop after "drift detected → retrain". This repo pushes one level further: it
**benchmarks retraining policies** across drift regimes and datasets, **validates candidate models
before promotion**, and scores policies on quality, recovery, stability, runtime cost and latency.

---

## Central Question

> Which retraining policy is best under a given drift regime, and how should candidate models be
> validated before they replace the incumbent?

The platform treats that question as a measurement problem and produces reproducible evidence.

---

## Highlights

- **Five policy families** benchmarked head-to-head: `no_retrain`, `scheduled`, `threshold`,
  `adaptive_window`, `challenger`.
- **Candidate promotion gate** — retrained models are validated on a held-out slice and only
  promoted if they beat the incumbent (or recover from severe drift within tolerance).
- **Drift-regime generator** — covariate, label-shift, concept and hybrid drift with configurable
  severity, onset, duration and label delay.
- **Multi-dataset evaluation** — `wine`, `breast_cancer`, `digits` and custom CSV.
- **End-to-end MLOps stack preserved** — FastAPI serving, MLflow tracking, Airflow retraining DAG,
  Docker Compose, PSI / Evidently drift monitoring.
- **Reproducible outputs** — JSON, CSV, Markdown and a standalone HTML dashboard per run.

---

## Architecture

### Benchmark path (primary)

```
Dataset → train / reference / holdout split → scenario batches (with drift)
   → per-policy simulation → candidate training → promotion validation
   → composite scoring → leaderboard + dashboard
```

### Operational path (retained)

```
Train → FastAPI serve → log predictions → monitor drift
   → Airflow retraining DAG → validate candidate → reload model
```

---

## Repository Layout

```
mlops-pipeline/
├── src/
│   ├── benchmark/            # scenarios.py · policies.py · runner.py
│   ├── training/             # train.py (RF / ExtraTrees, MLflow logging)
│   ├── monitoring/           # drift_monitor.py (Evidently + PSI fallback)
│   └── api/app.py            # FastAPI inference server
├── airflow/dags/             # retraining_dag.py
├── docker/                   # Dockerfiles + docker-compose.yml
├── configs/
│   ├── benchmark_config.yaml           # hardest research config
│   ├── benchmark_portfolio.yaml        # best demo config
│   ├── benchmark_digits_tuned.yaml     # retraining-friendly digits sweep
│   └── pipeline_config.yaml            # legacy operational stack config
├── scripts/                  # setup.sh · run_benchmark.sh · run_portfolio_benchmarks.sh
├── reports/                  # benchmark_summary.json · dashboard.html · ...
├── tests/                    # pytest
└── .github/workflows/        # lint + pytest + benchmark smoke
```

---

## Policy Families

| Policy            | Behaviour                                                              |
|-------------------|------------------------------------------------------------------------|
| `no_retrain`      | Control baseline — never retrains                                      |
| `scheduled`       | Fixed-interval retraining (calendar-driven)                            |
| `threshold`       | Retrains when drift share / quality drop crosses a threshold           |
| `adaptive_window` | Rolling labelled window; reacts to sustained drift                     |
| `challenger`      | Trains challenger, promotes only on validated gain, respects cooldown  |

**Promotion gate.** Every retrain produces a candidate that is scored on a held-out validation
slice against the incumbent. The candidate is promoted only if it clears a configurable margin
(or the incumbent is severely degraded and the regression is within tolerance). Retraining is never
blindly trusted.

---

## Drift Scenarios

| Type            | Implementation                                       |
|-----------------|------------------------------------------------------|
| Covariate       | Feature scaling, mean shift, added noise             |
| Label shift     | Oversampled class priors                             |
| Concept         | Rank-based class remapping by severity               |
| Hybrid          | Covariate + concept, layered                         |

Each scenario controls batch size, number of batches, drift onset, severity, label delay and seed,
and is persisted alongside the benchmark for inspection.

---

## Scoring

The composite score combines **average batch F1**, **final post-drift holdout F1**, **recovery**,
**stability**, **training runtime cost**, **retrain fixed cost**, **drift penalties** and
**inference latency**. Optimising any single component (e.g. final F1) would bias the benchmark
toward aggressive retrainers — the multi-component score forces an honest tradeoff.

---

## Quick Start

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/training.txt -r requirements/api.txt -r requirements/test.txt
```

### Test

```bash
pytest tests/test_pipeline.py -q
```

### Run benchmarks

```bash
# hardest research config
python3 src/benchmark/runner.py --config configs/benchmark_config.yaml \
  --output-dir reports/benchmark

# best demo config (recommended for presentation)
python3 src/benchmark/runner.py --config configs/benchmark_portfolio.yaml \
  --output-dir reports/benchmark-portfolio

# retraining-friendly digits
python3 src/benchmark/runner.py --config configs/benchmark_digits_tuned.yaml \
  --output-dir reports/benchmark-digits-tuned

# both showcase configs
bash scripts/run_portfolio_benchmarks.sh
```

### Operational stack (Docker Compose)

```bash
cd docker && docker compose up mlflow api airflow
# MLflow :5000 · API :8000 · Airflow :8080 · Dashboard :3000
```

### FastAPI directly

```bash
uvicorn src.api.app:app --reload --port 8000
# GET  /health   GET /info   POST /predict   POST /predict/batch
# POST /reload   GET /metrics/predictions
```

---

## Output Artefacts

Every benchmark writes:

| File                         | Purpose                                                 |
|------------------------------|---------------------------------------------------------|
| `benchmark_summary.json`     | Overall leaderboard and scenario winners                |
| `benchmark_results.json`     | Detailed policy × batch results                         |
| `benchmark_results.csv`      | Spreadsheet-friendly tabular results                    |
| `benchmark_report.md`        | Human-readable narrative report                         |
| `dashboard.html`             | Standalone visual dashboard                             |
| `scenarios/...`              | All generated scenario CSVs and manifests               |

Key metrics: `avg_rank`, `avg_composite_score`, `avg_final_holdout_f1`,
`avg_final_post_drift_f1`, `avg_promoted_retrains`, `avg_train_seconds`, `avg_inference_ms`.

---

## Why `no_retrain` Sometimes Wins

If labels arrive late, retraining cost is non-trivial, and the promotion gate is strict, doing
nothing is often the rational choice. The benchmark is designed to **allow that outcome to appear**
— that is precisely what makes it honest and more useful than a benchmark rigged to force
retraining to win every scenario. The `challenger` policy becomes competitive on less adversarial
configs (e.g. `benchmark_portfolio.yaml` on `breast_cancer`), where different policies win
different scenarios.

---

## Known Limitations

- Drift is synthetic, not from live production.
- Candidate validation is simulated on a held-out slice, not via shadow deployment.
- No feature store or cloud-native model-registry promotion workflow.
- Dashboard is static HTML; no live analytics backend.

These are deliberate tradeoffs for local reproducibility and define the roadmap (real production
datasets, delayed-feedback simulation, dollar-denominated cost models, model registry integration,
shadow / rollback policies, richer dashboards).

---

## Author

**Saket Maganti** — academic, portfolio and research use.
