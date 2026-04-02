# Drift Response Benchmark Platform

> A Level 4 MLOps project that goes beyond "train, serve, monitor" and asks a harder systems question:
>
> **When is retraining actually worth doing under drift, which policy should trigger it, and how should a candidate model be validated before promotion?**

---

## 1. Table of Contents

1. [What This Project Is](#2-what-this-project-is)
2. [Why This Project Exists](#3-why-this-project-exists)
3. [What Makes It Distinct](#4-what-makes-it-distinct)
4. [Current Scope and Project Range](#5-current-scope-and-project-range)
5. [Project Status](#6-project-status)
6. [Repository Layout](#7-repository-layout)
7. [Core System Architecture](#8-core-system-architecture)
8. [High-Level Data Flow](#9-high-level-data-flow)
9. [Main Workflows Supported by the Repo](#10-main-workflows-supported-by-the-repo)
10. [Primary Benchmarking Objective](#11-primary-benchmarking-objective)
11. [Secondary End-to-End MLOps Objective](#12-secondary-end-to-end-mlops-objective)
12. [Supported Datasets](#13-supported-datasets)
13. [Why Multiple Datasets Matter](#14-why-multiple-datasets-matter)
14. [Scenario Families](#15-scenario-families)
15. [What Each Drift Type Means Here](#16-what-each-drift-type-means-here)
16. [Policy Families](#17-policy-families)
17. [Promotion Logic](#18-promotion-logic)
18. [Scoring Philosophy](#19-scoring-philosophy)
19. [Project Components in Detail](#20-project-components-in-detail)
20. [Detailed Module Walkthrough](#21-detailed-module-walkthrough)
21. [Benchmark Runner in Detail](#22-benchmark-runner-in-detail)
22. [Scenario Generator in Detail](#23-scenario-generator-in-detail)
23. [Policy Engine in Detail](#24-policy-engine-in-detail)
24. [Training Module in Detail](#25-training-module-in-detail)
25. [Monitoring Module in Detail](#26-monitoring-module-in-detail)
26. [Inference API in Detail](#27-inference-api-in-detail)
27. [Airflow DAG in Detail](#28-airflow-dag-in-detail)
28. [Docker Stack in Detail](#29-docker-stack-in-detail)
29. [CI Workflow in Detail](#30-ci-workflow-in-detail)
30. [Configuration Files](#31-configuration-files)
31. [Default Research Config](#32-default-research-config-benchmark_configyaml)
32. [Portfolio Config](#33-portfolio-config-benchmark_portfolioyaml)
33. [Tuned Digits Config](#34-tuned-digits-config-benchmark_digits_tunedyaml)
34. [Generated Outputs](#35-generated-outputs)
35. [How to Read the Benchmark Results](#36-how-to-read-the-benchmark-results)
36. [Interpreting the Current Results](#37-interpreting-the-current-results)
37. [Why No-Retrain Can Still Win](#38-why-no-retrain-can-still-win)
38. [Why That Is Actually Valuable](#39-why-that-is-actually-valuable)
39. [Mac Setup Guide](#40-mac-setup-guide)
40. [Quick Start Commands](#41-quick-start-commands)
41. [Detailed Execution Paths](#42-detailed-execution-paths)
42. [Expected Run Times](#43-expected-run-times)
43. [How to Choose Which Config to Run](#44-how-to-choose-which-config-to-run)
44. [Recommended Demo Path](#45-recommended-demo-path)
45. [Recommended Interview Path](#46-recommended-interview-path)
46. [Files to Open After a Run](#47-files-to-open-after-a-run)
47. [Legacy Stack Notes](#48-legacy-stack-notes)
48. [Known Limitations](#49-known-limitations)
49. [Design Tradeoffs](#50-design-tradeoffs)
50. [Troubleshooting](#51-troubleshooting)
51. [FAQ](#52-faq)
52. [Resume and Portfolio Positioning](#53-resume-and-portfolio-positioning)
53. [Suggested Talking Track](#54-suggested-talking-track)
54. [Future Extensions](#55-future-extensions)
55. [Final Summary](#56-final-summary)

---

## 2. What This Project Is

This repository is a **drift-response benchmark platform** for MLOps systems.

It is not only:

- a training script
- a FastAPI model server
- a drift monitor
- an Airflow retraining DAG
- a Dockerized local stack

It is also:

- a policy comparison system
- a scenario generator
- a benchmark harness
- a candidate-promotion framework
- a runtime and cost analysis layer
- a reporting and dashboard generation pipeline

The central idea is simple:

Many MLOps projects stop after proving that drift can be detected and retraining can be triggered.

This project pushes one level further:

**It evaluates whether retraining should actually be promoted, under what drift conditions, on which datasets, and with what runtime/cost tradeoff.**

That makes the repository much closer to a **research + platform engineering project** than to a standard tutorial pipeline.

---

## 3. Why This Project Exists

In production ML systems, retraining is not always a free win.

Retraining has costs:

- compute cost
- operational complexity
- risk of bad promotion
- risk of overreacting to transient shifts
- dependency on label availability
- potential degradation on the original distribution

At the same time, doing nothing also has costs:

- stale models
- performance collapse under concept drift
- hidden quality loss
- delayed response to distribution shifts

So the real question is not:

"Can I trigger retraining?"

The real question is:

"**Which retraining policy is best under a given drift regime, and how should candidate models be validated before promotion?**"

That is the problem this repository is built to explore.

---

## 4. What Makes It Distinct

What makes this project more advanced than a typical MLOps portfolio project:

- It compares **multiple retraining policies**, not just a single retraining trigger.
- It evaluates those policies on **multiple datasets** and **multiple drift regimes**.
- It uses **post-drift holdout sets** so results are not biased toward the original distribution only.
- It includes **candidate promotion logic**, so retraining does not automatically replace the incumbent.
- It exports **structured reports** in JSON, CSV, Markdown, and HTML dashboard format.
- It retains an end-to-end MLOps stack with FastAPI, MLflow, Docker, and Airflow for context.

This means the repo can be framed as:

- an MLOps benchmark project
- an ML systems research project
- a model governance / promotion project
- a portfolio-quality platform engineering project

---

## 5. Current Scope and Project Range

This project sits roughly in the following range:

- Above a basic "single notebook ML project"
- Above a standard "train + FastAPI + Docker" demo
- Above a simple "drift detection and retraining" pipeline
- Below a full enterprise platform with Kubernetes, feature store, and cloud-native governance

A fair range description is:

**Advanced solo MLOps / ML systems project with research-grade benchmarking elements**

Or more explicitly:

**Level 4 portfolio project**

Why Level 4:

- it has architectural breadth
- it has policy comparison depth
- it has reproducible benchmark outputs
- it includes promotion logic and multiple datasets
- it produces insight, not just infrastructure

What it is not yet:

- a production multi-tenant ML platform
- a managed feature-store deployment
- a full online/offline serving parity platform
- a large-scale cloud deployment benchmark

So the best classification is:

**Advanced individual MLOps benchmark and policy evaluation platform**

---

## 6. Project Status

At the time of writing, the repository supports:

- benchmark generation
- multiple configs
- multiple policy families
- multiple datasets
- training and scoring
- API serving
- drift detection
- Airflow orchestration demo
- Dockerized local execution
- CI test and benchmark runs
- generated reports and dashboards

The repository has been run locally on macOS and validated through:

- direct benchmark execution
- alternate benchmark configs
- script-based setup
- script-based benchmark execution
- test suite execution

Recent validation status:

- `22 passed` in the local test suite

---

## 7. Repository Layout

Top-level layout:

```text
mlops-pipeline/
├── .github/workflows/
├── airflow/dags/
├── configs/
├── data/
├── docker/
├── docs/
├── model_artifacts/
├── reports/
├── requirements/
├── scripts/
├── src/
└── tests/
```

What each main directory does:

- `.github/workflows/`: CI pipeline for test + benchmark runs
- `airflow/dags/`: Airflow DAG showing a retraining workflow
- `configs/`: benchmark and pipeline configuration files
- `data/`: generated production batches and reference data
- `docker/`: Dockerfiles and `docker-compose.yml`
- `docs/`: execution guide and portfolio story docs
- `model_artifacts/`: saved model bundle for local serving
- `reports/`: benchmark outputs and drift reports
- `requirements/`: Python dependencies
- `scripts/`: convenience scripts for setup and benchmark execution
- `src/`: project source code
- `tests/`: automated tests

---

## 8. Core System Architecture

At a conceptual level, the project has **two overlapping architectures**:

### 8.1 Benchmark Architecture

This is the primary architecture today.

Flow:

1. Load a dataset
2. Split it into train / reference / holdout
3. Generate drift scenarios
4. Simulate sequential batches
5. Evaluate multiple policies
6. Train candidate retrain models when policies fire
7. Validate candidates before promotion
8. Score policy outcomes
9. Export reports and dashboards

### 8.2 Legacy End-to-End MLOps Stack

This remains in the repo and is still useful.

Flow:

1. Train baseline model
2. Serve model via FastAPI
3. Log predictions
4. Monitor drift
5. Trigger retraining in Airflow
6. Reload model via API

The benchmark layer is the higher-value part of the project now, but the legacy stack is still important because it shows the project is grounded in operational MLOps components, not only in synthetic experimentation.

---

## 9. High-Level Data Flow

The full project can be understood as five layers:

### Layer 1: Dataset Layer

Supported sources:

- built-in sklearn datasets
- custom CSV with a `target` column

### Layer 2: Scenario Layer

The project generates:

- reference distributions
- production-like batches
- delayed labels
- drift severity progression
- post-drift holdout evaluation sets

### Layer 3: Policy Layer

The project compares:

- do nothing
- fixed schedule
- threshold-based retrain
- adaptive window retrain
- challenger promotion strategy

### Layer 4: Promotion Layer

Retrained candidates are:

- trained
- validated against a held-out validation slice
- compared to the incumbent
- promoted only if they pass promotion criteria

### Layer 5: Reporting Layer

The project writes:

- JSON summaries
- JSON detailed results
- CSV tabular results
- Markdown reports
- standalone HTML dashboards

---

## 10. Main Workflows Supported by the Repo

The repo supports several different workflows.

### Workflow A: Run the hardest benchmark

Use:

- `configs/benchmark_config.yaml`

Purpose:

- research-style stress benchmark
- harder multi-class benchmark
- baseline may still win
- useful for honest system analysis

### Workflow B: Run the best portfolio benchmark

Use:

- `configs/benchmark_portfolio.yaml`

Purpose:

- presentation-quality results
- more varied scenario winners
- more compelling storytelling
- stronger demo artifact

### Workflow C: Run a tuned `digits` benchmark

Use:

- `configs/benchmark_digits_tuned.yaml`

Purpose:

- keep `digits`
- increase retraining competitiveness
- explore more retraining-friendly pressure conditions

### Workflow D: Run the original local MLOps stack

Use:

- Docker Compose
- FastAPI
- MLflow
- Airflow

Purpose:

- show operational components
- demonstrate serving and retraining flow

---

## 11. Primary Benchmarking Objective

The primary objective of the repository today is:

**to compare retraining policies under controlled drift conditions and measure whether retraining improves outcomes enough to justify the cost and promotion risk**

This objective is operationalized through:

- scenario generation
- policy simulation
- validation-based promotion
- weighted scoring
- multi-format reporting

---

## 12. Secondary End-to-End MLOps Objective

The secondary objective is:

**to retain a recognizable end-to-end MLOps stack so the benchmark platform is grounded in real MLOps building blocks**

This is why the repo still contains:

- FastAPI serving
- MLflow experiment logging
- Airflow DAG orchestration
- Docker Compose stack
- drift monitor
- model artifacts

---

## 13. Supported Datasets

The benchmark currently supports three built-in datasets.

### 13.1 `wine`

Characteristics:

- small
- multiclass
- easy to train quickly
- good for pipeline demos

Why it matters:

- ideal for smoke tests
- useful for light local benchmarking
- easy to understand

### 13.2 `breast_cancer`

Characteristics:

- binary classification
- more realistic separation under some drift conditions
- good for policy storytelling

Why it matters:

- tends to produce more interesting winner distributions
- works well for the portfolio config
- can make challenger-style promotion logic easier to demonstrate

### 13.3 `digits`

Characteristics:

- multiclass
- discrete feature space
- harder benchmark
- more demanding concept drift behavior

Why it matters:

- makes the benchmark less trivial
- creates more realistic failure conditions
- helps show that retraining is not always beneficial

### 13.4 Custom CSV

Requirement:

- must include a `target` column

Why it matters:

- makes the benchmark extensible
- allows custom experiments without rewriting the system

---

## 14. Why Multiple Datasets Matter

A single dataset can make a benchmark misleading.

If one dataset is too easy:

- all policies look good
- no-retrain may look unbeatable
- retraining costs dominate

If one dataset is too unstable:

- retraining may always look necessary
- the benchmark becomes biased toward aggressive policies

Using multiple datasets lets the project show that:

- policy performance is context-dependent
- drift-response strategy is not universal
- cost-aware promotion logic matters

This is one of the strongest points of the repository from a portfolio perspective.

---

## 15. Scenario Families

The project includes several drift scenario families.

These are not arbitrary labels.

Each scenario controls:

- drift type
- batch size
- number of batches
- batch index where drift starts
- drift severity
- label delay
- random seed

The scenarios currently used depend on the config file.

### Common scenario families in the repo

- `mild_covariate`
- `severe_covariate`
- `label_shift`
- `concept_drift`
- `hybrid_regime_shift`
- `adaptive_recovery_window`
- `late_breaking_regime_shift`
- `portfolio_*` variants
- `digits_*` variants

### Why scenario families matter

They let the project simulate:

- mild vs strong drift
- early vs late drift onset
- short vs long drift windows
- rapid feedback vs delayed feedback
- feature shift vs label remapping vs mixed regime change

---

## 16. What Each Drift Type Means Here

### 16.1 Covariate Drift

Implemented as:

- feature scaling
- mean shift
- added noise

Interpretation:

- the feature distribution moves
- the label mapping is intended to remain relatively stable

Typical question:

- can the incumbent stay robust enough without retraining?

### 16.2 Label Shift

Implemented as:

- oversampling one target class
- preserving overall batch size

Interpretation:

- class priors change
- the population becomes imbalanced

Typical question:

- do policy triggers react sensibly when class composition changes but mapping may still be similar?

### 16.3 Concept Drift

Implemented as:

- rank-based class remapping using selected features
- partial label reassignment driven by severity

Interpretation:

- the relationship between inputs and labels changes
- this is the drift type where retraining should matter most

Typical question:

- can a retrained candidate recover quality enough to justify promotion?

### 16.4 Hybrid Drift

Implemented as:

- covariate drift first
- concept drift layered on top

Interpretation:

- both distribution and label behavior change

Typical question:

- does the system overfit to the new regime, recover effectively, or fail to promote useful candidates?

---

## 17. Policy Families

The benchmark compares five policy families.

### 17.1 `no_retrain`

Behavior:

- never retrains
- serves as the control baseline

Why it matters:

- reveals whether retraining is truly necessary
- exposes cases where retraining cost outweighs benefit

### 17.2 `scheduled`

Behavior:

- retrains on a fixed interval

Why it matters:

- simulates calendar-driven retraining
- common in real systems where retraining is periodic

### 17.3 `threshold`

Behavior:

- retrains when drift or degradation exceeds a threshold

Why it matters:

- mirrors alert-driven retraining
- easy to explain and common in monitoring pipelines

### 17.4 `adaptive_window`

Behavior:

- retrains using rolling labeled data
- reacts to sustained drift or quality drop

Why it matters:

- simulates a more context-aware policy
- useful where recent behavior matters more than historical data

### 17.5 `challenger`

Behavior:

- retrains only when expected recovery gain seems worth the cost
- respects cooldown periods
- relies on candidate promotion validation

Why it matters:

- this is the most governance-like policy
- closest to "train challenger, validate, promote if justified"

---

## 18. Promotion Logic

One of the most important upgrades in this project is that retraining is not blindly trusted.

The current logic is:

1. A policy decides whether retraining should happen.
2. A candidate model is trained.
3. A validation slice is held out from the recent labeled buffer.
4. The incumbent and candidate are both scored on that validation slice.
5. The candidate is promoted only if:
   - it improves enough over the incumbent, or
   - a severe degradation condition is met and the regression is within tolerance

This is much more realistic than:

- retrain and automatically deploy

Because in real ML systems, retraining can absolutely make things worse.

This promotion gate is a major reason the project is stronger than a basic retraining demo.

---

## 19. Scoring Philosophy

The benchmark does not rank policies on accuracy alone.

It combines:

- average batch F1
- final post-drift holdout F1
- recovery score
- stability
- training runtime cost
- retrain fixed cost
- drift penalties
- inference latency penalty

This is intentional.

If you optimize only for final quality:

- aggressive retrainers may look unfairly good

If you optimize only for cost:

- `no_retrain` may always dominate

The benchmark instead tries to capture a more realistic tradeoff space.

This is still a designed scoring system, not an objective law.

That is why multiple configs are included:

- research config
- portfolio config
- tuned digits config

Each config reflects a slightly different emphasis.

---

## 20. Project Components in Detail

The core source modules are:

- `src/benchmark/scenarios.py`
- `src/benchmark/policies.py`
- `src/benchmark/runner.py`
- `src/training/train.py`
- `src/monitoring/drift_monitor.py`
- `src/api/app.py`

The supporting operational modules are:

- `airflow/dags/retraining_dag.py`
- `docker/docker-compose.yml`
- `.github/workflows/ci_cd.yml`

The support scripts are:

- `scripts/setup.sh`
- `scripts/run_benchmark.sh`
- `scripts/run_portfolio_benchmarks.sh`

The docs are:

- `docs/EXECUTION_GUIDE.md`
- `docs/PORTFOLIO_STORY.md`

---

## 21. Detailed Module Walkthrough

This section maps the code to the behavior you see at runtime.

### `src/benchmark/scenarios.py`

Responsibilities:

- load built-in or custom datasets
- split train / reference / holdout
- generate drifted sequential batches
- generate post-drift holdout evaluation sets
- persist scenario artifacts

### `src/benchmark/policies.py`

Responsibilities:

- define the policy decision interface
- define policy state
- implement policy families
- build policies from config

### `src/benchmark/runner.py`

Responsibilities:

- orchestrate the benchmark
- simulate batch progression
- collect labeled buffers
- run candidate training
- validate promotion
- score results
- write JSON, CSV, Markdown, HTML outputs

### `src/training/train.py`

Responsibilities:

- load data
- train supported tree-based models
- evaluate models
- build reusable training bundles
- log training runs to MLflow

### `src/monitoring/drift_monitor.py`

Responsibilities:

- run drift detection with Evidently when available
- fall back to PSI-based detection when Evidently is not available
- save drift reports and summaries

### `src/api/app.py`

Responsibilities:

- load saved model artifacts
- serve inference endpoints
- log predictions
- expose health and info endpoints
- support batch inference
- support model reload

---

## 22. Benchmark Runner in Detail

`src/benchmark/runner.py` is the heart of the repository.

What it does step by step:

1. Read a benchmark config
2. Build scenario specs
3. Build policy instances
4. Generate scenario artifacts
5. Train the initial incumbent model
6. Evaluate incumbent on baseline holdout and post-drift holdout
7. Iterate batch by batch
8. Maintain a labeled buffer respecting label delay
9. Detect drift
10. Score current policy performance
11. Ask the policy whether retraining should happen
12. If retraining is triggered:
    - split a validation slice
    - train candidate
    - compare candidate vs incumbent
    - promote only if rules are satisfied
13. Record batch-level results
14. Score final outcomes
15. Rank policies
16. Write summary and artifacts

Outputs written by the runner:

- `benchmark_summary.json`
- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_report.md`
- `dashboard.html`

---

## 23. Scenario Generator in Detail

The scenario generator is more than a random data perturbation utility.

It creates a reproducible experiment structure.

Each scenario contains:

- `train.csv`
- `reference.csv`
- `holdout.csv`
- `post_drift_holdout.csv`
- `batch_00.csv`, `batch_01.csv`, ...
- `scenario_manifest.csv`
- `scenario_spec.json`

Why this matters:

- you can inspect each generated scenario
- you can trace benchmark outcomes back to actual generated data
- you can compare scenario definitions across configs
- the benchmark is not a black box

---

## 24. Policy Engine in Detail

The policy layer uses a shared state object.

State fields include:

- current batch index
- drift detected or not
- drift share
- batch metrics
- degradation flag
- last retrain batch
- labeled buffer size
- consecutive drift batches
- recent F1 scores
- baseline post-drift F1
- expected recovery gain

This means policies are not blind.

They can reason using:

- drift evidence
- quality evidence
- time since last retrain
- available labeled data
- expected benefit

That structure is why the project can support multiple policy styles without duplicating the whole benchmark.

---

## 25. Training Module in Detail

`src/training/train.py` supports:

- loading sklearn wine dataset by default
- loading custom CSV data
- training:
  - `random_forest`
  - `extra_trees`
- scaling inputs with `StandardScaler`
- computing:
  - accuracy
  - weighted F1
  - macro F1
- saving:
  - `model.pkl`
  - `scaler.pkl`
  - `metadata.json`
- logging to MLflow

This module has two roles:

### Role A: standalone model training

Used by:

- manual training commands
- setup script
- local model artifact generation

### Role B: benchmark candidate training

Used by:

- benchmark runner
- candidate generation
- promotion validation flow

Because the same training bundle can be reused in multiple contexts, the benchmark stays grounded in a real model training component rather than a toy scoring abstraction.

---

## 26. Monitoring Module in Detail

`src/monitoring/drift_monitor.py` supports two monitoring paths.

### Path A: Evidently

If the `evidently` package is installed:

- full Evidently report objects can be created
- data drift and data quality reports can be saved
- HTML reports can be generated

### Path B: PSI Fallback

If Evidently is not installed:

- the code falls back to a Population Stability Index approach
- feature histograms are compared
- drift share is computed
- retraining trigger decisions can still be made

This is useful on macOS because:

- you can still run the project without blocking on monitoring extras
- the benchmark and test paths remain operational

---

## 27. Inference API in Detail

The FastAPI app is implemented in `src/api/app.py`.

Main endpoints:

### `GET /health`

Purpose:

- service health check
- confirms whether model is loaded

Behavior:

- returns 503 if model is not loaded
- returns healthy payload when model is available

### `GET /info`

Purpose:

- expose model metadata

Fields include:

- feature names
- classes
- training timestamp
- training metrics

### `POST /predict`

Purpose:

- single-record inference

Input:

- ordered feature list
- optional request ID

Output:

- predicted class
- probabilities
- latency
- model version

### `POST /predict/batch`

Purpose:

- multi-record inference

Input:

- list of feature arrays

Output:

- predictions
- probabilities
- count
- latency

### `POST /reload`

Purpose:

- reload model from disk after retraining

### `GET /metrics/predictions`

Purpose:

- inspect basic prediction log statistics

This API is not the main product of the repository anymore, but it remains useful because it demonstrates that the project still has operational serving components.

---

## 28. Airflow DAG in Detail

The Airflow DAG lives in:

- `airflow/dags/retraining_dag.py`

Its structure:

1. ingest latest production batch
2. detect drift
3. branch:
   - no retraining path
   - retraining path
4. validate new model
5. call API reload endpoint
6. log pipeline metrics

This DAG is important because it shows:

- the repo understands orchestration concepts
- retraining can be represented as a scheduled operational workflow
- the benchmark platform did not replace the operational MLOps context

Important note:

The benchmark platform is now the stronger and more distinctive part of the project.

The Airflow DAG is still useful, but it should be framed as:

- supporting operational context
- not the primary differentiator

---

## 29. Docker Stack in Detail

`docker/docker-compose.yml` defines these main services:

### `mlflow`

Purpose:

- experiment tracking server

Exposes:

- port `5000`

### `api`

Purpose:

- FastAPI inference server

Exposes:

- port `8000`

### `trainer`

Purpose:

- on-demand training container

### `benchmark`

Purpose:

- run benchmark inside a container profile

### `airflow`

Purpose:

- local Airflow demo

Exposes:

- port `8080`

### `dashboard`

Purpose:

- static dashboard server via nginx

Exposes:

- port `3000`

This Compose file is useful for:

- showing service separation
- local demos
- operational context

However, for many users the fastest path is still the pure Python benchmark commands rather than Docker.

---

## 30. CI Workflow in Detail

CI is defined in:

- `.github/workflows/ci_cd.yml`

It runs two jobs:

### `test`

Steps:

- checkout code
- set up Python 3.11
- install dependencies
- run Ruff
- run pytest with coverage
- upload coverage

### `benchmark`

Steps:

- install benchmark dependencies
- run the benchmark
- upload benchmark artifacts

This is valuable because:

- the repo is not just locally runnable
- benchmark generation is part of CI thinking
- the project treats benchmark outputs as artifacts, not only logs

---

## 31. Configuration Files

Main config files:

- `configs/pipeline_config.yaml`
- `configs/benchmark_config.yaml`
- `configs/benchmark_portfolio.yaml`
- `configs/benchmark_digits_tuned.yaml`

These separate concerns:

- operational MLOps stack config
- hard research benchmark
- portfolio-friendly benchmark
- tuned digits benchmark

That separation is important.

It means the repo can support:

- honest research runs
- showcase runs
- stress runs

without pretending that one config fits all use cases.

---

## 32. Default Research Config: `benchmark_config.yaml`

This is the hardest default benchmark.

Highlights:

- default dataset: `digits`
- initial model: `random_forest`
- retrain model: `extra_trees`
- candidate promotion enabled
- seven scenario families
- five policy families

Use this when you want:

- a hard stress benchmark
- a skeptical environment where retraining is not guaranteed to win
- a more research-like evaluation setting

Interpretation:

- if `no_retrain` wins here, that is not a bug
- it means your retraining policies are not yet strong enough for that pressure setting

---

## 33. Portfolio Config: `benchmark_portfolio.yaml`

This is the best config to show in a demo or interview.

Highlights:

- default dataset: `breast_cancer`
- more presentation-friendly policy separation
- more varied scenario winners
- `challenger` performs competitively

Use this when you want:

- a strong demo
- a clear story
- a good dashboard outcome
- more intuitive interpretation

In practice, this config is currently the most compelling benchmark to present.

---

## 34. Tuned Digits Config: `benchmark_digits_tuned.yaml`

This config keeps `digits` but makes retraining more competitive.

Highlights:

- larger batch sizes
- stronger concept/hybrid shifts
- lower promotion barrier
- lower retraining cost penalties

Use this when you want:

- to stress retraining policies on a harder dataset
- to study whether digits can produce more retraining-friendly outcomes

This config is useful for experimentation, even if it is not always the cleanest one for presentation.

---

## 35. Generated Outputs

Every benchmark run produces several output formats.

### `benchmark_summary.json`

Contains:

- overall leaderboard
- scenario winners
- dataset name
- policy count
- scenario count

### `benchmark_results.json`

Contains:

- detailed policy-level and batch-level results

### `benchmark_results.csv`

Contains:

- tabular results suitable for spreadsheet inspection

### `benchmark_report.md`

Contains:

- human-readable benchmark summary

### `dashboard.html`

Contains:

- standalone visual dashboard

### `scenarios/...`

Contains:

- all generated scenario assets

This output design is one of the strongest practical features of the project.

It means the repo is useful for:

- engineering review
- data inspection
- portfolio demos
- CSV export and analysis
- downstream visualization

---

## 36. How to Read the Benchmark Results

The most important fields are:

### `avg_rank`

Lower is better.

This captures how often a policy places well across scenarios.

### `avg_composite_score`

Higher is better.

This is the weighted benchmark score.

### `avg_final_holdout_f1`

How the final model performs on the original holdout distribution.

### `avg_final_post_drift_f1`

How the final model performs on the post-drift holdout distribution.

This is one of the most important fields in the whole project.

### `avg_promoted_retrains`

How often candidate models are actually promoted.

This helps you distinguish:

- policies that trigger often
- policies that trigger but fail promotion
- policies that promote selectively

### `avg_train_seconds`

Average retraining time.

Useful for cost and runtime interpretation.

### `avg_inference_ms`

Average inference latency observed in the benchmark loop.

---

## 37. Interpreting the Current Results

From recent local runs:

### Default `digits` benchmark

Pattern:

- `no_retrain` still tends to rank first
- `adaptive_window`, `challenger`, and `scheduled` cluster behind it

Interpretation:

- the benchmark is harsh
- retraining is expensive
- delayed labels and promotion checks make aggressive retraining difficult

### Portfolio `breast_cancer` benchmark

Pattern:

- `challenger` becomes highly competitive and can finish first overall by average rank
- different policies win different scenarios

Interpretation:

- this config is better for presentation
- the benchmark tells a clearer story about selective retraining

### Tuned `digits` benchmark

Pattern:

- `no_retrain` may still lead
- but the stress profile is different and more retraining-friendly than the raw default

Interpretation:

- good for experimentation
- less ideal than the portfolio config for presentation

---

## 38. Why No-Retrain Can Still Win

This deserves a direct explanation.

If `no_retrain` wins, that does **not** automatically mean the benchmark is wrong.

Possible reasons:

- labels arrive too late
- retraining uses limited recent data
- candidate promotion is strict
- retraining cost penalties are meaningful
- the incumbent remains robust enough
- drift is not severe enough to justify intervention

This is exactly why the benchmark is useful:

it can reveal when retraining is a bad idea.

That is a more mature result than a benchmark designed to force retraining to win every time.

---

## 39. Why That Is Actually Valuable

Many ML demos implicitly assume:

- drift appears
- retraining happens
- performance improves

But in real systems:

- retraining can overfit
- retraining can react too late
- retraining can cost more than it is worth
- retraining can degrade on the original distribution

This repository is stronger because it allows those outcomes to appear.

That makes it:

- more honest
- more useful
- more realistic
- more interesting in interviews

---

## 40. Mac Setup Guide

This repository has been run locally on macOS.

Recommended setup:

```bash
cd /Users/saketmaganti/projects/MLOPS/mlops-pipeline
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements/training.txt -r requirements/api.txt -r requirements/test.txt
```

Why `python3` matters on Mac:

- `python` may not point to the expected interpreter
- Homebrew and system Python setups differ

If you already have a venv one directory above:

```bash
source ../venv/bin/activate
```

Either works as long as you install the repo requirements into the active environment.

---

## 41. Quick Start Commands

### Run tests

```bash
pytest tests/test_pipeline.py -q
```

### Run the default benchmark

```bash
python3 src/benchmark/runner.py --output-dir reports/benchmark --config configs/benchmark_config.yaml
```

### Run the portfolio benchmark

```bash
python3 src/benchmark/runner.py --output-dir reports/benchmark-portfolio --config configs/benchmark_portfolio.yaml
```

### Run the tuned digits benchmark

```bash
python3 src/benchmark/runner.py --output-dir reports/benchmark-digits-tuned --config configs/benchmark_digits_tuned.yaml
```

### Run both showcase configs

```bash
bash scripts/run_portfolio_benchmarks.sh
```

---

## 42. Detailed Execution Paths

### Path A: Minimal verification

Use when:

- you just want to confirm the repo works

Commands:

```bash
pytest tests/test_pipeline.py -q
python3 src/benchmark/runner.py --output-dir reports/benchmark --config configs/benchmark_config.yaml
```

### Path B: Best demo path

Use when:

- you want the strongest presentation result

Commands:

```bash
python3 src/benchmark/runner.py --output-dir reports/benchmark-portfolio --config configs/benchmark_portfolio.yaml
```

Then open:

- `reports/benchmark-portfolio/benchmark_report.md`
- `reports/benchmark-portfolio/dashboard.html`

### Path C: Full setup path

Use when:

- you want baseline data, local model artifacts, and a benchmark run

Command:

```bash
bash scripts/setup.sh
```

### Path D: Local model training only

Use when:

- you only need model artifacts

Command:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlruns.db python3 src/training/train.py --model-dir model_artifacts --model-type random_forest --n-estimators 160 --max-depth 14
```

### Path E: Docker path

Use when:

- you want the operational stack

Commands:

```bash
cd docker
docker compose up mlflow api airflow
```

---

## 43. Expected Run Times

These are approximate, local Mac-oriented estimates.

### Environment creation

- under 1 minute

### Dependency install

- 3 to 10 minutes on first setup

### Test suite

- around 1 to 3 minutes

### Default benchmark

- around 45 to 120 seconds

### Portfolio benchmark

- around 30 to 90 seconds

### Tuned digits benchmark

- around 45 to 120 seconds

### Both showcase configs

- around 2 to 4 minutes

### Setup script end-to-end

- around 4 to 10 minutes depending on environment state

### Docker full stack

- 4 to 10 minutes on first image build
- 1 to 3 minutes on later runs

---

## 44. How to Choose Which Config to Run

If you want:

- the hardest research benchmark: use `benchmark_config.yaml`
- the best demo: use `benchmark_portfolio.yaml`
- a retraining-friendly digits experiment: use `benchmark_digits_tuned.yaml`

If you only run one config for presentation, run:

- `benchmark_portfolio.yaml`

---

## 45. Recommended Demo Path

For a portfolio demo:

1. run the portfolio config
2. open the dashboard
3. show that different policies win different scenarios
4. explain that retraining is candidate-validated, not blindly trusted

Best command:

```bash
python3 src/benchmark/runner.py --output-dir reports/benchmark-portfolio --config configs/benchmark_portfolio.yaml
```

---

## 46. Recommended Interview Path

In an interview, emphasize these points:

1. The project benchmarks **policies**, not only models.
2. Retraining is **guarded by promotion logic**.
3. Results are **dataset-dependent** and **drift-dependent**.
4. The project includes both:
   - benchmark infrastructure
   - operational MLOps components

If you need one sentence:

> I built an MLOps benchmark platform that evaluates when retraining is actually worth promoting under different types of drift, rather than assuming retraining is always beneficial.

---

## 47. Files to Open After a Run

Most useful files:

### After default benchmark

- `reports/benchmark/benchmark_report.md`
- `reports/benchmark/dashboard.html`
- `reports/benchmark/benchmark_results.csv`

### After portfolio benchmark

- `reports/benchmark-portfolio/benchmark_report.md`
- `reports/benchmark-portfolio/dashboard.html`
- `reports/benchmark-portfolio/benchmark_results.csv`

### After tuned digits benchmark

- `reports/benchmark-digits-tuned/benchmark_report.md`
- `reports/benchmark-digits-tuned/dashboard.html`
- `reports/benchmark-digits-tuned/benchmark_results.csv`

### For documentation and explanation

- `docs/EXECUTION_GUIDE.md`
- `docs/PORTFOLIO_STORY.md`

---

## 48. Legacy Stack Notes

The repo still includes:

- API serving
- drift monitoring
- Airflow orchestration
- MLflow logging
- Docker Compose deployment

These pieces matter because they show that the project did not become a pure offline benchmark disconnected from MLOps practice.

However, the benchmark platform is now the stronger differentiator.

If you are presenting the project, the best narrative is:

- operational stack for realism
- benchmark layer for depth and originality

---

## 49. Known Limitations

This repository is strong, but not infinite in scope.

Current limitations include:

- the benchmark uses synthetic drift, not live production datasets
- the serving stack is local/demo-oriented
- candidate validation is local and simulated
- there is no full cloud-native deployment governance layer
- no feature store integration
- no model registry promotion workflow beyond the local benchmark/promotion logic
- default dashboard is generated static HTML, not a live analytics app

These are fair limitations, but they do not reduce the value of the project as an advanced portfolio system.

---

## 50. Design Tradeoffs

Several deliberate tradeoffs were made.

### Tradeoff 1: speed vs realism

Small/medium datasets were used so:

- local execution stays fast
- the project remains runnable on consumer hardware

### Tradeoff 2: synthetic drift vs real drift

Synthetic drift gives:

- controlled experiments
- repeatability
- parameterized scenario families

But it is not identical to:

- real production drift

### Tradeoff 3: static dashboard vs web app

Static HTML dashboard gives:

- portability
- zero server requirement
- easy artifact sharing

But not:

- live filtering
- persistent backend

### Tradeoff 4: benchmark-first vs cloud-first

This project prioritizes:

- explainability
- policy comparison
- reproducibility

over:

- large-scale cloud deployment complexity

---

## 51. Troubleshooting

### Problem: `requirements/training.txt` not found

Cause:

- you are in the wrong directory

Fix:

```bash
cd /Users/saketmaganti/projects/MLOPS/mlops-pipeline
```

### Problem: `Evidently not installed`

Meaning:

- fallback PSI path is being used

Fix:

```bash
python3 -m pip install -r requirements/training.txt
```

### Problem: benchmark is slow

Causes:

- first dependency load
- larger tuned config
- file output overhead

Fix:

- use the portfolio config first

### Problem: Docker takes long

Cause:

- first image build

Fix:

- prefer direct Python benchmark commands if you only need the benchmark

### Problem: pytest cache warnings

Meaning:

- local environment may prevent `.pytest_cache` writes

Impact:

- not a benchmark failure
- not a code correctness issue

---

## 52. FAQ

### Is this a production deployment project?

Partly.

It contains production-like components, but its strongest form today is as a benchmark and policy evaluation platform.

### Is this more of an MLOps project or a research project?

It is both.

Best description:

- MLOps benchmark and ML systems project

### Which config should I show on a resume or in an interview?

- `configs/benchmark_portfolio.yaml`

### Which config should I use for honest stress testing?

- `configs/benchmark_config.yaml`

### Which dataset gives the strongest demo story?

- `breast_cancer` via the portfolio config

### Does the project still include an API?

Yes.

### Does the project still include Airflow?

Yes.

### Is retraining automatic?

A candidate can be trained automatically, but promotion is validated rather than blindly automatic.

---

## 53. Resume and Portfolio Positioning

Best framing:

**Built a drift-response benchmark platform for MLOps systems, comparing multiple retraining and promotion policies across controlled drift regimes and datasets, with candidate validation, runtime/cost scoring, and dashboard/report generation.**

This framing is stronger than:

- "Built a FastAPI ML app"
- "Built an Airflow retraining pipeline"

Because it emphasizes:

- policy reasoning
- evaluation depth
- systems tradeoffs
- reproducibility

---

## 54. Suggested Talking Track

If you are explaining the project verbally:

> I started with a normal MLOps pipeline idea, but I realized most projects stop at showing that drift can trigger retraining. So I pushed it further into a benchmark platform that compares multiple retraining policies, validates candidates before promotion, and scores them across datasets and drift regimes. The project still includes a FastAPI service, MLflow logging, Docker, and Airflow, but the main differentiator is that it measures whether retraining is actually worth doing, not just whether it can be triggered.

That is a strong and honest description.

---

## 55. Future Extensions

Possible future upgrades:

- add real-world tabular datasets beyond sklearn built-ins
- add online evaluation / delayed-feedback simulation
- add explicit cost models in dollars
- add model registry integration
- add shadow deployment simulation
- add rollback policy simulation
- add confidence calibration metrics
- add richer live dashboarding
- add more algorithms beyond tree ensembles
- add richer scenario families driven by domain-specific assumptions

These are real extensions, not mandatory prerequisites.

The repository already has substantial depth as it stands.

---

## 56. Final Summary

This repository is best understood as:

**a Level 4 MLOps benchmark platform with operational serving and orchestration context**

What it does:

- generates drift scenarios
- compares retraining policies
- validates candidate promotion
- measures quality, recovery, stability, runtime, and cost
- exports reports and dashboards
- retains API, monitoring, MLflow, Docker, and Airflow context

Why it matters:

- it asks a deeper systems question than a standard MLOps demo
- it produces insight, not just infrastructure
- it is runnable on consumer hardware
- it supports both research-style and portfolio-style configurations

If you only remember one thing about this project, remember this:

**The project does not assume retraining is always correct. It benchmarks when retraining should actually be promoted.**

