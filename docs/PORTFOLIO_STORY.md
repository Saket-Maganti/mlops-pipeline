# Portfolio Story

## One-Line Positioning

This project is an MLOps benchmark platform that evaluates when automated retraining is actually worth doing under different types of drift.

## Strongest Demo Angle

The strongest framing is not "I built FastAPI + MLflow + Airflow."

It is:

"I built a policy benchmark that compares no retraining, scheduled retraining, threshold retraining, adaptive retraining, and challenger promotion logic across multiple drift regimes and datasets, then measures recovery, runtime, stability, and promotion cost."

## Best Config To Show

Use the portfolio config:

```bash
python3 src/benchmark/runner.py \
  --output-dir reports/benchmark-portfolio \
  --config configs/benchmark_portfolio.yaml
```

Why:
- it is easier to explain than the hardest `digits` benchmark
- it produces a more interesting policy race
- it better demonstrates why selective retraining can beat brute-force retraining

## What To Point Out In A Demo

1. The benchmark does not automatically trust retraining.
   Candidate models are trained and then promoted only if they beat the incumbent on a validation slice.

2. The project compares policy families, not just models.
   The important artifact is the decision system around retraining, not only the classifier itself.

3. Results differ by dataset and drift regime.
   That is the point: there is no universally best retraining policy.

4. The outputs are reviewable.
   JSON, CSV, Markdown, and an HTML dashboard are all generated from the same run.

## Honest Interpretation

- On harder settings like `digits`, doing nothing can still win overall when retraining is expensive or labels arrive late.
- On portfolio-friendly settings like `breast_cancer`, selective retraining policies can become competitive or win.
- That means the benchmark is surfacing a real systems tradeoff instead of producing a predetermined winner.

## Resume Version

"Built a drift-response benchmark platform for MLOps systems, comparing five retraining policies across seven scenario families and multiple datasets, with selective model promotion, runtime/cost scoring, and dashboard/report generation."
