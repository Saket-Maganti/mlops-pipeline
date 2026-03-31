#!/usr/bin/env bash
set -e

python3 src/benchmark/runner.py \
  --output-dir reports/benchmark-portfolio \
  --config configs/benchmark_portfolio.yaml

python3 src/benchmark/runner.py \
  --output-dir reports/benchmark-digits-tuned \
  --config configs/benchmark_digits_tuned.yaml

echo "Portfolio report: reports/benchmark-portfolio/benchmark_report.md"
echo "Portfolio dashboard: reports/benchmark-portfolio/dashboard.html"
echo "Tuned digits report: reports/benchmark-digits-tuned/benchmark_report.md"
echo "Tuned digits dashboard: reports/benchmark-digits-tuned/dashboard.html"
