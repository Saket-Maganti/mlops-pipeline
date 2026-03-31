#!/usr/bin/env bash
set -e

python src/benchmark/runner.py \
  --output-dir reports/benchmark \
  --config configs/benchmark_config.yaml

echo "Benchmark markdown report: reports/benchmark/benchmark_report.md"
echo "Benchmark dashboard: reports/benchmark/dashboard.html"
echo "Benchmark CSV: reports/benchmark/benchmark_results.csv"
