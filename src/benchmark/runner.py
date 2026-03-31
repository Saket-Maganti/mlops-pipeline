"""
Benchmark runner that compares drift-response policies across scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml

from src.benchmark.policies import PolicyState, build_policies
from src.benchmark.scenarios import create_benchmark_suite, parse_scenario_specs
from src.monitoring.drift_monitor import fallback_drift_detection
from src.training.train import fit_training_bundle, score_model_bundle
from src.utils.helpers import save_json, timestamp


def _labeled_rows_up_to(batches: List[pd.DataFrame], current_batch_index: int) -> pd.DataFrame:
    frames = []
    for batch_df in batches[: current_batch_index + 1]:
        mask = batch_df["label_available_batch"] <= current_batch_index
        labeled = batch_df.loc[mask].drop(columns=["label_available_batch", "scenario_name"], errors="ignore")
        if not labeled.empty:
            frames.append(labeled)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _rolling_window(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.iloc[-max_rows:].reset_index(drop=True)


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _split_training_and_validation(
    labeled_buffer: pd.DataFrame,
    validation_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if labeled_buffer.empty or len(labeled_buffer) <= validation_rows:
        return labeled_buffer, pd.DataFrame()
    validation_frame = labeled_buffer.iloc[-validation_rows:].reset_index(drop=True)
    training_frame = labeled_buffer.iloc[:-validation_rows].reset_index(drop=True)
    return training_frame, validation_frame


def _resolve_model_params(benchmark_cfg: Dict[str, object], prefix: str) -> Dict[str, object]:
    return {
        "model_type": benchmark_cfg[f"{prefix}_model_type"],
        "n_estimators": int(benchmark_cfg[f"{prefix}_n_estimators"]),
        "max_depth": int(benchmark_cfg[f"{prefix}_max_depth"]),
    }


def _estimate_expected_recovery_gain(
    baseline_post_drift_f1: float,
    current_batch_f1: float,
    drift_share: float,
    labeled_buffer_size: int,
) -> float:
    quality_gap = max(0.0, baseline_post_drift_f1 - current_batch_f1)
    data_readiness = min(1.0, labeled_buffer_size / 128.0)
    return quality_gap * 0.65 + drift_share * 0.20 + data_readiness * 0.05


def _train_candidate_bundle(
    training_frame: pd.DataFrame,
    reference_df: pd.DataFrame,
    benchmark_cfg: Dict[str, object],
    scenario_seed: int,
    batch_index: int,
    use_reference_mix: bool,
) -> tuple[Dict[str, object], int]:
    reference_mix_ratio = float(benchmark_cfg["reference_mix_ratio"])
    training_candidate = training_frame.copy()
    if use_reference_mix and reference_mix_ratio > 0 and not training_candidate.empty:
        reference_rows = max(1, int(len(training_candidate) * reference_mix_ratio))
        reference_mix = reference_df.sample(
            n=min(reference_rows, len(reference_df)),
            replace=len(reference_df) < reference_rows,
            random_state=scenario_seed + batch_index,
        )
        training_candidate = pd.concat([reference_mix, training_candidate], ignore_index=True)

    bundle = fit_training_bundle(
        training_candidate,
        random_state=scenario_seed + batch_index,
        **_resolve_model_params(benchmark_cfg, "retrain"),
    )
    return bundle, len(training_candidate)


def _simulate_policy(
    scenario: Dict[str, object],
    policy,
    benchmark_cfg: Dict[str, object],
) -> Dict[str, object]:
    train_df = scenario["train_df"]
    reference_df = scenario["reference_df"]
    holdout_df = scenario["holdout_df"]
    post_drift_holdout_df = scenario["post_drift_holdout_df"]
    batches = scenario["batches"]
    degradation_threshold = float(benchmark_cfg["degradation_threshold"])
    retrain_window_rows = int(benchmark_cfg["retrain_window_rows"])
    train_seconds_cost = float(benchmark_cfg["train_seconds_cost"])
    retrain_fixed_cost = float(benchmark_cfg["retrain_fixed_cost"])
    drift_penalty_weight = float(benchmark_cfg["drift_penalty_weight"])
    recovery_weight = float(benchmark_cfg["recovery_weight"])
    batch_weight = float(benchmark_cfg["batch_weight"])
    final_weight = float(benchmark_cfg["final_weight"])
    inference_penalty_weight = float(benchmark_cfg["inference_penalty_weight"])
    validation_rows = int(benchmark_cfg["promotion_validation_rows"])
    promotion_min_gain = float(benchmark_cfg["promotion_min_gain"])
    max_regression_tolerance = float(benchmark_cfg["promotion_max_regression"])

    model_bundle = fit_training_bundle(
        train_df,
        random_state=scenario["seed"],
        **_resolve_model_params(benchmark_cfg, "initial"),
    )
    baseline_metrics = score_model_bundle(model_bundle, holdout_df)
    baseline_f1 = baseline_metrics["f1_weighted"]
    baseline_post_drift_metrics = score_model_bundle(model_bundle, post_drift_holdout_df)
    baseline_post_drift_f1 = baseline_post_drift_metrics["f1_weighted"]

    batch_results = []
    retrain_events = []
    train_times = []
    inference_times = []
    drift_streak = 0
    last_retrain_batch = -1
    recent_f1_scores: List[float] = []
    downtime_risk = 0

    for batch_index, batch_df in enumerate(batches):
        labeled_buffer = _labeled_rows_up_to(batches, batch_index)

        inference_start = time.perf_counter()
        batch_metrics = score_model_bundle(model_bundle, batch_df)
        inference_time = (time.perf_counter() - inference_start) * 1000
        inference_times.append(inference_time)
        recent_f1_scores.append(batch_metrics["f1_weighted"])

        drift_result = fallback_drift_detection(
            reference_df,
            batch_df.drop(columns=["label_available_batch", "scenario_name"], errors="ignore"),
            threshold=0.30,
        )
        drift_detected = bool(drift_result["drift_detected"])
        drift_streak = drift_streak + 1 if drift_detected else 0
        degraded = batch_metrics["f1_weighted"] < baseline_f1 - degradation_threshold

        state = PolicyState(
            current_batch_index=batch_index,
            drift_detected=drift_detected,
            drift_share=float(drift_result["drift_share"]),
            batch_metrics=batch_metrics,
            degraded=degraded,
            last_retrain_batch=last_retrain_batch,
            labeled_buffer_size=len(labeled_buffer),
            consecutive_drift_batches=drift_streak,
            recent_f1_scores=recent_f1_scores.copy(),
            baseline_post_drift_f1=baseline_post_drift_f1,
            expected_recovery_gain=_estimate_expected_recovery_gain(
                baseline_post_drift_f1=baseline_post_drift_f1,
                current_batch_f1=batch_metrics["f1_weighted"],
                drift_share=float(drift_result["drift_share"]),
                labeled_buffer_size=len(labeled_buffer),
            ),
        )
        decision = policy.decide(state)

        retrained = False
        retrain_seconds = 0.0
        train_rows = 0
        promoted = False
        promotion_gain = 0.0
        if decision.retrain and not labeled_buffer.empty:
            training_frame = labeled_buffer.copy()
            if decision.train_on_window:
                training_frame = _rolling_window(training_frame, retrain_window_rows)
            elif not decision.train_on_all_labeled:
                training_frame = training_frame.copy()
            training_frame, validation_frame = _split_training_and_validation(training_frame, validation_rows)
            if training_frame.empty:
                training_frame = labeled_buffer.copy()
                validation_frame = pd.DataFrame()

            train_start = time.perf_counter()
            candidate_bundle, train_rows = _train_candidate_bundle(
                training_frame=training_frame,
                reference_df=reference_df,
                benchmark_cfg=benchmark_cfg,
                scenario_seed=scenario["seed"],
                batch_index=batch_index,
                use_reference_mix=decision.use_reference_mix,
            )
            retrain_seconds = time.perf_counter() - train_start
            train_times.append(retrain_seconds)
            retrained = True

            if validation_frame.empty:
                promoted = True
                promotion_gain = promotion_min_gain
            else:
                incumbent_validation = score_model_bundle(model_bundle, validation_frame)
                candidate_validation = score_model_bundle(candidate_bundle, validation_frame)
                promotion_gain = candidate_validation["f1_weighted"] - incumbent_validation["f1_weighted"]
                severe_need = degraded and state.expected_recovery_gain >= promotion_min_gain * 1.5
                promoted = (
                    promotion_gain >= promotion_min_gain
                    or (severe_need and promotion_gain >= -max_regression_tolerance)
                )

            if promoted:
                model_bundle = candidate_bundle
                last_retrain_batch = batch_index
            retrain_events.append(
                {
                    "batch_index": batch_index,
                    "reason": decision.reason,
                    "train_rows": train_rows,
                    "retrain_seconds": round(retrain_seconds, 4),
                    "promoted": promoted,
                    "promotion_gain": round(promotion_gain, 4),
                }
            )
        elif decision.retrain and labeled_buffer.empty:
            downtime_risk += 1

        batch_results.append(
            {
                "batch_index": batch_index,
                "f1_weighted": round(batch_metrics["f1_weighted"], 4),
                "accuracy": round(batch_metrics["accuracy"], 4),
                "drift_detected": drift_detected,
                "drift_share": round(float(drift_result["drift_share"]), 4),
                "degraded": degraded,
                "retrain": retrained,
                "retrain_reason": decision.reason,
                "labeled_rows_available": len(labeled_buffer),
                "train_rows_used": train_rows,
                "promoted": promoted,
                "promotion_gain": round(promotion_gain, 4),
                "expected_recovery_gain": round(state.expected_recovery_gain, 4),
                "inference_ms": round(inference_time, 3),
            }
        )

    final_metrics = score_model_bundle(model_bundle, holdout_df)
    final_post_drift_metrics = score_model_bundle(model_bundle, post_drift_holdout_df)
    avg_batch_f1 = _safe_mean([row["f1_weighted"] for row in batch_results])
    avg_drift_share = _safe_mean([row["drift_share"] for row in batch_results])
    avg_inference_ms = _safe_mean(inference_times)
    recovery_score = final_post_drift_metrics["f1_weighted"] - baseline_post_drift_f1
    quality_stability = 1.0 - min(1.0, float(np.std([row["f1_weighted"] for row in batch_results])) if batch_results else 0.0)
    promoted_count = sum(1 for event in retrain_events if event["promoted"])
    cost_score = promoted_count * retrain_fixed_cost + float(np.sum(train_times)) * train_seconds_cost
    risk_penalty = downtime_risk * 0.05 + avg_drift_share * drift_penalty_weight + avg_inference_ms * inference_penalty_weight / 1000.0
    composite_score = (
        avg_batch_f1 * batch_weight
        + final_post_drift_metrics["f1_weighted"] * final_weight
        + recovery_score * recovery_weight
        + quality_stability * float(benchmark_cfg["stability_weight"])
        - cost_score
        - risk_penalty
    )

    return {
        "policy": policy.name,
        "scenario": scenario["name"],
        "drift_type": scenario["drift_type"],
        "dataset_name": benchmark_cfg["dataset_name"],
        "baseline_f1": round(baseline_f1, 4),
        "baseline_post_drift_f1": round(baseline_post_drift_f1, 4),
        "avg_batch_f1": round(avg_batch_f1, 4),
        "final_holdout_f1": round(final_metrics["f1_weighted"], 4),
        "final_post_drift_f1": round(final_post_drift_metrics["f1_weighted"], 4),
        "recovery_score": round(recovery_score, 4),
        "quality_stability": round(quality_stability, 4),
        "avg_drift_share": round(avg_drift_share, 4),
        "retrain_count": len(retrain_events),
        "promoted_retrain_count": promoted_count,
        "avg_inference_ms": round(avg_inference_ms, 3),
        "total_train_seconds": round(float(np.sum(train_times)), 4),
        "downtime_risk_events": downtime_risk,
        "composite_score": round(composite_score, 4),
        "batches": batch_results,
        "retrain_events": retrain_events,
    }


def _rank_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for result in results:
        grouped[result["scenario"]].append(result)

    ranked = []
    for scenario_name, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: row["composite_score"], reverse=True)
        for rank, row in enumerate(ordered, start=1):
            enriched = dict(row)
            enriched["rank_within_scenario"] = rank
            ranked.append(enriched)
    return ranked


def _build_summary(ranked_results: List[Dict[str, object]]) -> Dict[str, object]:
    leaderboard: Dict[str, List[float]] = defaultdict(list)
    for row in ranked_results:
        leaderboard[row["policy"]].append(row["rank_within_scenario"])

    overall = []
    for policy, ranks in leaderboard.items():
        policy_rows = [row for row in ranked_results if row["policy"] == policy]
        overall.append(
            {
                "policy": policy,
                "avg_rank": round(float(np.mean(ranks)), 2),
                "avg_composite_score": round(float(np.mean([row["composite_score"] for row in policy_rows])), 4),
                "avg_final_holdout_f1": round(float(np.mean([row["final_holdout_f1"] for row in policy_rows])), 4),
                "avg_final_post_drift_f1": round(float(np.mean([row["final_post_drift_f1"] for row in policy_rows])), 4),
                "avg_train_seconds": round(float(np.mean([row["total_train_seconds"] for row in policy_rows])), 4),
                "avg_promoted_retrains": round(float(np.mean([row["promoted_retrain_count"] for row in policy_rows])), 2),
                "avg_inference_ms": round(float(np.mean([row["avg_inference_ms"] for row in policy_rows])), 4),
            }
        )

    overall.sort(key=lambda row: (row["avg_rank"], -row["avg_composite_score"]))
    return {
        "generated_at": timestamp(),
        "overall_leaderboard": overall,
        "scenario_winners": [
            {
                "scenario": row["scenario"],
                "policy": row["policy"],
                "composite_score": row["composite_score"],
                "final_holdout_f1": row["final_holdout_f1"],
                "final_post_drift_f1": row["final_post_drift_f1"],
            }
            for row in sorted(ranked_results, key=lambda item: (item["scenario"], item["rank_within_scenario"]))
            if row["rank_within_scenario"] == 1
        ],
    }


def _results_to_csv(ranked_results: List[Dict[str, object]], output_path: Path) -> None:
    rows = []
    for item in ranked_results:
        rows.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"batches", "retrain_events"}
            }
        )
    if not rows:
        output_path.write_text("")
        return
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_dashboard_data(summary: Dict[str, object], ranked_results: List[Dict[str, object]]) -> Dict[str, object]:
    scenarios = sorted({row["scenario"] for row in ranked_results})
    policies = sorted({row["policy"] for row in ranked_results})
    return {
        "summary": summary,
        "leaderboard": ranked_results,
        "scenarios": scenarios,
        "policies": policies,
    }


def _render_dashboard_html(summary: Dict[str, object], ranked_results: List[Dict[str, object]]) -> str:
    dashboard_data = json.dumps(_build_dashboard_data(summary, ranked_results))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drift Benchmark Dashboard</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffaf2;
      --ink: #1e2a2f;
      --muted: #6f7c80;
      --accent: #0c7c59;
      --accent-2: #f28f3b;
      --border: #d7d0c4;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      background: radial-gradient(circle at top left, #fff7e8, var(--bg) 55%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
      margin-bottom: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(30, 42, 47, 0.06);
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p {{ color: var(--muted); line-height: 1.5; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(12, 124, 89, 0.10);
      color: var(--accent);
      font-size: 12px;
      margin-right: 8px;
      margin-bottom: 8px;
    }}
    .winner {{
      font-weight: 700;
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .metric {{
      font-size: 28px;
      font-weight: 700;
    }}
    .spark {{
      display: grid;
      gap: 8px;
    }}
    .bar {{
      height: 10px;
      background: rgba(12, 124, 89, 0.14);
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }}
    @media (max-width: 900px) {{
      .hero, .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <section class="card">
        <h1>Drift Response Benchmark</h1>
        <p>Policy comparison across drift scenarios with recovery, stability, runtime, and cost-sensitive scoring.</p>
        <div id="winner-pills"></div>
      </section>
      <section class="card">
        <h2>Top Policy</h2>
        <div class="metric" id="top-policy">-</div>
        <p id="top-policy-meta"></p>
      </section>
    </div>
    <div class="grid">
      <section class="card">
        <h2>Overall Leaderboard</h2>
        <table id="leaderboard-table"></table>
      </section>
      <section class="card">
        <h2>Policy Composite Scores</h2>
        <div id="score-bars" class="spark"></div>
      </section>
      <section class="card">
        <h2>Scenario Winners</h2>
        <table id="scenario-table"></table>
      </section>
      <section class="card">
        <h2>Detailed Results</h2>
        <table id="detail-table"></table>
      </section>
    </div>
  </div>
  <script>
    const data = {dashboard_data};
    const leaderboard = data.summary.overall_leaderboard;
    const winners = data.summary.scenario_winners;
    const detail = data.leaderboard;

    const top = leaderboard[0];
    document.getElementById("top-policy").textContent = top ? top.policy : "-";
    document.getElementById("top-policy-meta").textContent = top
      ? "Avg rank " + top.avg_rank + ", avg final F1 " + top.avg_final_holdout_f1 + ", avg train secs " + top.avg_train_seconds
      : "No data";

    document.getElementById("winner-pills").innerHTML = winners
      .map((winner) => "<span class=\\"pill\\">" + winner.scenario + ": " + winner.policy + "</span>")
      .join("");

    document.getElementById("leaderboard-table").innerHTML =
      "<tr><th>Policy</th><th>Avg Rank</th><th>Avg Composite</th><th>Avg Final F1</th><th>Avg Post-Drift F1</th><th>Avg Promoted Retrains</th><th>Avg Inference ms</th></tr>" +
      leaderboard
        .map((row) => "<tr><td class=\\"" + (row === top ? "winner" : "") + "\\">" + row.policy + "</td><td>" + row.avg_rank + "</td><td>" + row.avg_composite_score + "</td><td>" + row.avg_final_holdout_f1 + "</td><td>" + row.avg_final_post_drift_f1 + "</td><td>" + row.avg_promoted_retrains + "</td><td>" + row.avg_inference_ms + "</td></tr>")
        .join("");

    const maxComposite = Math.max(...leaderboard.map((row) => row.avg_composite_score), 0.0001);
    document.getElementById("score-bars").innerHTML = leaderboard
      .map((row) =>
        "<div><div style=\\"display:flex;justify-content:space-between;\\"><strong>" + row.policy + "</strong><span>" + row.avg_composite_score + "</span></div>" +
        "<div class=\\"bar\\"><span style=\\"width:" + ((row.avg_composite_score / maxComposite) * 100) + "%\\"></span></div></div>"
      )
      .join("");

    document.getElementById("scenario-table").innerHTML =
      "<tr><th>Scenario</th><th>Winning Policy</th><th>Composite</th><th>Final F1</th><th>Post-Drift F1</th></tr>" +
      winners
        .map((row) => "<tr><td>" + row.scenario + "</td><td class=\\"winner\\">" + row.policy + "</td><td>" + row.composite_score + "</td><td>" + row.final_holdout_f1 + "</td><td>" + row.final_post_drift_f1 + "</td></tr>")
        .join("");

    document.getElementById("detail-table").innerHTML =
      "<tr><th>Scenario</th><th>Policy</th><th>Rank</th><th>Retrains</th><th>Stability</th><th>Composite</th></tr>" +
      detail
        .map((row) => "<tr><td>" + row.scenario + "</td><td>" + row.policy + "</td><td>" + row.rank_within_scenario + "</td><td>" + row.retrain_count + "</td><td>" + row.quality_stability + "</td><td>" + row.composite_score + "</td></tr>")
        .join("");
  </script>
</body>
</html>
"""


def _render_markdown(summary: Dict[str, object], ranked_results: List[Dict[str, object]]) -> str:
    lines = [
        "# Drift Response Benchmark",
        "",
        "## Overall Leaderboard",
        "",
        "| Policy | Avg Rank | Avg Composite | Avg Final F1 | Avg Post-Drift F1 | Avg Train Secs | Avg Promoted Retrains | Avg Inference ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["overall_leaderboard"]:
        lines.append(
            f"| {row['policy']} | {row['avg_rank']:.2f} | {row['avg_composite_score']:.4f} | {row['avg_final_holdout_f1']:.4f} | {row['avg_final_post_drift_f1']:.4f} | {row['avg_train_seconds']:.4f} | {row['avg_promoted_retrains']:.2f} | {row['avg_inference_ms']:.4f} |"
        )

    lines.extend(["", "## Scenario Winners", ""])
    for winner in summary["scenario_winners"]:
        lines.append(
            f"- `{winner['scenario']}`: `{winner['policy']}` won with composite `{winner['composite_score']:.4f}`, final F1 `{winner['final_holdout_f1']:.4f}`, and post-drift F1 `{winner['final_post_drift_f1']:.4f}`"
        )

    lines.extend(["", "## Detailed Results", ""])
    for row in ranked_results:
        lines.append(
            f"- `{row['scenario']}` / `{row['policy']}`: rank {row['rank_within_scenario']}, avg batch F1 `{row['avg_batch_f1']:.4f}`, final F1 `{row['final_holdout_f1']:.4f}`, post-drift F1 `{row['final_post_drift_f1']:.4f}`, stability `{row['quality_stability']:.4f}`, retrains `{row['retrain_count']}`, train secs `{row['total_train_seconds']:.4f}`"
        )
    return "\n".join(lines) + "\n"


def run_benchmark(
    output_dir: str,
    data_path: Optional[str] = None,
    config_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, object]:
    """Run all benchmark scenarios and persist results."""
    config = {
        "dataset_name": "wine",
        "initial_model_type": "random_forest",
        "initial_n_estimators": 160,
        "initial_max_depth": 14,
        "retrain_model_type": "extra_trees",
        "retrain_n_estimators": 220,
        "retrain_max_depth": 18,
        "degradation_threshold": 0.08,
        "retrain_window_rows": 144,
        "reference_mix_ratio": 0.35,
        "promotion_validation_rows": 48,
        "promotion_min_gain": 0.02,
        "promotion_max_regression": 0.01,
        "train_seconds_cost": 0.02,
        "retrain_fixed_cost": 0.03,
        "drift_penalty_weight": 0.08,
        "recovery_weight": 0.18,
        "batch_weight": 0.42,
        "final_weight": 0.28,
        "stability_weight": 0.10,
        "inference_penalty_weight": 0.01,
        "scenarios": [],
        "policies": [],
    }
    if config_path:
        with open(config_path) as handle:
            loaded = yaml.safe_load(handle) or {}
        config.update(loaded.get("benchmark", loaded))
    if dataset_name:
        config["dataset_name"] = dataset_name

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_specs = parse_scenario_specs(config.get("scenarios"))
    scenarios = create_benchmark_suite(
        output_dir=str(out_dir / "scenarios"),
        data_path=data_path,
        dataset_name=str(config["dataset_name"]),
        scenario_specs=scenario_specs,
    )
    policies = build_policies(config.get("policies"))
    raw_results = []
    for scenario in scenarios:
        for policy in policies:
            raw_results.append(
                _simulate_policy(
                    scenario=scenario,
                    policy=policy,
                    benchmark_cfg=config,
                )
            )

    ranked_results = _rank_results(raw_results)
    summary = _build_summary(ranked_results)
    summary["dataset_name"] = config["dataset_name"]
    summary["policy_count"] = len(policies)
    summary["scenario_count"] = len(scenarios)

    save_json({"results": ranked_results}, str(out_dir / "benchmark_results.json"))
    save_json(summary, str(out_dir / "benchmark_summary.json"))
    _results_to_csv(ranked_results, out_dir / "benchmark_results.csv")
    markdown = _render_markdown(summary, ranked_results)
    (out_dir / "benchmark_report.md").write_text(markdown)
    (out_dir / "dashboard.html").write_text(_render_dashboard_html(summary, ranked_results))

    return {
        "summary": summary,
        "results": ranked_results,
        "report_path": str(out_dir / "benchmark_report.md"),
        "dashboard_path": str(out_dir / "dashboard.html"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/benchmark")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--config", default="configs/benchmark_config.yaml")
    args = parser.parse_args()

    result = run_benchmark(
        output_dir=args.output_dir,
        data_path=args.data_path,
        config_path=args.config,
        dataset_name=args.dataset_name,
    )
    print(json.dumps(result["summary"], indent=2))
