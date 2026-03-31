"""
Data Drift Monitor using Evidently AI
Compares production batches against reference data and triggers retraining if needed.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently not installed. Using fallback drift detection.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports/drift")


def fallback_drift_detection(reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.3) -> dict:
    """
    Simple PSI-based drift detection when Evidently is not available.
    Population Stability Index (PSI) > 0.2 indicates significant drift.
    """
    feature_cols = [c for c in reference.columns if c != "target"]
    drift_scores = {}
    drifted_features = []

    for col in feature_cols:
        ref_vals = reference[col].dropna()
        cur_vals = current[col].dropna()
        combined = np.concatenate([ref_vals.to_numpy(), cur_vals.to_numpy()])
        bins = np.histogram_bin_edges(combined, bins=10)
        if len(np.unique(bins)) < 2:
            drift_scores[col] = 0.0
            continue
        ref_hist, _ = np.histogram(ref_vals, bins=bins)
        cur_hist, _ = np.histogram(cur_vals, bins=bins)
        ref_hist = ref_hist / max(ref_hist.sum(), 1)
        cur_hist = cur_hist / max(cur_hist.sum(), 1)
        ref_hist = np.where(ref_hist == 0, 1e-6, ref_hist)
        cur_hist = np.where(cur_hist == 0, 1e-6, cur_hist)
        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
        drift_scores[col] = float(psi)
        if psi > 0.2:
            drifted_features.append(col)

    drift_detected = len(drifted_features) / len(feature_cols) > threshold
    return {
        "drift_detected": drift_detected,
        "drift_share": len(drifted_features) / len(feature_cols),
        "drifted_features": drifted_features,
        "feature_psi_scores": drift_scores,
        "method": "PSI",
    }


def run_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_path: str,
) -> dict:
    """Run full Evidently drift report and save HTML."""
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference, current_data=current)
    report.save_html(report_path)

    result = report.as_dict()
    metrics = result.get("metrics", [])

    drift_metric = next(
        (m for m in metrics if m.get("metric") == "DatasetDriftMetric"),
        None,
    )

    if drift_metric:
        drift_result = drift_metric.get("result", {})
        return {
            "drift_detected": drift_result.get("dataset_drift", False),
            "drift_share": drift_result.get("share_of_drifted_columns", 0),
            "drifted_features": drift_result.get("drifted_columns", []),
            "drifted_feature_count": drift_result.get("number_of_drifted_columns", 0),
            "method": "Evidently",
            "report_path": report_path,
        }
    return {"drift_detected": False, "method": "Evidently", "error": "Could not extract metrics"}


def monitor_batch(
    reference_path: str,
    current_path: str,
    threshold: float = DRIFT_THRESHOLD,
    output_dir: str = REPORTS_DIR,
) -> dict:
    """
    Run drift monitoring on a single batch.
    Returns drift result dict with retraining recommendation.
    """
    os.makedirs(output_dir, exist_ok=True)
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    # Drop target for feature drift analysis
    ref_features = reference.drop(columns=["target"], errors="ignore")
    cur_features = current.drop(columns=["target"], errors="ignore")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = Path(current_path).stem

    if EVIDENTLY_AVAILABLE:
        report_path = os.path.join(output_dir, f"drift_report_{batch_name}_{timestamp}.html")
        drift_result = run_evidently_report(ref_features, cur_features, report_path)
    else:
        drift_result = fallback_drift_detection(ref_features, cur_features, threshold)

    drift_result["batch_file"] = current_path
    drift_result["timestamp"] = timestamp
    drift_result["threshold"] = threshold
    drift_result["trigger_retraining"] = drift_result.get("drift_share", 0) > threshold

    # Save JSON result
    result_path = os.path.join(output_dir, f"drift_result_{batch_name}_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump(drift_result, f, indent=2)

    logger.info(f"Drift detected: {drift_result['drift_detected']} | Share: {drift_result.get('drift_share', 'N/A'):.2%}")
    logger.info(f"Trigger retraining: {drift_result['trigger_retraining']}")

    return drift_result


def monitor_directory(
    reference_path: str,
    production_dir: str,
    threshold: float = DRIFT_THRESHOLD,
    output_dir: str = REPORTS_DIR,
) -> list:
    """Monitor all batches in a production directory."""
    results = []
    batch_files = sorted(Path(production_dir).glob("batch_*.csv"))

    logger.info(f"Monitoring {len(batch_files)} batches against reference: {reference_path}")

    for batch_file in batch_files:
        result = monitor_batch(str(reference_path), str(batch_file), threshold, output_dir)
        results.append(result)

    # Summary
    n_drifted = sum(1 for r in results if r["drift_detected"])
    summary = {
        "total_batches": len(results),
        "drifted_batches": n_drifted,
        "retraining_triggered": sum(1 for r in results if r["trigger_retraining"]),
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    summary_path = os.path.join(output_dir, "monitoring_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Summary: {n_drifted}/{len(results)} batches drifted")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--current", type=str, default=None)
    parser.add_argument("--production-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD)
    parser.add_argument("--output-dir", type=str, default=REPORTS_DIR)
    args = parser.parse_args()

    if args.current:
        result = monitor_batch(args.reference, args.current, args.threshold, args.output_dir)
        print(json.dumps(result, indent=2))
    elif args.production_dir:
        monitor_directory(args.reference, args.production_dir, args.threshold, args.output_dir)
    else:
        print("Provide --current or --production-dir")
