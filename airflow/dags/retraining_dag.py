"""
Airflow DAG: Automated MLOps Retraining Pipeline
Runs every 6 hours: ingests data -> detects drift -> retrains if needed -> deploys
"""

from datetime import datetime, timedelta
import json
import os
import subprocess
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Variable.get("MLOPS_BASE_DIR", default_var="/opt/airflow/mlops")
DATA_DIR = Variable.get("DATA_DIR", default_var=f"{BASE_DIR}/data/production")
MODEL_DIR = Variable.get("MODEL_DIR", default_var=f"{BASE_DIR}/model_artifacts")
REFERENCE_DATA = Variable.get("REFERENCE_DATA", default_var=f"{BASE_DIR}/data/reference.csv")
DRIFT_THRESHOLD = float(Variable.get("DRIFT_THRESHOLD", default_var="0.3"))
API_URL = Variable.get("API_URL", default_var="http://mlops-api:8000")

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "mlops_retraining_pipeline",
    default_args=default_args,
    description="Automated drift detection and model retraining",
    schedule_interval="0 */6 * * *",  # Every 6 hours
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "retraining", "drift"],
)


# ─── Tasks ─────────────────────────────────────────────────────────────────────

def ingest_production_data(**context):
    """Simulate ingesting latest production batch from S3/storage."""
    import glob
    batch_files = sorted(glob.glob(f"{DATA_DIR}/batch_*.csv"))
    if not batch_files:
        raise ValueError(f"No batch files found in {DATA_DIR}")

    latest_batch = batch_files[-1]
    logger.info(f"Latest batch: {latest_batch}")
    context["task_instance"].xcom_push(key="latest_batch", value=latest_batch)
    context["task_instance"].xcom_push(key="batch_count", value=len(batch_files))
    return latest_batch


def check_drift(**context):
    """Run drift detection using Evidently/PSI."""
    import sys
    sys.path.insert(0, BASE_DIR)
    from src.monitoring.drift_monitor import monitor_batch

    latest_batch = context["task_instance"].xcom_pull(
        task_ids="ingest_data", key="latest_batch"
    )

    result = monitor_batch(
        reference_path=REFERENCE_DATA,
        current_path=latest_batch,
        threshold=DRIFT_THRESHOLD,
    )

    context["task_instance"].xcom_push(key="drift_result", value=result)
    context["task_instance"].xcom_push(key="drift_detected", value=result["drift_detected"])
    context["task_instance"].xcom_push(key="drift_share", value=result.get("drift_share", 0))

    logger.info(f"Drift detected: {result['drift_detected']} (share: {result.get('drift_share', 0):.2%})")
    return result


def branch_on_drift(**context):
    """Branch: retrain if drift exceeds threshold."""
    drift_result = context["task_instance"].xcom_pull(
        task_ids="detect_drift", key="drift_result"
    )
    if drift_result and drift_result.get("trigger_retraining", False):
        logger.info("Drift threshold exceeded -> triggering retraining")
        return "retrain_model"
    else:
        logger.info("No significant drift -> skipping retraining")
        return "no_retraining_needed"


def retrain_model(**context):
    """Trigger model retraining using latest data."""
    import sys
    sys.path.insert(0, BASE_DIR)

    latest_batch = context["task_instance"].xcom_pull(
        task_ids="ingest_data", key="latest_batch"
    )

    # Import and run training
    from src.training.train import main as train_main
    import argparse

    class Args:
        data_path = latest_batch
        model_dir = MODEL_DIR
        n_estimators = 150
        max_depth = 12

    run_id, metrics = train_main(Args())
    logger.info(f"Retraining complete. Run ID: {run_id}, F1: {metrics['f1_weighted']:.4f}")
    context["task_instance"].xcom_push(key="run_id", value=run_id)
    context["task_instance"].xcom_push(key="new_f1", value=metrics["f1_weighted"])
    return run_id


def validate_new_model(**context):
    """Validate new model beats minimum performance threshold."""
    new_f1 = context["task_instance"].xcom_pull(
        task_ids="retrain_model", key="new_f1"
    )
    MIN_F1 = 0.75
    if new_f1 < MIN_F1:
        raise ValueError(f"New model F1 {new_f1:.4f} below threshold {MIN_F1}. Aborting deployment.")
    logger.info(f"Model validation passed: F1={new_f1:.4f}")
    return True


def deploy_model(**context):
    """Deploy model by calling the API reload endpoint."""
    import requests
    try:
        response = requests.post(f"{API_URL}/reload", timeout=30)
        response.raise_for_status()
        logger.info(f"Model deployed successfully: {response.json()}")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

    run_id = context["task_instance"].xcom_pull(task_ids="retrain_model", key="run_id")
    new_f1 = context["task_instance"].xcom_pull(task_ids="retrain_model", key="new_f1")
    drift_share = context["task_instance"].xcom_pull(task_ids="detect_drift", key="drift_share")

    logger.info(
        f"Deployment complete | Run: {run_id} | F1: {new_f1:.4f} | Drift: {drift_share:.2%}"
    )


def log_pipeline_metrics(**context):
    """Log pipeline run metrics to a summary file."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dag_run_id": context["run_id"],
        "drift_detected": context["task_instance"].xcom_pull(task_ids="detect_drift", key="drift_detected"),
        "drift_share": context["task_instance"].xcom_pull(task_ids="detect_drift", key="drift_share"),
    }
    os.makedirs(f"{BASE_DIR}/logs", exist_ok=True)
    with open(f"{BASE_DIR}/logs/pipeline_runs.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    logger.info(f"Pipeline metrics logged: {summary}")


# ─── DAG Structure ─────────────────────────────────────────────────────────────

ingest = PythonOperator(
    task_id="ingest_data",
    python_callable=ingest_production_data,
    dag=dag,
)

detect_drift = PythonOperator(
    task_id="detect_drift",
    python_callable=check_drift,
    dag=dag,
)

branch = BranchPythonOperator(
    task_id="branch_drift_check",
    python_callable=branch_on_drift,
    dag=dag,
)

no_retrain = DummyOperator(task_id="no_retraining_needed", dag=dag)

retrain = PythonOperator(
    task_id="retrain_model",
    python_callable=retrain_model,
    dag=dag,
    execution_timeout=timedelta(minutes=30),
)

validate = PythonOperator(
    task_id="validate_model",
    python_callable=validate_new_model,
    dag=dag,
)

deploy = PythonOperator(
    task_id="deploy_model",
    python_callable=deploy_model,
    dag=dag,
)

log_metrics = PythonOperator(
    task_id="log_metrics",
    python_callable=log_pipeline_metrics,
    dag=dag,
    trigger_rule="none_failed_min_one_success",
)

# ─── Dependencies ───────────────────────────────────────────────────────────────
ingest >> detect_drift >> branch
branch >> no_retrain >> log_metrics
branch >> retrain >> validate >> deploy >> log_metrics
