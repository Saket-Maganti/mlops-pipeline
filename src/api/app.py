"""
FastAPI Inference Server
Serves wine classifier predictions with health checks, metrics, and drift logging.
"""

from contextlib import asynccontextmanager
import os
import json
import time
import logging
import joblib
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Wine Classifier API",
    description="Production ML inference with drift monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model_artifacts")
PREDICTIONS_LOG = os.getenv("PREDICTIONS_LOG", "/app/logs/predictions.jsonl")

model = None
scaler = None
metadata = None


def load_model():
    global model, scaler, metadata
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path = os.path.join(MODEL_DIR, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    logger.info(f"Model loaded. Features: {len(metadata['feature_names'])}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
    except Exception as e:
        logger.error(f"Model load failed: {e}")
    yield


app.router.lifespan_context = lifespan


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Feature values in order")
    request_id: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    instances: List[List[float]]
    request_ids: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]
    predicted_class: str
    request_id: Optional[str]
    latency_ms: float
    model_version: str


def log_prediction(request_data: dict, response_data: dict):
    """Log prediction for drift monitoring."""
    os.makedirs(os.path.dirname(PREDICTIONS_LOG), exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "features": request_data.get("features"),
        "prediction": response_data.get("prediction"),
        "probability": response_data.get("probability"),
    }
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    payload = {
        "status": "healthy" if model is not None else "model_not_loaded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
    }
    if model is None:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/info")
def info():
    if metadata is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "feature_names": metadata["feature_names"],
        "classes": metadata["classes"],
        "trained_at": metadata["trained_at"],
        "metrics": metadata["metrics"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    if len(request.features) != len(metadata["feature_names"]):
        raise HTTPException(
            400,
            f"Expected {len(metadata['feature_names'])} features, got {len(request.features)}",
        )

    start = time.time()
    X = np.array(request.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0].tolist()
    latency = (time.time() - start) * 1000

    class_names = ["class_0", "class_1", "class_2"]
    response = PredictionResponse(
        prediction=pred,
        probability=proba,
        predicted_class=class_names[pred],
        request_id=request.request_id,
        latency_ms=round(latency, 2),
        model_version=metadata.get("trained_at", "unknown"),
    )

    background_tasks.add_task(log_prediction, request.model_dump(), response.model_dump())
    return response


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if not request.instances:
        raise HTTPException(400, "No instances provided")
    expected_features = len(metadata["feature_names"])
    if any(len(instance) != expected_features for instance in request.instances):
        raise HTTPException(400, f"Each instance must include {expected_features} features")

    start = time.time()
    X = np.array(request.instances)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled).tolist()
    probas = model.predict_proba(X_scaled).tolist()
    latency = (time.time() - start) * 1000

    return {
        "predictions": preds,
        "probabilities": probas,
        "count": len(preds),
        "latency_ms": round(latency, 2),
    }


@app.post("/reload")
def reload_model():
    """Reload model from disk (called after retraining)."""
    try:
        load_model()
        return {"status": "reloaded", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(500, f"Reload failed: {str(e)}")


@app.get("/metrics/predictions")
def prediction_metrics():
    """Return basic prediction stats from log."""
    if not os.path.exists(PREDICTIONS_LOG):
        return {"total_predictions": 0}

    records = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

    if not records:
        return {"total_predictions": 0}

    preds = [r["prediction"] for r in records]
    from collections import Counter
    return {
        "total_predictions": len(records),
        "class_distribution": Counter(preds),
        "last_prediction": records[-1]["timestamp"],
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
