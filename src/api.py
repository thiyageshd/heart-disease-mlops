"""
api.py
------
Phase 3: model-serving API.

Loads the pipeline saved by src/train.py (preprocess + classifier in one object)
and exposes:
    GET  /health          -> liveness + model metadata
    POST /predict         -> single-patient risk prediction
    POST /predict/batch   -> multiple patients in one call
    GET  /                 -> redirect to interactive docs

Because the API loads the *same* fitted pipeline used at training time, there is
no train/serve skew: identical imputation, scaling, and encoding are applied.

Run locally:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
    # interactive docs at http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# ------------------------------------------------------------------ logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-api")

# ------------------------------------------------------------------ artifacts
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

_model = None
_meta: dict = {}


def load_model():
    """Load the fitted pipeline and metadata once at startup."""
    global _model, _meta
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python src/train.py` first."
        )
    _model = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        _meta = json.loads(META_PATH.read_text())
    logger.info("Model loaded: winner=%s", _meta.get("winner", "unknown"))
    return _model


# ------------------------------------------------------------------ schema
class PatientFeatures(BaseModel):
    """One patient record in the raw UCI Cleveland schema (13 features)."""
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[0, 1, 2, 3] = Field(..., description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=260, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression vs rest")
    slope: Literal[0, 1, 2] = Field(..., description="Slope of peak exercise ST (0-2)")
    ca: Literal[0, 1, 2, 3, 4] = Field(..., description="Major vessels colored (0-3; 4=unknown)")
    thal: Literal[0, 1, 2, 3] = Field(..., description="Thalassemia (0=unknown,1-3)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 57, "sex": 1, "cp": 0, "trestbps": 130, "chol": 236,
                "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0,
                "oldpeak": 0.0, "slope": 1, "ca": 1, "thal": 2,
            }
        }
    }


class BatchRequest(BaseModel):
    patients: list[PatientFeatures]


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no disease, 1 = disease")
    label: str
    probability: float = Field(..., description="P(disease), the positive class")
    confidence: float = Field(..., description="Confidence in the predicted class")


# ------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()          # startup: load the pipeline once
    yield
    # (nothing to clean up on shutdown)


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart-disease risk from patient health data. "
                "Serves the model trained in the MLOps pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---- Prometheus metrics: exposes /metrics with request counts, latency, etc.
# Optional import so the app still runs if the package isn't installed.
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus instrumentation enabled at /metrics")
except ImportError:  # pragma: no cover
    logger.warning("prometheus-fastapi-instrumentator not installed; /metrics disabled")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Liveness probe + model info (used by K8s readiness/liveness checks)."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model": _meta.get("winner", "unknown"),
        "metrics": _meta.get("metrics", {}),
    }


def _predict_df(df: pd.DataFrame) -> list[PredictionResponse]:
    feature_order = _meta.get("feature_order", list(df.columns))
    df = df[feature_order]
    preds = _model.predict(df)
    probas = _model.predict_proba(df)[:, 1]
    out = []
    for p, prob in zip(preds, probas):
        p = int(p)
        confidence = float(prob if p == 1 else 1 - prob)
        out.append(PredictionResponse(
            prediction=p,
            label="Heart disease" if p == 1 else "No heart disease",
            probability=round(float(prob), 4),
            confidence=round(confidence, 4),
        ))
    return out


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame([features.model_dump()])
        result = _predict_df(df)[0]
        logger.info("predict -> %s (p=%.3f)", result.prediction, result.probability)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(req: BatchRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.patients:
        raise HTTPException(status_code=400, detail="Empty patient list")
    try:
        df = pd.DataFrame([p.model_dump() for p in req.patients])
        results = _predict_df(df)
        logger.info("batch predict -> %d records", len(results))
        return results
    except Exception as exc:  # noqa: BLE001
        logger.exception("batch prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
