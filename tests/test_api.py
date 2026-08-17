"""Tests for the saved model artifact and the FastAPI serving layer."""
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "heart_pipeline.joblib"

VALID_PATIENT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
    "ca": 0, "thal": 1,
}


# ----------------------------------------------------------------- model tests
def test_model_artifact_exists():
    assert MODEL.exists(), "Run `python src/train.py` to produce the model."


def test_model_predicts_binary():
    pipe = joblib.load(MODEL)
    df = pd.DataFrame([VALID_PATIENT])
    pred = pipe.predict(df)
    assert pred[0] in (0, 1)


def test_model_probabilities_sum_to_one():
    pipe = joblib.load(MODEL)
    df = pd.DataFrame([VALID_PATIENT])
    proba = pipe.predict_proba(df)[0]
    assert abs(proba.sum() - 1.0) < 1e-6
    assert all(0 <= p <= 1 for p in proba)


# ------------------------------------------------------------------- api tests
@pytest.fixture(scope="module")
def client():
    from src.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid(client):
    r = client.post("/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert 0 <= body["confidence"] <= 1
    assert body["label"] in ("Heart disease", "No heart disease")


def test_predict_rejects_out_of_range(client):
    bad = dict(VALID_PATIENT, age=999)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # pydantic validation


def test_predict_rejects_missing_field(client):
    bad = dict(VALID_PATIENT)
    del bad["chol"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"patients": [VALID_PATIENT, VALID_PATIENT]})
    assert r.status_code == 200
    assert len(r.json()) == 2
