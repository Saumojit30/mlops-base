import os
import json
import pytest
from fastapi.testclient import TestClient
from tests.test_models import get_or_create_model
from src.api import app, INFERENCE_LOG_FILE

# Ensure test models are primed before API tests execute
get_or_create_model("xgb_model.joblib")
get_or_create_model("rf_model.joblib")

client = TestClient(app)

def test_health_endpoint():
    """Validates /health responds with 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "champion_model" in data
    assert "active_models" in data

def test_model_info_endpoint():
    """Validates /model-info returns required metadata and metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "champion_metrics" in data
    assert "required_features" in data
    assert "available_architectures" in data

def test_single_prediction_xgboost():
    """Tests single property prediction using the champion XGBoost model."""
    payload = {
        "med_inc": 3.87,
        "house_age": 28.0,
        "ave_rooms": 5.4,
        "ave_bedrms": 1.05,
        "population": 1425.0,
        "ave_occup": 3.0,
        "latitude": 35.2,
        "longitude": -119.5,
        "model_type": "xgboost"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_usd"] > 0
    assert "XGBoost" in data["model_used"]
    assert data["model_version"] == "v2.1"
    assert data["latency_ms"] >= 0

def test_single_prediction_random_forest():
    """Tests single property prediction using the Random Forest model."""
    payload = {
        "med_inc": 4.5,
        "house_age": 15.0,
        "ave_rooms": 6.0,
        "ave_bedrms": 1.1,
        "population": 800.0,
        "ave_occup": 2.5,
        "latitude": 34.05,
        "longitude": -118.25,
        "model_type": "random_forest"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_usd"] > 0
    assert "Random Forest" in data["model_used"]

def test_input_validation_error():
    """Ensures Pydantic validation rejects negative income or out-of-bound coordinates."""
    invalid_payload = {
        "med_inc": -3.0,  # Invalid: negative income
        "house_age": 28.0,
        "ave_rooms": 5.4,
        "ave_bedrms": 1.05,
        "population": 1425.0,
        "ave_occup": 3.0,
        "latitude": 95.0,  # Invalid: latitude > 43.0
        "longitude": -119.5
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_batch_prediction():
    """Tests high-throughput batch prediction endpoint."""
    batch_payload = {
        "items": [
            {
                "med_inc": 3.5,
                "house_age": 25.0,
                "ave_rooms": 5.0,
                "ave_bedrms": 1.0,
                "population": 1200.0,
                "ave_occup": 3.0,
                "latitude": 36.0,
                "longitude": -120.0
            },
            {
                "med_inc": 6.2,
                "house_age": 10.0,
                "ave_rooms": 7.0,
                "ave_bedrms": 1.2,
                "population": 900.0,
                "ave_occup": 2.4,
                "latitude": 37.5,
                "longitude": -122.0
            }
        ],
        "model_type": "xgboost"
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions_usd"]) == 2
    assert all(p > 0 for p in data["predictions_usd"])

def test_inference_logging_occurred():
    """Verifies that background inference events were successfully logged to disk."""
    assert os.path.exists(INFERENCE_LOG_FILE), "Inference log file was not created!"
    with open(INFERENCE_LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) > 0, "Inference log file is empty!"
    latest_event = json.loads(lines[-1])
    assert "timestamp" in latest_event
    assert "features" in latest_event
    assert "prediction_usd" in latest_event
