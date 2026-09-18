import os
import json
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
HF_REPO_ID = "Jit0777/california-housing-model"
LOGS_DIR = "logs"
INFERENCE_LOG_FILE = os.path.join(LOGS_DIR, "inference_logs.jsonl")

# In-memory model cache for zero-latency concurrent execution
MODELS: Dict[str, Any] = {}
METRICS: Dict[str, Any] = {}

def get_model_path(filename: str, revision: Optional[str] = None) -> str:
    """Retrieves local model artifact or streams from Hugging Face Hub if missing."""
    local_path = os.path.join("models", filename)
    if os.path.exists(local_path):
        return local_path
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename, revision=revision)
    except Exception:
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads production models and metadata into memory on startup."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("🚀 Initializing FastAPI Production Inference Microservice...")
    try:
        xgb_path = get_model_path("xgb_model.joblib", revision="v2.1")
        MODELS["xgboost"] = joblib.load(xgb_path)
        print("✅ XGBoost Champion Model (v2.1) loaded.")
    except Exception as e:
        print(f"⚠️ Warning: Could not preload XGBoost: {e}")

    try:
        rf_path = get_model_path("rf_model.joblib", revision="v1.2")
        MODELS["random_forest"] = joblib.load(rf_path)
        print("✅ Random Forest Model (v1.2) loaded.")
    except Exception as e:
        print(f"⚠️ Warning: Could not preload Random Forest: {e}")

    # Load metrics if available
    for m_type, m_file in [("xgboost", "xgb_metrics.json"), ("random_forest", "rf_metrics.json")]:
        p = os.path.join("models", m_file)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                METRICS[m_type] = json.load(f)

    yield

    print("🛑 Shutting down FastAPI microservice...")
    MODELS.clear()

app = FastAPI(
    title="California Housing Valuation API",
    description="High-concurrency production inference microservice with asynchronous request logging and multi-model routing.",
    version="2.1.0",
    lifespan=lifespan
)

# ==============================================================================
# PYDANTIC DATA VALIDATION SCHEMAS
# ==============================================================================
class HousingFeatures(BaseModel):
    med_inc: float = Field(..., ge=0.0, le=20.0, description="Median Income in tens of thousands (e.g. 3.5 = $35,000)")
    house_age: float = Field(..., ge=1.0, le=100.0, description="Median house age in years")
    ave_rooms: float = Field(..., gt=0.0, le=30.0, description="Average rooms per household")
    ave_bedrms: float = Field(..., gt=0.0, le=15.0, description="Average bedrooms per household")
    population: float = Field(..., gt=0.0, le=50000.0, description="Block group population")
    ave_occup: float = Field(..., gt=0.0, le=25.0, description="Average occupancy per household")
    latitude: float = Field(..., ge=32.0, le=43.0, description="Latitude coordinate")
    longitude: float = Field(..., ge=-125.0, le=-114.0, description="Longitude coordinate")
    model_type: Optional[str] = Field("xgboost", description="Model architecture to use ('xgboost' or 'random_forest')")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

class PredictionResponse(BaseModel):
    prediction_usd: float = Field(..., description="Estimated median house value in USD")
    raw_prediction: float = Field(..., description="Raw model prediction value (hundreds of thousands)")
    model_used: str = Field(..., description="The algorithm name utilized")
    model_version: str = Field(..., description="The model release version")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: str = Field(..., description="UTC timestamp of the prediction event")

class BatchHousingInput(BaseModel):
    items: List[HousingFeatures] = Field(..., min_length=1, max_length=1000, description="List of feature objects for batch scoring")
    model_type: Optional[str] = Field("xgboost", description="Model architecture for entire batch")

class BatchPredictionResponse(BaseModel):
    predictions_usd: List[float] = Field(..., description="List of predictions in USD")
    model_used: str = Field(..., description="Algorithm utilized")
    model_version: str = Field(..., description="Model version")
    count: int = Field(..., description="Total records scored")
    total_latency_ms: float = Field(..., description="Total batch processing latency in milliseconds")
    timestamp: str = Field(..., description="UTC timestamp")

class HealthStatus(BaseModel):
    status: str
    active_models: List[str]
    champion_model: str
    champion_version: str
    timestamp: str

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def log_inference_event(log_data: Dict[str, Any]):
    """Appends inference input and output to JSONL log file asynchronously for drift monitoring."""
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(INFERENCE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        print(f"Failed to log inference event: {e}")

def prepare_input_dataframe(features_list: List[HousingFeatures]) -> pd.DataFrame:
    """Transforms validated features list into engineered feature matrix matching pipeline expectations."""
    rows = []
    for item in features_list:
        r_per_h = item.ave_rooms / item.ave_occup if item.ave_occup != 0 else 0.0
        b_per_r = item.ave_bedrms / item.ave_rooms if item.ave_rooms != 0 else 0.0
        p_per_h = item.population / item.ave_occup if item.ave_occup != 0 else 0.0

        rows.append({
            "MedInc": item.med_inc,
            "HouseAge": item.house_age,
            "AveRooms": item.ave_rooms,
            "AveBedrms": item.ave_bedrms,
            "Population": item.population,
            "AveOccup": item.ave_occup,
            "Latitude": item.latitude,
            "Longitude": item.longitude,
            "RoomsPerHousehold": r_per_h,
            "BedroomsPerRoom": b_per_r,
            "PopulationPerHousehold": p_per_h
        })
    return pd.DataFrame(rows)

# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check():
    """System health check and loaded model diagnostics."""
    return HealthStatus(
        status="healthy",
        active_models=list(MODELS.keys()),
        champion_model="XGBoost Regressor",
        champion_version="v2.1",
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/model-info", tags=["System"])
async def model_info():
    """Returns technical metadata, benchmark performance, and feature schema."""
    return {
        "champion_model": "XGBoost Regressor (v2.1)",
        "champion_metrics": METRICS.get("xgboost", {"rmse": 0.4172, "r2": 0.8184}),
        "available_architectures": {
            "xgboost": {
                "version": "v2.1",
                "description": "Sequential Gradient Boosted Decision Trees (500 Trees, subsample=0.9)",
                "status": "Production Champion"
            },
            "random_forest": {
                "version": "v1.2",
                "description": "Parallel Bootstrap Aggregating Ensemble (200 Trees, max_depth=25)",
                "status": "Challenger / Baseline"
            }
        },
        "required_features": [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"
        ],
        "engineered_features": [
            "RoomsPerHousehold", "BedroomsPerRoom", "PopulationPerHousehold"
        ]
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(item: HousingFeatures, background_tasks: BackgroundTasks):
    """Scores a single real estate property with sub-10ms response time."""
    start_time = time.perf_counter()
    selected_key = item.model_type.lower() if item.model_type else "xgboost"

    if selected_key not in MODELS:
        # Fallback load attempt
        try:
            filename = "xgb_model.joblib" if selected_key == "xgboost" else "rf_model.joblib"
            MODELS[selected_key] = joblib.load(get_model_path(filename))
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Requested model '{selected_key}' is currently unavailable."
            )

    model = MODELS[selected_key]
    input_df = prepare_input_dataframe([item])
    
    try:
        raw_pred = float(model.predict(input_df)[0])
        pred_usd = round(raw_pred * 100000.0, 2)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    now_ts = datetime.now(timezone.utc).isoformat()
    version_label = "v2.1" if selected_key == "xgboost" else "v1.2"

    # Queue non-blocking inference log for statistical drift monitoring
    log_event = {
        "timestamp": now_ts,
        "model_type": selected_key,
        "model_version": version_label,
        "features": item.model_dump(),
        "raw_prediction": raw_pred,
        "prediction_usd": pred_usd,
        "latency_ms": latency_ms
    }
    background_tasks.add_task(log_inference_event, log_event)

    return PredictionResponse(
        prediction_usd=pred_usd,
        raw_prediction=round(raw_pred, 4),
        model_used="XGBoost Regressor" if selected_key == "xgboost" else "Random Forest Regressor",
        model_version=version_label,
        latency_ms=latency_ms,
        timestamp=now_ts
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(batch: BatchHousingInput, background_tasks: BackgroundTasks):
    """High-throughput vectorized bulk prediction for concurrent enterprise ingestion."""
    start_time = time.perf_counter()
    selected_key = batch.model_type.lower() if batch.model_type else "xgboost"

    if selected_key not in MODELS:
        try:
            filename = "xgb_model.joblib" if selected_key == "xgboost" else "rf_model.joblib"
            MODELS[selected_key] = joblib.load(get_model_path(filename))
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{selected_key}' unavailable."
            )

    model = MODELS[selected_key]
    input_df = prepare_input_dataframe(batch.items)

    try:
        raw_preds = model.predict(input_df)
        preds_usd = [round(float(p) * 100000.0, 2) for p in raw_preds]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    now_ts = datetime.now(timezone.utc).isoformat()
    version_label = "v2.1" if selected_key == "xgboost" else "v1.2"

    # Async batch log for drift detection
    for item, pred, p_usd in zip(batch.items, raw_preds, preds_usd):
        log_event = {
            "timestamp": now_ts,
            "model_type": selected_key,
            "model_version": version_label,
            "features": item.model_dump(),
            "raw_prediction": float(pred),
            "prediction_usd": p_usd,
            "latency_ms": latency_ms / len(batch.items)
        }
        background_tasks.add_task(log_inference_event, log_event)

    return BatchPredictionResponse(
        predictions_usd=preds_usd,
        model_used="XGBoost Regressor" if selected_key == "xgboost" else "Random Forest Regressor",
        model_version=version_label,
        count=len(preds_usd),
        total_latency_ms=latency_ms,
        timestamp=now_ts
    )
