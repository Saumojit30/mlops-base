import os
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

HF_REPO_ID = "Jit0777/california-housing-model"

def get_model_path(filename):
    """Loads local artifact or downloads from Hugging Face Model Hub if in CI environment."""
    local_path = os.path.join("models", filename)
    if os.path.exists(local_path):
        return local_path
    print(f"Downloading {filename} from Hugging Face Model Hub ({HF_REPO_ID})...")
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)

def test_models_exist():
    rf_path = get_model_path("rf_model.joblib")
    xgb_path = get_model_path("xgb_model.joblib")
    assert os.path.exists(rf_path), "Random Forest model weights could not be loaded!"
    assert os.path.exists(xgb_path), "XGBoost model weights could not be loaded!"

def test_model_inference():
    rf_path = get_model_path("rf_model.joblib")
    xgb_path = get_model_path("xgb_model.joblib")
    
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)
    
    dummy_input = pd.DataFrame({
        "MedInc": [3.5],
        "HouseAge": [28.0],
        "AveRooms": [5.0],
        "AveBedrms": [1.1],
        "Population": [1400.0],
        "AveOccup": [3.0],
        "Latitude": [35.0],
        "Longitude": [-119.0],
        "RoomsPerHousehold": [5.0 / 3.0],
        "BedroomsPerRoom": [1.1 / 5.0],
        "PopulationPerHousehold": [1400.0 / 3.0]
    })
    
    rf_pred = float(rf_model.predict(dummy_input)[0])
    xgb_pred = float(xgb_model.predict(dummy_input)[0])
    
    assert rf_pred > 0, "RF prediction should be positive"
    assert xgb_pred > 0, "XGBoost prediction should be positive"
