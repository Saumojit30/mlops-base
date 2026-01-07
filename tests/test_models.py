import os
import joblib
import pandas as pd

def test_models_exist():
    assert os.path.exists("models/rf_model.joblib"), "Random Forest model weights missing!"
    assert os.path.exists("models/xgb_model.joblib"), "XGBoost model weights missing!"

def test_model_inference():
    rf_model = joblib.load("models/rf_model.joblib")
    xgb_model = joblib.load("models/xgb_model.joblib")
    
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
