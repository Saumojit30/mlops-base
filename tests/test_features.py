import pandas as pd
from src.train_xgboost import engineer_features

def test_engineer_features():
    dummy_data = pd.DataFrame({
        "MedInc": [3.5],
        "HouseAge": [28.0],
        "AveRooms": [5.0],
        "AveBedrms": [1.0],
        "Population": [1000.0],
        "AveOccup": [2.5],
        "Latitude": [35.0],
        "Longitude": [-119.0]
    })
    
    transformed_df = engineer_features(dummy_data)
    
    assert "RoomsPerHousehold" in transformed_df.columns
    assert "BedroomsPerRoom" in transformed_df.columns
    assert "PopulationPerHousehold" in transformed_df.columns
    
    assert transformed_df["RoomsPerHousehold"].iloc[0] == 2.0  # 5.0 / 2.5
    assert transformed_df["BedroomsPerRoom"].iloc[0] == 0.2    # 1.0 / 5.0
    assert transformed_df["PopulationPerHousehold"].iloc[0] == 400.0 # 1000 / 2.5
