import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def engineer_features(df):
    """Applies domain feature engineering and data cleaning for v1.2 release."""
    df_copy = df.copy()
    
    # 1. Feature Engineering
    df_copy["RoomsPerHousehold"] = df_copy["AveRooms"] / df_copy["AveOccup"]
    df_copy["BedroomsPerRoom"] = df_copy["AveBedrms"] / df_copy["AveRooms"]
    df_copy["PopulationPerHousehold"] = df_copy["Population"] / df_copy["AveOccup"]
    
    return df_copy

def train_model():
    print("Loading raw data...")
    df = pd.read_csv("data/raw/housing.csv")
    
    # Clean outlier target capping at 5.0 ($500,000) for cleaner learning
    print("Applying v1.2 data cleaning (removing 500k target ceiling capping)...")
    df = df[df["MedHouseVal"] < 5.0]

    print("Applying v1.2 feature engineering...")
    df = engineer_features(df)
    
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.columns
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_features)]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200, 
            max_depth=25, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            random_state=42, 
            n_jobs=-1
        ))
    ])

    print("Training v1.2 Random Forest model with engineered features...")
    pipeline.fit(X_train, y_train)

    print("Evaluating v1.2 model on test set...")
    preds = pipeline.predict(X_test)
    metrics = {
        "version": "v1.2",
        "features_added": ["RoomsPerHousehold", "BedroomsPerRoom", "PopulationPerHousehold"],
        "data_cleaning": "Target capped values (<5.0) removed",
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }
    print(f"v1.2 Metrics: {metrics}")

    print("Saving v1.2 model and metrics...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/rf_model.joblib")
    
    with open("models/rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Done! v1.2 Model saved to models/rf_model.joblib")

if __name__ == "__main__":
    train_model()
