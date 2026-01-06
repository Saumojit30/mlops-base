import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def engineer_features(df):
    """Applies domain feature engineering and data cleaning."""
    df_copy = df.copy()
    
    # Feature Engineering Ratios
    df_copy["RoomsPerHousehold"] = df_copy["AveRooms"] / df_copy["AveOccup"]
    df_copy["BedroomsPerRoom"] = df_copy["AveBedrms"] / df_copy["AveRooms"]
    df_copy["PopulationPerHousehold"] = df_copy["Population"] / df_copy["AveOccup"]
    
    return df_copy

def train_model():
    print("Loading raw data for XGBoost (v2.0)...")
    df = pd.read_csv("data/raw/housing.csv")
    
    print("Applying target cleaning (removing 500k ceiling capping)...")
    df = df[df["MedHouseVal"] < 5.0]

    print("Applying feature engineering for XGBoost...")
    df = engineer_features(df)
    
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.columns
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_features)]
    )

    # Define XGBoost Regressor Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Training XGBoost Regressor model (v2.0)...")
    pipeline.fit(X_train, y_train)

    print("Evaluating v2.0 XGBoost model on test set...")
    preds = pipeline.predict(X_test)
    metrics = {
        "version": "v2.0",
        "algorithm": "XGBoost Regressor",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }
    print(f"v2.0 XGBoost Metrics: {metrics}")

    print("Saving v2.0 XGBoost model and metrics...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/xgb_model.joblib")
    
    with open("models/xgb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Done! v2.0 Model saved to models/xgb_model.joblib")

if __name__ == "__main__":
    train_model()
