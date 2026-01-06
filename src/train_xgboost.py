import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    print("Loading raw data for XGBoost (v2.1 Hyperparameter Tuning)...")
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

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, n_jobs=-1))
    ])

    # Search space for v2.1 XGBoost Hyperparameter Optimization
    param_distributions = {
        'regressor__n_estimators': [300, 500],
        'regressor__learning_rate': [0.03, 0.05, 0.08],
        'regressor__max_depth': [5, 6, 8],
        'regressor__subsample': [0.7, 0.8, 0.9],
        'regressor__colsample_bytree': [0.7, 0.8, 0.9],
        'regressor__gamma': [0, 0.1]
    }

    print("Running RandomizedSearchCV to optimize XGBoost hyperparameters...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    print(f"Best XGBoost Hyperparameters: {search.best_params_}")

    print("Evaluating v2.1 optimized XGBoost model on test set...")
    preds = best_pipeline.predict(X_test)
    metrics = {
        "version": "v2.1",
        "algorithm": "XGBoost Regressor (Tuned)",
        "best_params": search.best_params_,
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }
    print(f"v2.1 XGBoost Metrics: {metrics}")

    print("Saving v2.1 XGBoost model and metrics...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/xgb_model.joblib")
    
    with open("models/xgb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Done! v2.1 Model saved to models/xgb_model.joblib")

if __name__ == "__main__":
    train_model()
