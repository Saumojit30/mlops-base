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

def train_model():
    print("Loading raw data...")
    df = pd.read_csv("data/raw/housing.csv")
    
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.columns
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_features)]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Define Hyperparameter Search Grid for v1.1 Optimization
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [15, 25, None],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2]
    }

    print("Tuning hyperparameters with GridSearchCV for v1.1...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    print("Evaluating optimized v1.1 model on test set...")
    preds = best_pipeline.predict(X_test)
    metrics = {
        "version": "v1.1",
        "best_params": grid_search.best_params_,
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }
    print(f"v1.1 Metrics: {metrics}")

    print("Saving v1.1 model and metrics...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/rf_model.joblib")
    
    with open("models/rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Done! v1.1 Model saved to models/rf_model.joblib")

if __name__ == "__main__":
    train_model()
