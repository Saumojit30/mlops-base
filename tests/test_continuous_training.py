import os
import json
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.continuous_training import (
    engineer_features,
    load_and_prepare_data,
    evaluate_model,
    build_challenger_pipeline,
    run_continuous_training
)


@pytest.fixture
def sample_housing_df():
    """Generates synthetic housing data for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "MedInc": np.random.uniform(1.0, 10.0, n),
        "HouseAge": np.random.uniform(1, 50, n),
        "AveRooms": np.random.uniform(3.0, 8.0, n),
        "AveBedrms": np.random.uniform(1.0, 2.0, n),
        "Population": np.random.uniform(200, 3000, n),
        "AveOccup": np.random.uniform(2.0, 5.0, n),
        "Latitude": np.random.uniform(32.0, 42.0, n),
        "Longitude": np.random.uniform(-124.0, -114.0, n),
        "MedHouseVal": np.random.uniform(1.0, 4.8, n)
    })
    return df


def test_engineer_features(sample_housing_df):
    fe_df = engineer_features(sample_housing_df)
    assert "RoomsPerHousehold" in fe_df.columns
    assert "BedroomsPerRoom" in fe_df.columns
    assert "PopulationPerHousehold" in fe_df.columns
    assert np.all(fe_df["RoomsPerHousehold"] > 0)
    assert np.all(fe_df["BedroomsPerRoom"] > 0)


def test_load_and_prepare_data(tmp_path, sample_housing_df):
    baseline_path = os.path.join(tmp_path, "baseline.csv")
    # Include an outlier above 5.0 to verify ceiling capping
    df_with_outlier = sample_housing_df.copy()
    df_with_outlier.loc[0, "MedHouseVal"] = 5.2
    df_with_outlier.to_csv(baseline_path, index=False)

    X_train, X_test, y_train, y_test = load_and_prepare_data(
        baseline_path=baseline_path,
        test_size=0.2,
        random_state=42
    )

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert (y_train < 5.0).all()
    assert (y_test < 5.0).all()
    assert "RoomsPerHousehold" in X_train.columns


def test_evaluate_model(sample_housing_df):
    fe_df = engineer_features(sample_housing_df)
    X = fe_df.drop("MedHouseVal", axis=1)
    y = fe_df["MedHouseVal"]

    pipeline = build_challenger_pipeline({
        "n_estimators": 5,
        "max_depth": 3,
        "random_state": 42
    })
    pipeline.fit(X, y)

    metrics = evaluate_model(pipeline, X, y)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_continuous_training_promotion_flow(tmp_path, sample_housing_df):
    data_path = os.path.join(tmp_path, "train_data.csv")
    sample_housing_df.to_csv(data_path, index=False)

    champion_path = os.path.join(tmp_path, "champion.joblib")
    metrics_path = os.path.join(tmp_path, "metrics.json")
    reports_dir = os.path.join(tmp_path, "reports")

    # Fast parameters for test
    fast_params = {
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42
    }

    # Case 1: No previous champion -> auto-promotion
    res1 = run_continuous_training(
        baseline_path=data_path,
        champion_path=champion_path,
        metrics_path=metrics_path,
        output_dir=reports_dir,
        min_improvement=0.0,
        challenger_params=fast_params
    )
    assert res1["decision"] == "PROMOTED"
    assert res1["promoted"] is True
    assert os.path.exists(champion_path)
    assert os.path.exists(metrics_path)
    assert os.path.exists(os.path.join(reports_dir, "retraining_summary.json"))
    assert os.path.exists(os.path.join(reports_dir, "retraining_summary.md"))

    # Case 2: Challenger fails high min_improvement gate -> REJECTED
    res2 = run_continuous_training(
        baseline_path=data_path,
        champion_path=champion_path,
        metrics_path=metrics_path,
        output_dir=reports_dir,
        min_improvement=0.99,  # Impossible threshold (99% improvement)
        challenger_params=fast_params
    )
    assert res2["decision"] == "REJECTED"
    assert res2["promoted"] is False

    # Case 3: Force promote overrides threshold
    res3 = run_continuous_training(
        baseline_path=data_path,
        champion_path=champion_path,
        metrics_path=metrics_path,
        output_dir=reports_dir,
        min_improvement=0.99,
        force_promote=True,
        challenger_params=fast_params
    )
    assert res3["decision"] == "PROMOTED"
    assert res3["promoted"] is True
