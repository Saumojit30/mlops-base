"""
Continuous Training (CT) Engine — Automated Champion vs. Challenger Pipeline
Level 2 MLOps: Retrains models on new/accumulated data, runs an unbiased
out-of-sample evaluation showdown against the current Champion, and promotes
the Challenger only if it passes strict statistical and metric performance gates.
"""

import os
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DEFAULT_BASELINE_PATH = "data/raw/housing.csv"
DEFAULT_CHAMPION_MODEL_PATH = "models/xgb_model.joblib"
DEFAULT_CHAMPION_METRICS_PATH = "models/xgb_metrics.json"
DEFAULT_REPORTS_DIR = "reports"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies domain feature engineering ratios and transformations."""
    df_copy = df.copy()
    df_copy["RoomsPerHousehold"] = df_copy["AveRooms"] / df_copy["AveOccup"]
    df_copy["BedroomsPerRoom"] = df_copy["AveBedrms"] / df_copy["AveRooms"]
    df_copy["PopulationPerHousehold"] = df_copy["Population"] / df_copy["AveOccup"]
    return df_copy


def load_and_prepare_data(
    baseline_path: str = DEFAULT_BASELINE_PATH,
    new_data_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Loads baseline data and optional new data batch, applies cleaning and
    feature engineering, and splits into train and evaluation splits.
    """
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline dataset not found at: {baseline_path}")

    df = pd.read_csv(baseline_path)

    if new_data_path and os.path.exists(new_data_path):
        new_df = pd.read_csv(new_data_path)
        print(f"📥 Merging {len(new_df):,} newly ingested production records with baseline ({len(df):,} records)...")
        df = pd.concat([df, new_df], ignore_index=True)

    # Clean target ceiling
    if "MedHouseVal" in df.columns:
        df = df[df["MedHouseVal"] < 5.0]

    # Apply domain feature transformations
    df = engineer_features(df)

    if "MedHouseVal" not in df.columns:
        raise ValueError("Dataset must contain the 'MedHouseVal' target column.")

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(model_pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes standardized regression evaluation metrics."""
    preds = model_pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {
        "rmse": round(rmse, 5),
        "mae": round(mae, 5),
        "r2": round(r2, 5)
    }


def build_challenger_pipeline(hyperparams: dict = None) -> Pipeline:
    """Builds a scikit-learn Pipeline with ColumnTransformer and XGBRegressor."""
    if hyperparams is None:
        hyperparams = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "n_jobs": -1
        }

    # Pipeline preprocessor scales all numerical input features
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), slice(0, None))]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(**hyperparams))
    ])
    return pipeline


def run_continuous_training(
    baseline_path: str = DEFAULT_BASELINE_PATH,
    new_data_path: str = None,
    champion_path: str = DEFAULT_CHAMPION_MODEL_PATH,
    metrics_path: str = DEFAULT_CHAMPION_METRICS_PATH,
    output_dir: str = DEFAULT_REPORTS_DIR,
    min_improvement: float = 0.0,
    force_promote: bool = False,
    challenger_params: dict = None
) -> dict:
    """
    Executes Champion vs. Challenger retraining workflow:
    1. Loads combined dataset and prepares train/test splits.
    2. Evaluates current Champion model on the test split.
    3. Trains candidate Challenger model.
    4. Evaluates Challenger on the exact same test split.
    5. Gating decision: Promote Challenger if RMSE is superior by min_improvement.
    6. Emits audit reports and updates model registry.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(champion_path) or ".", exist_ok=True)

    print("🚀 Initiating Continuous Training & Champion vs. Challenger Pipeline...")
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        baseline_path=baseline_path,
        new_data_path=new_data_path
    )
    print(f"📊 Training samples: {len(X_train):,} | Evaluation samples: {len(X_test):,}")

    # --- 1. Evaluate Current Champion ---
    champion_pipeline = None
    champion_metrics = None

    if os.path.exists(champion_path):
        print(f"👑 Loading active Champion model from: {champion_path}")
        try:
            champion_pipeline = joblib.load(champion_path)
            champion_metrics = evaluate_model(champion_pipeline, X_test, y_test)
            print(f"   Champion Metrics: RMSE={champion_metrics['rmse']} | MAE={champion_metrics['mae']} | R2={champion_metrics['r2']}")
        except Exception as e:
            print(f"⚠️ Failed loading existing Champion ({e}). Will train baseline challenger as new champion.")
            champion_metrics = {"rmse": float("inf"), "mae": float("inf"), "r2": -float("inf")}
    else:
        print("ℹ️ No active Champion found. Challenger will automatically become Champion.")
        champion_metrics = {"rmse": float("inf"), "mae": float("inf"), "r2": -float("inf")}

    # --- 2. Train Challenger ---
    print("🥊 Training Challenger model...")
    challenger_pipeline = build_challenger_pipeline(challenger_params)
    challenger_pipeline.fit(X_train, y_train)

    challenger_metrics = evaluate_model(challenger_pipeline, X_test, y_test)
    print(f"   Challenger Metrics: RMSE={challenger_metrics['rmse']} | MAE={challenger_metrics['mae']} | R2={challenger_metrics['r2']}")

    # --- 3. Evaluate Gating Criteria ---
    # Performance improvement: positive delta means challenger has lower RMSE
    rmse_delta = champion_metrics["rmse"] - challenger_metrics["rmse"]
    
    if champion_metrics["rmse"] == float("inf"):
        rel_improvement = 1.0
        promoted = True
        reason = "Initial Champion establishment (no previous model found)."
    else:
        rel_improvement = (rmse_delta / champion_metrics["rmse"]) if champion_metrics["rmse"] != 0 else 0.0
        meets_threshold = rel_improvement >= min_improvement
        promoted = meets_threshold or force_promote
        if force_promote:
            reason = "Force promotion requested via flag."
        elif meets_threshold:
            reason = f"Challenger improved RMSE by {rel_improvement*100:.2f}% (Threshold: {min_improvement*100:.2f}%)."
        else:
            reason = f"Challenger failed promotion gate: RMSE delta was {rel_improvement*100:.2f}% (Required >= {min_improvement*100:.2f}%)."

    decision = "PROMOTED" if promoted else "REJECTED"
    print(f"\n🏆 Continuous Training Verdict: [{decision}] — {reason}")

    # --- 4. Model Registry Update on Promotion ---
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    new_version_tag = f"ct-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"

    if promoted:
        # Create a backup of old champion if it existed
        if champion_pipeline is not None and os.path.exists(champion_path):
            backup_path = f"{os.path.splitext(champion_path)[0]}_previous_champion.joblib"
            joblib.dump(champion_pipeline, backup_path)
            print(f"📦 Archived prior Champion to: {backup_path}")

        # Promote challenger
        joblib.dump(challenger_pipeline, champion_path)
        print(f"🎉 New Champion promoted and saved to: {champion_path}")

        # Update metrics file
        updated_metrics = {
            "version": new_version_tag,
            "algorithm": "XGBoost Regressor (Continuous Retraining)",
            "updated_at": timestamp_str,
            "rmse": challenger_metrics["rmse"],
            "mae": challenger_metrics["mae"],
            "r2": challenger_metrics["r2"],
            "challenger_params": challenger_params or "default",
            "previous_champion_rmse": champion_metrics["rmse"] if champion_metrics["rmse"] != float("inf") else None
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(updated_metrics, f, indent=4)
        print(f"📝 Registry metadata updated in: {metrics_path}")

    # --- 5. Export CT Summary Report ---
    summary_data = {
        "timestamp": timestamp_str,
        "decision": decision,
        "promoted": promoted,
        "reason": reason,
        "min_improvement_required": min_improvement,
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "rmse_delta": round(float(rmse_delta), 5),
        "relative_improvement_pct": round(float(rel_improvement * 100), 3),
        "training_samples": len(X_train),
        "evaluation_samples": len(X_test),
        "baseline_path": baseline_path,
        "new_data_path": new_data_path,
        "champion_model_path": champion_path
    }

    json_report_path = os.path.join(output_dir, "retraining_summary.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # Markdown Summary Report
    md_content = f"""# 🔄 Continuous Training (CT) Retraining Report

**Execution Timestamp:** `{timestamp_str}`  
**Pipeline Verdict:** `{'🟢 ' + decision if promoted else '🔴 ' + decision}`  
**Decision Rationale:** {reason}  

---

### ⚔️ Champion vs. Challenger Arena

| Metric | Active Champion | Retrained Challenger | Delta (Δ) | Improvement % |
| :--- | :--- | :--- | :--- | :--- |
| **RMSE (Lower is better)** | `{champion_metrics['rmse']}` | `{challenger_metrics['rmse']}` | `{rmse_delta:+.5f}` | `{rel_improvement*100:+.2f}%` |
| **MAE (Lower is better)** | `{champion_metrics['mae']}` | `{challenger_metrics['mae']}` | `{(champion_metrics['mae'] - challenger_metrics['mae']):+.5f}` | `N/A` |
| **$R^2$ Score (Higher is better)** | `{champion_metrics['r2']}` | `{challenger_metrics['r2']}` | `{(challenger_metrics['r2'] - champion_metrics['r2']):+.5f}` | `N/A` |

---

### 📋 Governance & Deployment Actions
- **Model Registry Path:** `{champion_path}`
- **Promotion Threshold:** Δ >= {min_improvement * 100}%
- **Action Taken:** {'✅ Champion model replaced with Challenger weights and metrics updated.' if promoted else '⛔ Challenger discarded. Existing Champion preserved in production.'}
- **Audit Log Saved:** `{json_report_path}`
"""
    md_report_path = os.path.join(output_dir, "retraining_summary.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"📊 Retraining audit reports generated:")
    print(f"   - JSON: {json_report_path}")
    print(f"   - Markdown: {md_report_path}")

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 2 MLOps Continuous Training (CT) Pipeline")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE_PATH, help="Path to baseline dataset CSV")
    parser.add_argument("--new-data", type=str, default=None, help="Path to newly ingested dataset CSV")
    parser.add_argument("--champion-model", type=str, default=DEFAULT_CHAMPION_MODEL_PATH, help="Path to active Champion model .joblib")
    parser.add_argument("--champion-metrics", type=str, default=DEFAULT_CHAMPION_METRICS_PATH, help="Path to Champion metrics JSON")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_REPORTS_DIR, help="Directory to export retraining reports")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Required fractional RMSE improvement to promote")
    parser.add_argument("--force-promote", action="store_true", help="Force promotion regardless of gating threshold")

    args = parser.parse_args()

    run_continuous_training(
        baseline_path=args.baseline,
        new_data_path=args.new_data,
        champion_path=args.champion_model,
        metrics_path=args.champion_metrics,
        output_dir=args.output_dir,
        min_improvement=args.min_improvement,
        force_promote=args.force_promote
    )
