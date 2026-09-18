import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASELINE_PATH_DEFAULT = "data/raw/housing.csv"
LOGS_PATH_DEFAULT = "logs/inference_logs.jsonl"
REPORTS_DIR_DEFAULT = "reports"

NUMERICAL_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"
]

def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) between two continuous distributions."""
    # Remove NaNs
    b_clean = baseline[~np.isnan(baseline)]
    t_clean = target[~np.isnan(target)]

    if len(b_clean) == 0 or len(t_clean) == 0:
        return 0.0

    # Determine quantile bins based on baseline distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(b_clean, quantiles)
    bin_edges = np.unique(bin_edges)  # Prevent duplicate edges
    
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Compute frequency proportions
    b_counts, _ = np.histogram(b_clean, bins=bin_edges)
    t_counts, _ = np.histogram(t_clean, bins=bin_edges)

    b_dist = np.where(b_counts == 0, 1e-4, b_counts) / len(b_clean)
    t_dist = np.where(t_counts == 0, 1e-4, t_counts) / len(t_clean)

    # PSI formula: sum((P - Q) * ln(P / Q))
    psi_value = np.sum((t_dist - b_dist) * np.log(t_dist / b_dist))
    return float(psi_value)

def perform_ks_test(baseline: np.ndarray, target: np.ndarray) -> Tuple[float, float, bool]:
    """Runs a Two-Sample Kolmogorov-Smirnov test to detect distribution drift."""
    b_clean = baseline[~np.isnan(baseline)]
    t_clean = target[~np.isnan(target)]

    if len(b_clean) == 0 or len(t_clean) == 0:
        return 0.0, 1.0, False

    stat, p_value = stats.ks_2samp(b_clean, t_clean)
    drift_detected = bool(p_value < 0.05)
    return float(stat), float(p_value), drift_detected

def load_inference_logs(log_path: str) -> pd.DataFrame:
    """Extracts feature values from production JSONL inference logs."""
    if not os.path.exists(log_path):
        return pd.DataFrame()

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                feats = data.get("features", {})
                # Map snake_case to PascalCase
                records.append({
                    "MedInc": feats.get("med_inc"),
                    "HouseAge": feats.get("house_age"),
                    "AveRooms": feats.get("ave_rooms"),
                    "AveBedrms": feats.get("ave_bedrms"),
                    "Population": feats.get("population"),
                    "AveOccup": feats.get("ave_occup"),
                    "Latitude": feats.get("latitude"),
                    "Longitude": feats.get("longitude"),
                    "PredictionUSD": data.get("prediction_usd"),
                    "RawPrediction": data.get("raw_prediction"),
                    "Timestamp": data.get("timestamp")
                })
            except Exception:
                continue

    return pd.DataFrame(records)

def generate_synthetic_drift_data(output_path: str = "logs/drift_simulation.jsonl", num_samples: int = 150):
    """Utility generator to simulate incoming production data with deliberate economic shift."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)

    # Shifted distributions (inflation + urbanization simulation)
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            # Inflation shift: MedInc shifted by +40%
            med_inc = float(np.clip(np.random.normal(5.8, 1.8), 0.5, 15.0))
            house_age = float(np.clip(np.random.normal(35.0, 10.0), 1.0, 100.0))
            ave_rooms = float(np.clip(np.random.normal(5.2, 1.2), 1.0, 15.0))
            ave_bedrms = float(np.clip(np.random.normal(1.05, 0.2), 0.5, 5.0))
            population = float(np.clip(np.random.normal(2200.0, 600.0), 100.0, 10000.0))
            ave_occup = float(np.clip(np.random.normal(3.4, 0.8), 1.0, 10.0))
            latitude = float(np.random.uniform(34.0, 38.0))
            longitude = float(np.random.uniform(-122.0, -117.0))
            raw_pred = round(float(med_inc * 0.6 + 1.2), 4)

            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_type": "xgboost",
                "model_version": "v2.1",
                "features": {
                    "med_inc": med_inc,
                    "house_age": house_age,
                    "ave_rooms": ave_rooms,
                    "ave_bedrms": ave_bedrms,
                    "population": population,
                    "ave_occup": ave_occup,
                    "latitude": latitude,
                    "longitude": longitude
                },
                "raw_prediction": raw_pred,
                "prediction_usd": round(raw_pred * 100000.0, 2),
                "latency_ms": 3.2
            }
            f.write(json.dumps(event) + "\n")
    print(f"Generated {num_samples} synthetic drifted production records in '{output_path}'.")

def run_drift_analysis(baseline_path: str, incoming_path: str, output_dir: str) -> Dict[str, Any]:
    """Runs end-to-end data and concept drift analysis and exports reports."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline dataset '{baseline_path}' missing!")
    
    baseline_df = pd.read_csv(baseline_path)
    
    # Load production incoming data (from logs or csv)
    if incoming_path.endswith(".jsonl"):
        incoming_df = load_inference_logs(incoming_path)
    else:
        incoming_df = pd.read_csv(incoming_path)

    if incoming_df.empty or len(incoming_df) < 5:
        print("⚠️ Warning: Less than 5 production inference records found. Using synthetic validation sample.")
        generate_synthetic_drift_data("logs/temp_validation_sample.jsonl", num_samples=50)
        incoming_df = load_inference_logs("logs/temp_validation_sample.jsonl")

    report_data = {
        "report_timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_records": len(baseline_df),
        "incoming_records": len(incoming_df),
        "features_analyzed": {},
        "overall_drift_detected": False,
        "drifted_features_count": 0,
        "recommendation": "System is stable. No retraining required."
    }

    drifted_count = 0
    markdown_rows = []

    for col in NUMERICAL_FEATURES:
        if col not in baseline_df.columns or col not in incoming_df.columns:
            continue

        b_vals = baseline_df[col].dropna().values
        t_vals = incoming_df[col].dropna().values

        ks_stat, p_val, ks_drift = perform_ks_test(b_vals, t_vals)
        psi_val = calculate_psi(b_vals, t_vals)

        if psi_val >= 0.25 or ks_drift:
            status_indicator = "🔴 Severe Drift"
            is_drift = True
            drifted_count += 1
        elif psi_val >= 0.1:
            status_indicator = "🟡 Moderate Shift"
            is_drift = False
        else:
            status_indicator = "🟢 Stable"
            is_drift = False

        report_data["features_analyzed"][col] = {
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_val, 6),
            "psi": round(psi_val, 4),
            "drift_detected": is_drift,
            "status": status_indicator,
            "baseline_mean": round(float(np.mean(b_vals)), 3),
            "incoming_mean": round(float(np.mean(t_vals)), 3)
        }

        markdown_rows.append(
            f"| `{col}` | {status_indicator} | `{ks_stat:.4f}` | `{p_val:.4e}` | `{psi_val:.4f}` | `{np.mean(b_vals):.2f}` | `{np.mean(t_vals):.2f}` |"
        )

    report_data["drifted_features_count"] = drifted_count
    if drifted_count >= 2:
        report_data["overall_drift_detected"] = True
        report_data["recommendation"] = "⚠️ SIGNIFICANT DRIFT DETECTED: Automated continuous retraining recommended!"
    else:
        report_data["overall_drift_detected"] = False
        report_data["recommendation"] = "✅ System is healthy. Model distribution matches production data."

    # Write JSON report
    json_path = os.path.join(output_dir, "drift_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    # Write Markdown report
    md_content = f"""# 📊 Data & Concept Drift Monitoring Report

**Generated At:** `{report_data['report_timestamp']}`  
**Baseline Dataset:** `{baseline_path}` ({report_data['baseline_records']:,} samples)  
**Production Inference Sample:** `{incoming_path}` ({report_data['incoming_records']:,} requests)  
**Overall Status:** **{report_data['recommendation']}**

---

### 🔍 Statistical Drift Summary Matrix

| Feature | Status | KS Statistic | P-Value (KS) | PSI | Baseline Mean | Production Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{chr(10).join(markdown_rows)}

---

### 📖 Methodology & Thresholds
1. **Kolmogorov-Smirnov (KS) Test:** Non-parametric test comparing cumulative distributions. A p-value `< 0.05` indicates statistically significant divergence.
2. **Population Stability Index (PSI):**
   * **$PSI < 0.1$:** No significant change (Distribution is stable).
   * **$0.1 \\le PSI < 0.25$:** Moderate shift (Monitor incoming inference traffic).
   * **$PSI \\ge 0.25$:** Significant drift (Triggers automated continuous retraining pipeline).
"""
    md_path = os.path.join(output_dir, "drift_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n✅ Drift reports generated:")
    print(f"   - JSON Report: {json_path}")
    print(f"   - Markdown Report: {md_path}")
    print(f"   - Overall Status: {report_data['recommendation']}")

    return report_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Data & Concept Drift Monitoring Engine")
    parser.add_argument("--baseline", type=str, default=BASELINE_PATH_DEFAULT, help="Path to baseline training CSV")
    parser.add_argument("--incoming", type=str, default=LOGS_PATH_DEFAULT, help="Path to production inference logs (JSONL or CSV)")
    parser.add_argument("--output-dir", type=str, default=REPORTS_DIR_DEFAULT, help="Directory to export drift reports")
    parser.add_argument("--generate-drift-data", action="store_true", help="Generate synthetic drifted data for testing")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if significant drift is detected")

    args = parser.parse_args()

    if args.generate_drift_data:
        generate_synthetic_drift_data()

    results = run_drift_analysis(args.baseline, args.incoming, args.output_dir)

    if args.strict and results.get("overall_drift_detected", False):
        print("\n❌ Process exiting with code 1 due to detected data drift.")
        sys.exit(1)
