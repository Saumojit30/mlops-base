import os
import json
import numpy as np
import pytest
from src.monitor_drift import (
    calculate_psi,
    perform_ks_test,
    run_drift_analysis,
    generate_synthetic_drift_data
)

def test_calculate_psi_identical():
    """Identical distributions should have a PSI close to 0 (< 0.05)."""
    np.random.seed(42)
    dist1 = np.random.normal(5.0, 1.0, 1000)
    dist2 = np.random.normal(5.0, 1.0, 1000)
    psi = calculate_psi(dist1, dist2)
    assert psi < 0.05, f"Expected near-zero PSI for identical distributions, got {psi}"

def test_calculate_psi_shifted():
    """Significantly shifted distribution should trigger a high PSI (>= 0.25)."""
    np.random.seed(42)
    baseline = np.random.normal(3.0, 1.0, 1000)
    shifted = np.random.normal(7.0, 1.0, 1000)  # Major economic shift
    psi = calculate_psi(baseline, shifted)
    assert psi >= 0.25, f"Expected high PSI for major distribution shift, got {psi}"

def test_ks_test_detection():
    """Two-sample KS test should detect distribution difference with p < 0.05."""
    np.random.seed(42)
    baseline = np.random.normal(3.0, 1.0, 500)
    shifted = np.random.normal(5.0, 1.0, 500)
    stat, p_value, drift_detected = perform_ks_test(baseline, shifted)
    assert drift_detected is True
    assert p_value < 0.05
    assert stat > 0.1

def test_run_drift_analysis_generates_reports(tmp_path):
    """Verifies end-to-end drift analysis runs and generates both JSON and Markdown reports."""
    baseline_path = "data/raw/housing.csv"
    sim_log = str(tmp_path / "sim_logs.jsonl")
    report_dir = str(tmp_path / "reports")

    # Generate synthetic drifted data
    generate_synthetic_drift_data(sim_log, num_samples=60)

    # Run analysis
    report = run_drift_analysis(
        baseline_path=baseline_path,
        incoming_path=sim_log,
        output_dir=report_dir
    )

    assert "features_analyzed" in report
    assert "MedInc" in report["features_analyzed"]
    assert os.path.exists(os.path.join(report_dir, "drift_report.json"))
    assert os.path.exists(os.path.join(report_dir, "drift_report.md"))

    with open(os.path.join(report_dir, "drift_report.json"), "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["incoming_records"] == 60
