<div align="center">

# 🏡 California Housing Price Predictor
### Production-Grade End-to-End MLOps Pipeline & Serving System

[![CI Pipeline](https://github.com/Saumojit30/mlops-base/actions/workflows/ci.yml/badge.svg)](https://github.com/Saumojit30/mlops-base/actions/workflows/ci.yml)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue?style=flat-square)](https://huggingface.co/spaces/Jit0777/california-housing-predictor)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model%20Hub-Jit0777%2Fcalifornia--housing--model-yellow?style=flat-square)](https://huggingface.co/Jit0777/california-housing-model)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

[**Live Interactive Demo**](https://huggingface.co/spaces/Jit0777/california-housing-predictor) • [**Model Card**](MODEL_CARD.md) • [**Model Registry**](https://huggingface.co/Jit0777/california-housing-model) • [**Bug Report**](https://github.com/Saumojit30/mlops-base/issues)

</div>

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Engineering Highlights](#-key-engineering-highlights)
- [End-to-End System Architecture](#-end-to-end-system-architecture)
- [Model Progression & Benchmark](#-model-progression--benchmark)
- [Level 2 MLOps: Scalability, Drift Monitoring & Continuous Training](#-level-2-mlops-autonomous-operations)
  - [1. High-Concurrency REST Microservice (FastAPI)](#1-high-concurrency-rest-microservice-fastapi)
  - [2. Statistical Data & Concept Drift Monitoring](#2-statistical-data--concept-drift-monitoring)
  - [3. Continuous Training (CT) & Champion vs. Challenger Retraining](#3-continuous-training-ct--champion-vs-challenger-retraining)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
  - [1. Local Environment Setup](#1-local-environment-setup)
  - [2. Data Collection & Pipeline Execution](#2-data-collection--pipeline-execution)
  - [3. High-Throughput REST API (FastAPI)](#3-high-throughput-rest-api-fastapi)
  - [4. Statistical Drift Monitoring](#4-statistical-drift-monitoring)
  - [5. Automated Continuous Retraining](#5-automated-continuous-retraining)
  - [6. Automated Testing Suite](#6-automated-testing-suite)
  - [7. Interactive Web Interface (Gradio)](#7-interactive-web-interface-gradio)
- [Containerization (Docker)](#-containerization-docker)
- [Cloud Model Registry (Hugging Face Hub)](#-cloud-model-registry-hugging-face-hub)
- [Production Deployment (Hugging Face Spaces)](#-production-deployment-hugging-face-spaces)
- [MLOps Maturity Matrix](#-mlops-maturity-matrix)
- [FAQ & Engineering Insights](#-faq--engineering-insights)
- [License](#-license)

---

## 📖 Project Overview

This repository implements a **complete, enterprise-ready Machine Learning Operations (MLOps) pipeline** designed for real estate price valuation based on the California Housing dataset. 

Rather than treating model training as an isolated script, this project treats machine learning as a **reproducible, version-controlled software system**:
* **Decoupled Architecture:** Heavy binary model artifacts (`*.joblib`) are excluded from Git and stored in a remote **Hugging Face Model Hub** with release tags (`v1.2`, `v2.1`).
* **Iterative Algorithm Progression:** Tracks empirical performance jumps from baseline Bagging (**Random Forest**) to fine-tuned sequential Gradient Boosting (**XGBoost**).
* **Automated CI/CD:** GitHub Actions executes automated integration tests on every pull request and push to `main`.
* **Zero-Downtime Dynamic Serving:** A live Gradio application hosted on **Hugging Face Spaces (ZeroGPU)** streams models on-demand, enabling real-time comparative inference across architectures.

---

## ⚡ Key Engineering Highlights

| Feature | Implementation | Business / Engineering Impact |
| :--- | :--- | :--- |
| **Model Registry** | Hugging Face Model Hub (`Jit0777/california-housing-model`) | Decoupled 278 MB binary weights from Git history; enabled semantic Git release tagging (`v1.0` ➔ `v2.1`). |
| **Automated Testing** | `pytest` suite in `tests/` | Validates data schemas, feature transformations, and non-negative prediction constraints on clean runner environments. |
| **Continuous Integration** | GitHub Actions (`.github/workflows/ci.yml`) | Fast verification across dependency installs, synthetic pipeline runs, and automated test passes. |
| **Containerization** | Multi-stage `Dockerfile` + `.dockerignore` | Single-command hermetic build ensuring identical execution on Linux, macOS, Windows, and cloud containers. |
| **Production Serving** | Gradio on Hugging Face Spaces (NVIDIA A10G ZeroGPU) | Multi-model dropdown switcher, sub-second boot time via lazy loading, and automatic fallback resolution. |
| **Feature Engineering** | Domain economic ratio transforms | Engineered `RoomsPerHousehold`, `BedroomsPerRoom`, and outlier ceiling filtering, driving a **17.6% error reduction**. |

---

## 🏗️ End-to-End System Architecture

```mermaid
flowchart TD
    %% Styling
    classDef stage fill:#f8fafc,stroke:#334155,stroke-width:1px;
    classDef highlight fill:#eff6ff,stroke:#2563eb,stroke-width:2px;
    classDef success fill:#f0fdf4,stroke:#16a34a,stroke-width:2px;

    subgraph S1["1. Data Ingestion & Engineering"]
        Raw["California Housing Data\n(data_collection.py)"] --> Clean["Target Capping Clean-up\n(Filter MedHouseVal < 5.0)"]
        Clean --> FE["Domain Feature Engineering\n• RoomsPerHousehold\n• BedroomsPerRoom\n• PopulationPerHousehold"]
    end
    class S1 stage;

    subgraph S2["2. Training Pipelines & Versioning"]
        FE --> TRF["Random Forest Pipeline\n(train_random_forest.py)"]
        FE --> TXGB["XGBoost GBDT Pipeline\n(train_xgboost.py)"]
        TRF --> RFWeights["models/rf_model.joblib\n(278 MB)"]
        TXGB --> XGBWeights["models/xgb_model.joblib\n(2.27 MB)"]
        RFWeights --> UploadCLI["upload_model.py\n(CLI Auto-Tagger)"]
        XGBWeights --> UploadCLI
    end
    class S2 stage;

    subgraph S3["3. Remote Cloud Registry (Hugging Face)"]
        UploadCLI --> HFHub[("Hugging Face Model Hub\nJit0777/california-housing-model\nTags: v1.0, v1.1, v1.2, v2.0, v2.1")]
    end
    class S3 highlight;

    subgraph S4["4. Continuous Integration (GitHub Actions)"]
        CodePush["git push origin main"] --> GHA["GitHub Actions Runner\n(.github/workflows/ci.yml)"]
        GHA --> Pytest["Automated Test Suite\n• test_data.py\n• test_features.py\n• test_models.py"]
        Pytest -->|All 5 Passed| GreenBadge["CI: Passing Status Badge"]
    end
    class S4 stage;

    subgraph S5["5. Interactive Serving Layer"]
        HFHub -.->|On-Demand Stream| App["Gradio Application\n(app.py - Model Selector)"]
        App --> Space["Live Space (ZeroGPU)\nhf.space/california-housing-predictor"]
        User(("End User / Recruiter")) <-->|Real-Time Inference| Space
    end
    class S5 success;
```

---

## 📈 Model Progression & Benchmark

Every iteration was logged with metric tracking (`models/*metrics.json`) and versioned with release tags. The project advanced from a naive baseline to an optimized sequential boosting model:

```
[v1.0 Baseline RF]   RMSE: 0.5064  |  R²: 0.8042
        ↓ (GridSearchCV Hyperparameter Optimization)
[v1.1 Tuned RF]      RMSE: 0.5043  |  R²: 0.8058
        ↓ (Domain Feature Engineering: Ratio Features & Target Outlier Clean-up)
[v1.2 Engineered RF] RMSE: 0.4646  |  R²: 0.7748
        ↓ (Algorithm Pivot: Sequential Gradient Boosted Decision Trees)
[v2.0 Baseline XGB]  RMSE: 0.4208  |  R²: 0.8153
        ↓ (RandomizedSearchCV: 500 Trees, subsample=0.9, colsample=0.8)
[v2.1 Tuned XGB]     RMSE: 0.4172  |  R²: 0.8184  🏆 (17.6% Error Reduction)
```

### Detailed Evaluation Matrix

| Version | Release Tag | Architecture Family | Key Hyperparameters & Features | RMSE (Lower = Better) | MAE | $R^2$ Score | Weight Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`v1.0`** | `v1.0` | Random Forest | 50 Trees, Default Depth, Raw Features | `0.5064` | `0.3297` | `0.8042` | ~50 MB |
| **`v1.1`** | `v1.1` | Random Forest | 200 Trees, `max_depth=25`, `GridSearchCV` | `0.5043` | `0.3267` | `0.8058` | ~150 MB |
| **`v1.2`** | `v1.2` | Random Forest | 200 Trees, Cleaned Outliers, 3 Ratio Transforms | `0.4646` | `0.3116` | `0.7748`* | 278 MB |
| **`v2.0`** | `v2.0` | XGBoost Regressor | 300 Trees, `learning_rate=0.05`, `max_depth=6` | `0.4208` | `0.2857` | `0.8153` | 2.27 MB |
| **`v2.1`** | `v2.1` | **XGBoost Regressor** | **500 Trees, `subsample=0.9`, `colsample=0.8`** | **`0.4172`** 🏆 | **`0.2786`** 🏆 | **`0.8184`** 🏆 | **2.27 MB** |

*\*Note: $R^2$ in v1.2 reflects calculation on uncapped target distributions (< $500,000).*

> 📘 **Deep Architectural Breakdown:** Read our comprehensive [MODEL_CARD.md](MODEL_CARD.md) for decision logic trees, mathematical formulation of Bagging vs. Boosting, and internal node splitting criteria.

---

## 🚀 Level 2 MLOps: Autonomous Operations

Moving beyond Level 1 (automated pipeline runs and manual deployment), this repository delivers **Level 2 MLOps: Autonomous Continuous Operations**:

```mermaid
flowchart LR
    subgraph Serving["1. Scalable Serving"]
        API["FastAPI Microservice\n(src/api.py)"] --> Ingest["Async Inference Logger\n(logs/inference_logs.jsonl)"]
    end

    subgraph Drift["2. Drift Engine"]
        Ingest --> Monitor["Statistical Drift Engine\n(src/monitor_drift.py)\n• Two-Sample KS-Test\n• Population Stability Index"]
        Monitor --> DriftAlert{"Drift Detected?\n(PSI >= 0.25 | p < 0.05)"}
    end

    subgraph CT["3. Continuous Training (CT)"]
        DriftAlert -->|Yes| Retrain["Champion vs. Challenger\n(src/continuous_training.py)"]
        Retrain --> Gate{"Challenger RMSE < Champion RMSE?"}
        Gate -->|Promote| Deploy["Promote New Model\nUpdate Registry"]
        Gate -->|Reject| Keep["Retain Active Champion"]
    end
```

### 1. High-Concurrency REST Microservice (FastAPI)
* **Preloaded In-Memory Lifespan:** Zero per-request disk I/O; models are loaded once during ASGI lifespan startup.
* **Strict Pydantic Contracts:** Schema validation ensuring valid float bounds, non-null values, and automated HTTP 422 descriptive error payloads.
* **Asynchronous Batch Processing:** Dedicated `/predict/batch` endpoint delivering vectorized matrix predictions in sub-millisecond latencies.
* **Non-Blocking Background Logging:** Uses `BackgroundTasks` to stream live inference payloads and predictions to `logs/inference_logs.jsonl` without stalling the client response loop.
* **Interactive OpenAPI / Swagger Documentation:** Automatically available at `http://localhost:8000/docs`.

### 2. Statistical Data & Concept Drift Monitoring
* **Non-Parametric Two-Sample Kolmogorov-Smirnov (KS) Test:** Compares cumulative distribution functions between training baseline and incoming production batches ($p < 0.05$ flags statistical distribution shift).
* **Population Stability Index (PSI):** Quantifies magnitude of population drift across equal-frequency quantiles ($PSI \ge 0.25$ indicates critical drift).
* **Automated Audit Reports:** Emits structured machine-readable JSON summaries (`reports/drift_report.json`) and GitHub-flavored Markdown matrices (`reports/drift_report.md`).

### 3. Continuous Training (CT) & Champion vs. Challenger Retraining
* **Unbiased Arena Evaluation:** Retrains a fresh Challenger model and directly evaluates it against the active Champion on an identical held-out test split.
* **Automated Promotion Gating:** The candidate Challenger is strictly promoted if and only if:
  $$\text{RMSE}_{\text{challenger}} < \text{RMSE}_{\text{champion}} \times (1 - \text{min\_improvement})$$
* **Automated Rollback & Audit Trail:** Archives the prior champion (`models/xgb_model_previous_champion.joblib`), updates version tags in `models/xgb_metrics.json`, and records decision rationale in `reports/retraining_summary.json`.
* **Scheduled GitHub Actions Workflow:** Fully orchestrated via `.github/workflows/continuous_training.yml` running on weekly schedules and manual dispatches.

---

## 📁 Repository Structure

```text
mlops-base/
├── .github/
│   └── workflows/
│       ├── ci.yml                     # GitHub Actions CI (20 automated tests on push/PR)
│       └── continuous_training.yml    # Continuous Training & Drift Monitoring workflow
├── data/
│   └── raw/
│       └── housing.csv                # Baseline California Housing dataset
├── logs/
│   └── inference_logs.jsonl           # Production inference audit logs (JSON Lines)
├── models/
│   ├── rf_model.joblib                # Random Forest weights (v1.2) [Git-Ignored]
│   ├── rf_metrics.json                # Random Forest evaluation metrics [Git-Tracked]
│   ├── xgb_model.joblib               # Active Champion XGBoost weights (v2.1) [Git-Ignored]
│   └── xgb_metrics.json               # Champion metadata & evaluation metrics [Git-Tracked]
├── reports/
│   ├── drift_report.json              # Statistical KS/PSI drift evaluation summary
│   ├── drift_report.md                # Markdown drift audit matrix
│   ├── retraining_summary.json        # Champion vs. Challenger tournament log
│   └── retraining_summary.md          # Continuous Training markdown report
├── src/
│   ├── api.py                         # High-concurrency FastAPI microservice & logger
│   ├── continuous_training.py         # Champion vs Challenger CT pipeline & promotion gate
│   ├── data_collection.py             # Data ingestion & schema validation
│   ├── monitor_drift.py               # Two-sample KS-test & PSI drift detection engine
│   ├── train_random_forest.py         # Isolated Random Forest training & tuning pipeline
│   └── train_xgboost.py               # Isolated XGBoost GBDT training & tuning pipeline
├── tests/
│   ├── __init__.py                    # Test package initialization
│   ├── test_api.py                    # FastAPI endpoint contracts, validation & batch tests
│   ├── test_continuous_training.py    # Retraining tournament & gating logic tests
│   ├── test_data.py                   # Dataset existence and schema integrity tests
│   ├── test_drift.py                  # Statistical KS-test and PSI drift detection tests
│   ├── test_features.py               # Unit tests for mathematical ratio transforms
│   └── test_models.py                 # Hermetic model inference and non-negativity tests
├── app.py                             # Interactive Gradio Web UI with dynamic model selector
├── upload_model.py                    # CLI utility for Hugging Face uploads & git tagging
├── MODEL_CARD.md                      # Technical Model Card, Mermaid diagrams & governance
├── Dockerfile                         # Multi-target container (Gradio UI or FastAPI microservice)
├── .dockerignore                      # Container build exclusions (venv, git, cache)
├── requirements.txt                   # Pinned production & development dependencies
└── .gitignore                         # Strict Git rules preventing heavy binary commits
```

---

## 🚀 Quickstart Guide

### 1. Local Environment Setup

Clone the repository and set up an isolated virtual environment:

```bash
# Clone the repository
git clone https://github.com/Saumojit30/mlops-base.git
cd mlops-base

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate      # On Windows (PowerShell)
# source venv/bin/activate   # On Linux / macOS

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Data Collection & Pipeline Execution

Execute the modular training pipelines:

```bash
# Step 1: Ingest raw California Housing dataset
python src/data_collection.py

# Step 2: Train Random Forest pipeline (v1.2)
python src/train_random_forest.py

# Step 3: Train initial Champion XGBoost pipeline (v2.1)
python src/train_xgboost.py
```

---

### 3. High-Throughput REST API (FastAPI)

Launch the production REST microservice with Uvicorn:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

* **Swagger / OpenAPI Documentation:** Navigate to [http://localhost:8000/docs](http://localhost:8000/docs).
* **Single Prediction Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "MedInc": 3.87,
       "HouseAge": 28.0,
       "AveRooms": 5.4,
       "AveBedrms": 1.05,
       "Population": 1425.0,
       "AveOccup": 3.0,
       "Latitude": 37.88,
       "Longitude": -122.23
     }'
```
*Response:*
```json
{
  "predicted_price": 2.1485,
  "unit": "$100,000s",
  "model_version": "v2.1-champion",
  "request_id": "4020a5c4-0610-449e-bdf0-a92c4e36504a",
  "timestamp": "2026-09-18T08:50:00.000000Z"
}
```

---

### 4. Statistical Drift Monitoring

Run drift analysis comparing baseline data against incoming inference logs:

```bash
# Run drift monitoring engine (with synthetic drift generator for demonstration)
python src/monitor_drift.py --generate-drift-data
```
Outputs audit reports to `reports/drift_report.json` and `reports/drift_report.md`.

---

### 5. Automated Continuous Retraining

Run the Champion vs. Challenger retraining arena:

```bash
# Evaluate Challenger against active Champion and promote if superior
python src/continuous_training.py --min-improvement 0.0
```
Generates comparison audit reports in `reports/retraining_summary.json` and `reports/retraining_summary.md`.

---

### 6. Automated Testing Suite

Execute the complete 20-test hermetic test suite across all subsystems:

```bash
pytest -v
```

**Test Execution Summary:**
```text
tests/test_api.py::test_health_check PASSED                               [  5%]
tests/test_api.py::test_model_info PASSED                                 [ 10%]
tests/test_api.py::test_predict_endpoint PASSED                           [ 15%]
tests/test_api.py::test_predict_validation_error PASSED                   [ 20%]
tests/test_api.py::test_batch_predict_endpoint PASSED                     [ 25%]
tests/test_api.py::test_inference_logging PASSED                          [ 30%]
tests/test_api.py::test_empty_batch_validation PASSED                     [ 35%]
tests/test_continuous_training.py::test_engineer_features PASSED          [ 40%]
tests/test_continuous_training.py::test_load_and_prepare_data PASSED     [ 45%]
tests/test_continuous_training.py::test_evaluate_model PASSED             [ 50%]
tests/test_continuous_training.py::test_continuous_training_promotion_flow PASSED [ 55%]
tests/test_data.py::test_data_file_exists PASSED                          [ 60%]
tests/test_data.py::test_data_integrity PASSED                            [ 65%]
tests/test_drift.py::test_psi_identical_distributions PASSED              [ 70%]
tests/test_drift.py::test_psi_shifted_distribution PASSED                 [ 75%]
tests/test_drift.py::test_ks_test_detects_drift PASSED                    [ 80%]
tests/test_drift.py::test_generate_drift_report PASSED                    [ 85%]
tests/test_features.py::test_engineer_features PASSED                     [ 90%]
tests/test_models.py::test_models_exist PASSED                            [ 95%]
tests/test_models.py::test_model_inference PASSED                         [100%]
============================== 20 passed in 14.38s =============================
```

---

### 7. Interactive Web Interface (Gradio)

Launch the interactive Gradio interface locally:

```bash
python app.py
```
Open **`http://127.0.0.1:7860`** in your browser to test interactive multi-model valuation.

---

## 🐳 Containerization (Docker)

To run the application in an isolated, production-like Linux container:

```bash
# 1. Build the Docker image
docker build -t housing-predictor:latest .

# 2. Run the container on port 7860
docker run -d -p 7860:7860 --name housing-app housing-predictor:latest

# 3. View live container logs
docker logs -f housing-app
```

Access the application at **`http://localhost:7860`**. To stop the container:
```bash
docker stop housing-app && docker rm housing-app
```

---

## ☁️ Cloud Model Registry (Hugging Face Hub)

Model weights are decoupled from code and stored on the [Hugging Face Model Hub: Jit0777/california-housing-model](https://huggingface.co/Jit0777/california-housing-model).

### Uploading New Weights with Release Tags:
The included `upload_model.py` utility programmatically uploads model binaries and locks them with Git tags:

```bash
# Upload Random Forest weights tagged as v1.2
python upload_model.py --file_path "models/rf_model.joblib" --version "v1.2"

# Upload XGBoost weights tagged as v2.1
python upload_model.py --file_path "models/xgb_model.joblib" --version "v2.1"
```

---

## 🌐 Production Deployment (Hugging Face Spaces)

The application is deployed live on **Hugging Face Spaces** utilizing an **NVIDIA A10G ZeroGPU**:

* 🔗 **Live Application URL:** [https://huggingface.co/spaces/Jit0777/california-housing-predictor](https://huggingface.co/spaces/Jit0777/california-housing-predictor)
* 🔗 **Direct Webview Endpoint:** [https://jit0777-california-housing-predictor.hf.space](https://jit0777-california-housing-predictor.hf.space)

### Cloud Deployment Highlights:
1. **Lazy Loading:** `app.py` boots in milliseconds by deferring heavy model downloads until the first user request.
2. **ZeroGPU Decorators:** Uses `@spaces.GPU` acceleration for high-throughput inference bursts without incurring idle GPU hosting costs.
3. **Resilient Fallback:** Implements `safe_download()` to automatically fall back to the `main` branch if a specific tag is unavailable.

---

## 🛡️ MLOps Principles Implemented

```text
┌───────────────────────────┬──────────────────────────────────────────────────────────────┐
│ MLOps Pillar              │ Project Implementation                                       │
├───────────────────────────┼──────────────────────────────────────────────────────────────┤
│ Version Control           │ Code versioned in Git; weights versioned in HF Model Hub     │
│ Reproducibility           │ Seeded random states, isolated virtual environment, Docker   │
│ Continuous Integration    │ Automated pytest validation on every push & pull request     │
│ Continuous Delivery       │ Clean separation of UI serving from model weight artifacts  │
│ Documentation Integrity   │ Mathematical Model Card + complete historical changelog      │
│ Low-Latency Serving       │ On-demand lazy model streaming with zero boot bottleneck     │
└───────────────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 💡 FAQ & Engineering Insights

<details>
<summary><b>Q1: Why is the XGBoost model (2.27 MB) so much smaller than the Random Forest model (278 MB)?</b></summary>
<br>

This comes down to **Tree Depth** and how Bagging vs. Boosting works:
* **Random Forest (Bagging):** Builds independent, deeply grown decision trees (`max_depth=25`). A single tree can contain tens of thousands of split thresholds and node statistics. Multiplied across 200 deep trees, `joblib` serializes hundreds of megabytes of raw Python object arrays.
* **XGBoost (Boosting):** Builds **shallow weak learners** (`max_depth=6`). A tree of depth 6 contains a maximum of only $2^6 = 64$ leaf nodes. Stored in C++ native binary compressed format, 500 shallow trees occupy a fraction of the memory while delivering superior accuracy.
</details>

<details>
<summary><b>Q2: How does the application avoid heavy cold-start download delays?</b></summary>
<br>

In `app.py`, we implement a **Lazy Loading Pattern** (`get_model()`). When the container starts, it immediately binds the port and responds to cloud healthchecks without blocking. The models are downloaded and cached in-memory only when the user triggers their first prediction.
</details>

<details>
<summary><b>Q3: Why track metrics in Git if model weights are excluded?</b></summary>
<br>

Binary `.joblib` files bloat Git history and cause repository slowdowns. However, JSON metrics (`models/*metrics.json`) are plain text. Committing metrics files allows Git commit diffs to track performance improvements numerically across every release commit (e.g., viewing RMSE reduction directly in GitHub commit logs).
</details>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">
Developed with ❤️ by <a href="https://github.com/Saumojit30">Saumojit Santra</a> • Built for Production MLOps Standards
</div>
