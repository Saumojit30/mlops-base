# Project Progression & Model Architecture

This document tracks the live progression of the **California Housing Predictor** project. It serves as both a historical log of model improvements (version control) and a technical breakdown of the underlying architecture currently deployed.

---

## 📈 Model Progression & Roadmap

We track our model versions using Hugging Face Git tags and local metrics tracking. Below is the performance progression across Random Forest iterations.

| Version | Status | Model Architecture | Key Hyperparameters / Feature Changes | RMSE (Lower = Better) | MAE | $R^2$ | HF Tag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`v1.0`** | Baseline | Random Forest | 50 Trees, Default Depth | `0.5064` | `0.3297` | `0.8042` | `v1.0` |
| **`v1.1`** | Optimized | Random Forest | 200 Trees, `max_depth=25` (`GridSearchCV`) | `0.5043` | `0.3267` | `0.8058` | `v1.1` |
| **`v1.2`** | 🟢 **ACTIVE** | Random Forest | Engineered Ratios + Capped Target Removal | **`0.4646`** | **`0.3116`** | `0.7748`* | `v1.2` |

*\*Note: $R^2$ in v1.2 is calculated on uncapped targets (< $500k).*

---

## 🧠 Current Architecture: v1.2 (Feature Engineered & Cleaned)

Our currently active model (`v1.2`) incorporates **Domain Feature Engineering** alongside the tuned Random Forest architecture.

```mermaid
flowchart TD
    %% Preprocessing & Feature Engineering
    RawData[Raw Data: housing.csv\n20,640 rows] --> DataClean["Data Cleaning\nFilter out capped targets (< $500k)"]
    DataClean --> FE["Feature Engineering\n1. RoomsPerHousehold = AveRooms / AveOccup\n2. BedroomsPerRoom = AveBedrms / AveRooms\n3. PopulationPerHousehold = Population / AveOccup"]
    
    FE --> A[Engineered Matrix: 11 Features]

    %% Training Phase
    A -->|1. Bootstrap Sampling| B1[Subset 1\n~16k rows]
    A -->|1. Bootstrap Sampling| BN[Subset 200\n~16k rows]

    %% Tree Growth
    B1 --> T1[Decision Tree 1\nMax Depth: 25]
    BN --> TN[Decision Tree 200\nMax Depth: 25]

    %% Inference Phase
    subgraph "Production Inference Phase (app.py)"
        NewData[User Input via Gradio UI] --> FE_Infer[Calculate Engineered Ratios]
        FE_Infer --> Infer1[Tree 1 Prediction]
        FE_Infer --> InferN[Tree 200 Prediction]
        
        Infer1 -->|Output: $310k| Agg[Aggregation\nAverage of all 200 Trees]
        InferN -->|Output: $290k| Agg
        
        Agg --> Final[Final Predicted Price: $305k]
    end
```

### Key Breakthroughs in v1.2
1. **Engineered Ratios:** Ratios like `RoomsPerHousehold` and `BedroomsPerRoom` provided much stronger predictive signals than raw total counts.
2. **Outlier Filtering:** Removing artificial ceiling values ($500,000 cap) allowed the decision splits to model realistic economic boundaries.
3. **Massive RMSE Reduction:** Dropped root mean squared error down to **0.4646** (~$46,460 prediction margin).
