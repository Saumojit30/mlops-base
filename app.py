import gradio as gr
import joblib
import pandas as pd
import os
from huggingface_hub import hf_hub_download

# ==========================================
# CONFIGURATION
# ==========================================
HF_REPO_ID = "Jit0777/california-housing-model"
HF_MODEL_VERSION = "v2.1" 

# Auto-detect execution environment:
IS_HF_SPACE = "SPACE_ID" in os.environ
USE_HUGGINGFACE_MODEL = IS_HF_SPACE or not (
    os.path.exists("models/xgb_model.joblib") and os.path.exists("models/rf_model.joblib")
)
# ==========================================

def safe_download(filename, revision=None):
    """Safely downloads model artifact with automatic fallback to main branch if tag not found."""
    try:
        if revision:
            return hf_hub_download(repo_id=HF_REPO_ID, filename=filename, revision=revision)
    except Exception as e:
        print(f"Tag {revision} not found or failed for {filename}, falling back to default branch: {e}")
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)

# Load models
models = {}
try:
    if USE_HUGGINGFACE_MODEL:
        print(f"Connecting to Hugging Face Model Hub ({HF_REPO_ID})...")
        xgb_path = safe_download("xgb_model.joblib", revision=HF_MODEL_VERSION)
        rf_path = safe_download("rf_model.joblib", revision="v1.2")
        models["⚡ XGBoost Regressor (v2.1 - Recommended)"] = joblib.load(xgb_path)
        models["🌲 Random Forest Regressor (v1.2)"] = joblib.load(rf_path)
        print("Successfully loaded cloud models from Hugging Face Model Hub.")
    else:
        print("Loading local model weights from models/ directory...")
        if os.path.exists("models/xgb_model.joblib"):
            models["⚡ XGBoost Regressor (v2.1 - Recommended)"] = joblib.load("models/xgb_model.joblib")
        if os.path.exists("models/rf_model.joblib"):
            models["🌲 Random Forest Regressor (v1.2)"] = joblib.load("models/rf_model.joblib")
        print("Loaded local models:", list(models.keys()))
except Exception as e:
    print(f"Error loading models: {e}")

def predict_price(model_choice, med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    selected_model = models.get(model_choice)
    if selected_model is None:
        return "Selected model not loaded! Please check model configuration."
    
    # Calculate engineered ratios matching pipeline inputs
    rooms_per_household = ave_rooms / ave_occup if ave_occup != 0 else 0
    bedrooms_per_room = ave_bedrms / ave_rooms if ave_rooms != 0 else 0
    population_per_household = population / ave_occup if ave_occup != 0 else 0

    input_data = pd.DataFrame({
        "MedInc": [med_inc],
        "HouseAge": [house_age],
        "AveRooms": [ave_rooms],
        "AveBedrms": [ave_bedrms],
        "Population": [population],
        "AveOccup": [ave_occup],
        "Latitude": [latitude],
        "Longitude": [longitude],
        "RoomsPerHousehold": [rooms_per_household],
        "BedroomsPerRoom": [bedrooms_per_room],
        "PopulationPerHousehold": [population_per_household]
    })
    
    prediction = selected_model.predict(input_data)[0]
    estimated_price = prediction * 100000 
    
    return f"${estimated_price:,.2f}"

# Create Gradio Interface
with gr.Blocks(title="California House Price Predictor") as demo:
    gr.Markdown("# 🏡 California House Price Predictor — MLOps Showcase")
    
    if IS_HF_SPACE or USE_HUGGINGFACE_MODEL:
        gr.Markdown(f"**Deployment Status:** 🌐 Live on Hugging Face Spaces | **Model Hub:** `{HF_REPO_ID}`")
    else:
        gr.Markdown("**Deployment Status:** 💻 Running Locally | **Models:** Local Artifacts")
        
    gr.Markdown("Select an algorithm architecture and enter neighborhood features to predict the median house value.")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(models.keys()), 
            value=list(models.keys())[0] if models else None, 
            label="🤖 Choose Machine Learning Model Architecture"
        )

    with gr.Row():
        with gr.Column():
            med_inc = gr.Slider(0, 15, value=3.5, label="Median Income (in $10,000s)")
            house_age = gr.Slider(1, 100, value=28, label="House Age (Years)")
            ave_rooms = gr.Slider(1, 10, value=5, label="Average Rooms")
            ave_bedrms = gr.Slider(0.5, 5, value=1.1, label="Average Bedrooms")
        
        with gr.Column():
            population = gr.Number(value=1400, label="Population")
            ave_occup = gr.Slider(1, 10, value=3, label="Average Occupancy")
            latitude = gr.Number(value=35.0, label="Latitude")
            longitude = gr.Number(value=-119.0, label="Longitude")
            
    predict_btn = gr.Button("Predict Price", variant="primary")
    output = gr.Textbox(label="Estimated Median House Price", text_align="center")
    
    predict_btn.click(
        fn=predict_price,
        inputs=[model_dropdown, med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=False)
