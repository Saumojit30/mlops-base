import gradio as gr
import joblib
import pandas as pd
import os
from huggingface_hub import hf_hub_download

# ==========================================
# CONFIGURATION
# ==========================================
# Set to True when you are ready to deploy!
USE_HUGGINGFACE_MODEL = False 

# Replace with your Hugging Face username and model repository name
HF_REPO_ID = "Jit0777/california-housing-model"

# The specific version (Git tag) you want to pull from Hugging Face
HF_MODEL_VERSION = "v1.0" 
# ==========================================

try:
    if USE_HUGGINGFACE_MODEL:
        print(f"Downloading model version '{HF_MODEL_VERSION}' from Hugging Face...")
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename="rf_model.joblib", 
            revision=HF_MODEL_VERSION
        )
        model = joblib.load(model_path)
        print("Hugging Face model loaded successfully!")
    else:
        print("Loading local model...")
        model = joblib.load("models/rf_model.joblib")
        print("Local model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def predict_price(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    if model is None:
        return "Model not found! Please check configuration."
    
    # Create a dataframe for the input data to match the pipeline's expected format
    input_data = pd.DataFrame({
        "MedInc": [med_inc],
        "HouseAge": [house_age],
        "AveRooms": [ave_rooms],
        "AveBedrms": [ave_bedrms],
        "Population": [population],
        "AveOccup": [ave_occup],
        "Latitude": [latitude],
        "Longitude": [longitude]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # The target in the California housing dataset is expressed in hundreds of thousands of dollars ($100,000)
    estimated_price = prediction * 100000 
    
    return f"${estimated_price:,.2f}"

# Create Gradio Interface
with gr.Blocks(title="California House Price Predictor") as demo:
    gr.Markdown("# 🏡 California House Price Predictor")
    
    if USE_HUGGINGFACE_MODEL:
        gr.Markdown(f"**Status:** Running from Hugging Face Cloud (Version: `{HF_MODEL_VERSION}`)")
    else:
        gr.Markdown("**Status:** Running Locally")
        
    gr.Markdown("Enter the characteristics of a neighborhood to predict the median house value.")
    
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
        inputs=[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=False)
