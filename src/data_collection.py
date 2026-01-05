import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def load_and_save_data():
    print("Fetching California Housing Data...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/housing.csv", index=False)
    print("Data successfully saved to data/raw/housing.csv")
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    load_and_save_data()
