import os
import pandas as pd

def test_data_file_exists():
    assert os.path.exists("data/raw/housing.csv"), "Raw dataset housing.csv missing!"

def test_data_integrity():
    df = pd.read_csv("data/raw/housing.csv")
    assert not df.empty, "Dataset is empty!"
    assert df.shape[1] == 9, f"Expected 9 columns, got {df.shape[1]}"
    assert "MedHouseVal" in df.columns, "Target column MedHouseVal missing!"
    assert df["MedHouseVal"].isnull().sum() == 0, "Target column contains null values!"
