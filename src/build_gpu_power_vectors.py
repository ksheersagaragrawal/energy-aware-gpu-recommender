"""makes GPU power-model vectors from cleaned TechPowerUp GPU specs

script loads the cleaned GPU candidate dataset, separates metadata,
model features, and prediction targets, one-hot encodes categorical GPU
features, standardizes numeric features for linear models, and saves a
model-ready vector dataset for TDP and PSU prediction

python src/build_gpu_power_vectors.py
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CLEANED_PATH = "data/cleaned/gpu_specs_cleaned.csv"
OUT_DIR = "data/vectors/gpu_power_vectors.csv"

def stand_cols(df):
    """standardizes values for a certain set of columns """
    needs_norm = ["process_nm","tmus", "rops","texture_rate","pixel_rate","direct_x","memory_mb","memory_speed_mhz","memory_bandwidth_gbs"]
    for feature in needs_norm:
        std = df[feature].std()
        if std == 0 or pd.isna(std):
            df[f"standard_{feature}"] = 0.0
        else:
            df[f"standard_{feature}"] = (df[feature] - df[feature].mean()) / std
    return df


def onehot_help(df):
    """onehot encode memory"""
    oneh = OneHotEncoder(sparse_output=False)
    cols = oneh.fit_transform(df[["memory_type_raw"]])
    encoded_col_names = oneh.get_feature_names_out(["memory_type_raw"])
    df[encoded_col_names] = cols
    return df

def load_data():
    """loads in cleaned data"""
    df = pd.read_csv(CLEANED_PATH)
    return df

def build_vectors(df):
    """get the data ready for models"""
    df = stand_cols(df)
    df = onehot_help(df)
    return df


def main():
    df = load_data()
    df = build_vectors(df)
    df.to_csv(OUT_DIR, index=False)
    print(f"\nSaved GPU power vectors to {OUT_DIR}")


if __name__ == "__main__":
    main()



