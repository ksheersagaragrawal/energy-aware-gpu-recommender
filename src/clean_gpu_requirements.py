"""Clean gpu_specs.csv for the energy-aware GPU recommender pipeline.

The script filters to desktop discrete single-GPU cards, standardizes hardware
spec units, renames columns to the shared naming, removes rows missing required
fields, and writes the cleaned candidate GPU dataset.

Command: python src/clean_gpu_requirements.py
"""
import pandas as pd
import numpy as np
import regex as reg
import argparse

RAW_PATH = "data/raw/gpu_specs.csv"
OUTPUT_PATH = "data/cleaned/gpu_specs_cleaned.csv"

def load_data():
    """ loads in the csv
    """
    df = pd.read_csv(RAW_PATH)
    return df

def clean(df, include_missing_target=False):
    """ does the cleaning via calling in helper functions
    """
    df = pd.read_csv("data/raw/gpu_specs.csv")
    df = filter_gpu_rows(df)
    columns_to_keep = ["Brand", "Name","Graphics Card__Production", "Graphics Processor__Process Size","Render Config__TMUs", "Render Config__ROPs", "Theoretical Performance__Texture Rate", "Theoretical Performance__Pixel Rate", "Graphics Features__DirectX", 
                       "Memory__Memory Size", "Clock Speeds__Memory Clock", "Memory__Memory Type", "Memory__Bandwidth",
                       "Board Design__Suggested PSU", "Board Design__TDP"]
    renaming = {"Brand": "brand", "Name": "name", "Graphics Card__Production": "production_status","Graphics Processor__Process Size": "process_nm","Render Config__TMUs": "tmus",  "Render Config__ROPs":"rops", "Theoretical Performance__Texture Rate":"texture_rate",
                "Theoretical Performance__Pixel Rate": "pixel_rate", "Graphics Features__DirectX":"direct_x", 
                 "Memory__Memory Size":"memory_mb", "Clock Speeds__Memory Clock":"memory_speed_mhz", "Memory__Memory Type": "memory_type_raw",
                  "Memory__Bandwidth":"memory_bandwidth_gbs", "Board Design__Suggested PSU": "psu_w",
                   "Board Design__TDP": "tdp_w" }
    df = df[columns_to_keep].rename(columns=renaming)
    df["process_nm"] = df["process_nm"].apply(process_helper)
    df["tmus"] = df["tmus"].apply(make_number)
    df["rops"] = df["rops"].apply(make_number)
    df["texture_rate"] = df["texture_rate"].apply(units_to_GTex)
    df["pixel_rate"] = df["pixel_rate"].apply(units_to_GPix)
    df["direct_x"] = df["direct_x"].apply(process_helper)
    df["memory_mb"] = df["memory_mb"].apply(units_to_bytes)
    df["memory_speed_mhz"] = df["memory_speed_mhz"].apply(process_helper)
    df["memory_bandwidth_gbs"] = df["memory_bandwidth_gbs"].apply(units_to_gbs)
    df["psu_w"] = df["psu_w"].apply(process_helper)
    df["tdp_w"] = df["tdp_w"].apply(process_helper)

    
    if include_missing_target:
        subset_cols = [ "process_nm","tmus", "rops", "texture_rate", "pixel_rate","direct_x", "memory_mb", "memory_bandwidth_gbs"]
    else:
        subset_cols = [ "process_nm","tmus", "rops", "texture_rate", "pixel_rate","direct_x", "memory_mb", "memory_bandwidth_gbs","tdp_w","psu_w"]
    df = df.dropna(subset=subset_cols)
    
    return df

def filter_gpu_rows(df):
    """remove multi-GPU, mobile, integrated, and non-graphics-card rows"""
    is_multi_gpu = (
        df["Top__TMUS"].apply(is_multi)
        | df["Top__ROPS"].apply(is_multi)
    )
    is_desktop_gpu = (
        df["Mobile Graphics__Release Date"].isna()
        & df["Integrated Graphics__Release Date"].isna()
    )
    is_graphics_card = df["Graphics Card__Production"].notna()
    keep_rows = (
        ~is_multi_gpu
        & is_desktop_gpu
        & is_graphics_card
    )
    return df[keep_rows].copy()

def make_number(value):
    """get rid of strings and Nans"""
    try:
        return float(value)
    except:
        return np.nan
    

def process_helper(num):
    """ strips units and standardizes formatting
    """
    if pd.isna(num):
        return np.nan

    letters = str(num).replace(",", "").strip().lower()
    if letters in {"", "unknown", "nan", "none", "system shared", "system dependent"}:
        return np.nan
    
    numbers = reg.findall(r"[-+]?\d*\.?\d+", letters)

    if len(numbers) == 0:
        return np.nan

    return float(numbers[0])


def is_multi(value):
    """ helper to find if a GPU is a multi GPU
    """
    if pd.isna(value):
        return False
    letters = str(value).replace(",", "").strip().lower()
    return bool(reg.search(r"\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?", letters))


def units_to_bytes(value):
    """ other units to mb """
    number = process_helper(value)
    if pd.isna(number):
        return np.nan

    if "gb" in str(value).lower():
        return number * 1024
    if "mb" in str(value).lower():
        return number
    if "kb" in str(value).lower():
        return number / 1024
    return number


def units_to_gbs(value):
    """other units to gbs"""
    number = process_helper(value)
    if pd.isna(number):
        return np.nan
    if "tb/s" in str(value).lower():
        return number * 1000
    if "gb/s" in str(value).lower():
        return number
    if "mb/s" in str(value).lower():
        return number / 1000
    return number


def units_to_GTex(value):
    """other units to GTexel/s."""
    number = process_helper(value)
    if pd.isna(number):
        return np.nan
    if "mtexel/s" in str(value).lower():
        return number / 1000
    if "gtexel/s" in str(value).lower():
        return number
    return number


def units_to_GPix(value):
    """other units to GPixel/s."""
    number = process_helper(value)
    if pd.isna(number):
        return np.nan
    if "mpixel/s" in str(value).lower():
        return number / 1000
    if "gpixel/s" in str(value).lower():
        return number
    return number

def main(output_path=OUTPUT_PATH, include_missing_target=False):
    df = load_data()
    df = clean(df, include_missing_target=include_missing_target)
    df.to_csv(output_path, index=False)
    print(f"Cleaned samples: {df.shape}")
    print(f"\nSaved to {output_path}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="params neeeded if you want to do a nonstandard run")
    parser.add_argument( "--output-path", default=OUTPUT_PATH, help=f"where to save the cleaned dataset. default: {OUTPUT_PATH}",)
    parser.add_argument( "--include-missing-target", action="store_true", help="Keep rows even that are missing the target columns",)
    args = parser.parse_args()
    main( output_path=args.output_path,include_missing_target=args.include_missing_target)
