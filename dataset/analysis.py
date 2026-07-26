"""
=========================================================
TG-FEA Research Project

Module:
    Dataset Analysis

Description:
    This module analyzes the complete N-BaIoT dataset
    before preprocessing.

Author:
    TG-FEA Research Team

=========================================================
"""
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_PATH


def print_header():
    """Print project header."""

    print("=" * 60)
    print(" TG-FEA DATASET ANALYSIS ")
    print("=" * 60)
    print(f"Dataset Path : {RAW_DATA_PATH}")
    print("=" * 60)


def get_device_folders(dataset_path: Path):
    """
    Returns all IoT device folders.
    """

    devices = [
        folder
        for folder in dataset_path.iterdir()
        if folder.is_dir()
    ]

    return sorted(devices)

def get_csv_files(device_folder: Path):
    """
    Recursively finds all CSV files inside a device folder.
    """

    csv_files = sorted(device_folder.rglob("*.csv"))

    return csv_files
  
def analyze_csv(csv_path: Path):
    """
    Analyze a single CSV file.
    """

    try:
        df = pd.read_csv(csv_path)

        # Determine device and category
        if csv_path.parent.name in ["gafgyt_attacks", "mirai_attacks"]:
            device = csv_path.parent.parent.name
            category = csv_path.parent.name
        else:
            device = csv_path.parent.name
            category = "benign_traffic"

        return {
            "device": device,
            "category": category,
            "file": csv_path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
        }

    except Exception as e:
        print(f"Error reading {csv_path}")
        print(e)
        return None
      
def main():

    print_header()

    devices = get_device_folders(RAW_DATA_PATH)

    all_stats = []

    print(f"\nScanning {len(devices)} IoT devices...\n")

    for device in devices:

        csv_files = get_csv_files(device)

        print(f"{device.name} ({len(csv_files)} CSV files)")

        for csv in tqdm(csv_files, leave=False):

            result = analyze_csv(csv)

            if result is not None:
                all_stats.append(result)

    summary = pd.DataFrame(all_stats)

    print("\n")
    print(summary.head())

    print("\n========================================")
    print(f"CSV Files Analysed : {len(summary)}")
    print("========================================")

if __name__ == "__main__":
    main()