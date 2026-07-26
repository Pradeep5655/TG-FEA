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

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------
# Project Root
# --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_PATH

# --------------------------------------------------------
# Results Directory
# --------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------
# Display Functions
# --------------------------------------------------------
def print_header():
    """Print project header."""

    print("=" * 60)
    print(" TG-FEA DATASET ANALYSIS ")
    print("=" * 60)
    print(f"Dataset Path : {RAW_DATA_PATH}")
    print("=" * 60)


# --------------------------------------------------------
# Dataset Discovery
# --------------------------------------------------------
def get_device_folders(dataset_path: Path):
    """Return all IoT device folders."""

    devices = [
        folder
        for folder in dataset_path.iterdir()
        if folder.is_dir()
    ]

    return sorted(devices)


def get_csv_files(device_folder: Path):
    """Recursively return all CSV files."""

    return sorted(device_folder.rglob("*.csv"))


# --------------------------------------------------------
# CSV Analysis
# --------------------------------------------------------
def analyze_csv(csv_path: Path):
    """Analyze a single CSV file."""

    try:
        df = pd.read_csv(csv_path)

        # Detect device and attack category
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
            "memory_mb": round(
                df.memory_usage(deep=True).sum() / (1024 ** 2),
                2
            ),
        }

    except Exception as e:
        print(f"Error reading: {csv_path}")
        print(e)
        return None


# --------------------------------------------------------
# Statistics
# --------------------------------------------------------
def generate_statistics(summary: pd.DataFrame):
    """Generate overall dataset statistics."""

    stats = {
        "total_devices": int(summary["device"].nunique()),
        "total_csv_files": int(len(summary)),
        "total_rows": int(summary["rows"].sum()),
        "total_columns": int(summary["columns"].max()),
        "missing_values": int(summary["missing"].sum()),
        "duplicate_rows": int(summary["duplicates"].sum()),
        "total_memory_mb": round(summary["memory_mb"].sum(), 2),
    }

    return stats


# --------------------------------------------------------
# Save Reports
# --------------------------------------------------------
def save_summary(summary: pd.DataFrame):
    """Save CSV summary."""

    summary.to_csv(
        RESULTS_DIR / "dataset_summary.csv",
        index=False
    )


def save_statistics(stats):
    """Save statistics JSON."""

    with open(
        RESULTS_DIR / "dataset_statistics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            indent=4
        )


def save_text_report(stats):
    """Save text report."""

    with open(
        RESULTS_DIR / "dataset_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 60 + "\n")
        f.write("TG-FEA DATASET REPORT\n")
        f.write("=" * 60 + "\n\n")

        for key, value in stats.items():
            f.write(f"{key}: {value}\n")


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():

    print_header()

    devices = get_device_folders(RAW_DATA_PATH)

    all_stats = []

    print(f"\nScanning {len(devices)} IoT devices...\n")

    for device in devices:

        csv_files = get_csv_files(device)

        print(f"{device.name} ({len(csv_files)} CSV files)")

        for csv_file in tqdm(csv_files, leave=False):

            result = analyze_csv(csv_file)

            if result is not None:
                all_stats.append(result)

    # Create DataFrame
    summary = pd.DataFrame(all_stats)

    # Save reports
    save_summary(summary)

    stats = generate_statistics(summary)

    save_statistics(stats)

    save_text_report(stats)

    # Display preview
    print("\nFirst Five Records\n")
    print(summary.head())

    print("\nDataset Statistics\n")

    for key, value in stats.items():
        print(f"{key:<20}: {value}")

    print("\nReports saved to:")

    print(RESULTS_DIR.resolve())


# --------------------------------------------------------
# Entry Point
# --------------------------------------------------------
if __name__ == "__main__":
    main()