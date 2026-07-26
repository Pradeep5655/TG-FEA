"""
=========================================================
TG-FEA Research Project

Module:
    Data Preprocessing

Description:
    Merge all N-BaIoT CSV files into a single labeled dataset.

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
# Output Directory
# --------------------------------------------------------
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------
# Get CSV Files
# --------------------------------------------------------
def get_csv_files(dataset_path: Path):
    """
    Return all dataset CSV files.
    Ignore documentation/example CSV files.
    """

    csv_files = []

    for csv_file in dataset_path.rglob("*.csv"):

        if csv_file.name == "demonstrate_structure.csv":
            continue

        csv_files.append(csv_file)

    return sorted(csv_files)


# --------------------------------------------------------
# Assign Attack Labels
# --------------------------------------------------------
def assign_label(csv_path: Path):
    """
    Determine device and attack type.
    """

    filename = csv_path.stem.lower()

    if filename == "benign_traffic":
        attack_type = "benign"
        device = csv_path.parent.name

    else:
        attack_folder = csv_path.parent.name

        if attack_folder == "gafgyt_attacks":
            attack_type = f"gafgyt_{filename}"

        elif attack_folder == "mirai_attacks":
            attack_type = f"mirai_{filename}"

        else:
            attack_type = filename

        device = csv_path.parent.parent.name

    return device, attack_type


# --------------------------------------------------------
# Load CSV
# --------------------------------------------------------
def load_csv(csv_path: Path):
    """
    Load one CSV and add metadata.
    """

    df = pd.read_csv(csv_path)

    device, attack_type = assign_label(csv_path)

    df["device"] = device
    df["attack_type"] = attack_type

    return df


# --------------------------------------------------------
# Create Numeric Labels
# --------------------------------------------------------
def create_label_mapping(df):
    """
    Create numeric labels.
    """

    attack_types = sorted(df["attack_type"].unique())

    label_mapping = {
        attack: idx
        for idx, attack in enumerate(attack_types)
    }

    df["label"] = df["attack_type"].map(label_mapping)

    return df, label_mapping


# --------------------------------------------------------
# Save Label Mapping
# --------------------------------------------------------
def save_label_mapping(label_mapping):
    """
    Save label mapping.
    """

    with open(
        PROCESSED_DIR / "label_mapping.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            label_mapping,
            f,
            indent=4
        )

def save_processed_dataset(df):
    """
    Save the merged processed dataset
    in both CSV and Pickle formats.
    """

    csv_file = PROCESSED_DIR / "processed_dataset.csv"
    pkl_file = PROCESSED_DIR / "processed_dataset.pkl"

    print("\nSaving CSV...")
    df.to_csv(csv_file, index=False)

    print("Saving Pickle...")
    df.to_pickle(pkl_file)

    print("\nProcessed dataset saved successfully.")

    print(f"CSV : {csv_file}")
    print(f"PKL : {pkl_file}")
    
def generate_preprocessing_report(df, label_mapping):
    """
    Generate preprocessing report.
    """

    report_file = PROCESSED_DIR / "preprocessing_report.txt"

    with open(report_file, "w", encoding="utf-8") as f:

        f.write("=" * 60 + "\n")
        f.write("TG-FEA PREPROCESSING REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total Samples : {len(df)}\n")
        f.write(f"Total Features: {df.shape[1]}\n")
        f.write(f"Missing Values: {df.isnull().sum().sum()}\n")
        f.write(f"Duplicate Rows: {df.duplicated().sum()}\n\n")

        f.write("Attack Labels\n")
        f.write("-" * 40 + "\n")

        for attack, label in label_mapping.items():
            count = (df["label"] == label).sum()
            f.write(f"{label:2d} : {attack:<20} {count}\n")

    print(f"\nPreprocessing report saved to:\n{report_file}")
    


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():

    csv_files = get_csv_files(RAW_DATA_PATH)

    print(f"\nFound {len(csv_files)} dataset CSV files.\n")

    assert len(csv_files) == 89, (
        f"Expected 89 CSV files, found {len(csv_files)}"
    )

    datasets = []

    for csv_file in tqdm(csv_files):

        df = load_csv(csv_file)

        datasets.append(df)

    merged_df = pd.concat(
        datasets,
        ignore_index=True
    )

    merged_df, label_mapping = create_label_mapping(merged_df)

    save_label_mapping(label_mapping)
    
    save_processed_dataset(merged_df)

    generate_preprocessing_report(
      merged_df,
      label_mapping
    )

    print("\nAttack Label Mapping\n")

    for attack, label in label_mapping.items():
        print(f"{label:2d} -> {attack}")

    print("\nMerged Dataset\n")
    print(merged_df.head())

    print("\nShape:", merged_df.shape)
    
    print("\nDataset Summary")
    print("-" * 40)

    print(f"Rows      : {merged_df.shape[0]}")
    print(f"Columns   : {merged_df.shape[1]}")
    print(f"Devices   : {merged_df['device'].nunique()}")
    print(f"Classes   : {merged_df['label'].nunique()}")


# --------------------------------------------------------
# Entry Point
# --------------------------------------------------------
if __name__ == "__main__":
    main()