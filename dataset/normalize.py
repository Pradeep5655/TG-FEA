import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CSV_FILE = PROCESSED_DIR / "processed_dataset.csv"

SCALER_FILE = PROCESSED_DIR / "scaler.pkl"

X_FILE = PROCESSED_DIR / "X.npy"

Y_FILE = PROCESSED_DIR / "y.npy"

def load_dataset():
    """
    Load the processed dataset.
    """

    print("Loading processed dataset...")

    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Processed dataset not found: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    print(f"Shape : {df.shape}")

    return df


def split_features_labels(df):
    """
    Split the dataframe into feature matrix X and target vector y.
    """

    feature_columns = [
        col for col in df.columns
        if col not in {"device", "attack_type", "label"}
    ]

    if not feature_columns:
        raise ValueError("No feature columns found in the processed dataset.")

    X = df[feature_columns].to_numpy(dtype=np.float32)

    if "label" in df.columns:
        y = df["label"].to_numpy(dtype=np.uint8)
    else:
        raise ValueError("The processed dataset does not contain a 'label' column.")

    return X, y


def normalize_features(X):
    """
    Normalize features using StandardScaler.
    """

    print("\nApplying StandardScaler...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = X_scaled.astype(np.float32)

    return X_scaled, scaler
  
def save_files(X, y, scaler):

    print("\nSaving files...")

    np.save(X_FILE, X)

    np.save(Y_FILE, y.astype(np.uint8))

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    print("Saved successfully.")
    
def generate_normalization_report(X, y):
    """
    Generate normalization report.
    """

    report_file = PROCESSED_DIR / "normalization_report.txt"

    with open(report_file, "w", encoding="utf-8") as f:

        f.write("=" * 60 + "\n")
        f.write("TG-FEA NORMALIZATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Normalization Method : StandardScaler\n\n")

        f.write("Input Dataset\n")
        f.write("-" * 30 + "\n")
        f.write(f"Samples           : {X.shape[0]}\n")
        f.write(f"Features          : {X.shape[1]}\n\n")

        f.write("Output Dataset\n")
        f.write("-" * 30 + "\n")
        f.write(f"Feature Shape     : {X.shape}\n")
        f.write(f"Label Shape       : {y.shape}\n\n")

        f.write("Data Types\n")
        f.write("-" * 30 + "\n")
        f.write(f"Features          : {X.dtype}\n")
        f.write(f"Labels            : {y.dtype}\n\n")

        f.write("Feature Statistics\n")
        f.write("-" * 30 + "\n")
        f.write(f"Global Mean       : {X.mean():.6f}\n")
        f.write(f"Global Std        : {X.std():.6f}\n\n")

        f.write("Generated Files\n")
        f.write("-" * 30 + "\n")
        f.write("X.npy\n")
        f.write("y.npy\n")
        f.write("scaler.pkl\n\n")

        f.write("Normalization Completed Successfully\n")

    print(f"\nNormalization report saved to:\n{report_file}")
    
def main():

    df = load_dataset()

    X, y = split_features_labels(df)

    X_scaled, scaler = normalize_features(X)

    save_files(X_scaled, y, scaler)
    
    save_files(X_scaled, y, scaler)

    generate_normalization_report(X_scaled, y)

    print("\nNormalization completed successfully.")


if __name__ == "__main__":
    main()
    
