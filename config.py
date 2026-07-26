"""
====================================================
Project Configuration File
TG-FEA: Trust-Gated Federated Ensemble
====================================================
"""

from pathlib import Path

# -----------------------------
# Project Root Directory
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------
# Dataset Paths
# -----------------------------
RAW_DATA_PATH = PROJECT_ROOT / "data" / "N_BaIoT"

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# -----------------------------
# Random Seed
# -----------------------------
RANDOM_SEED = 42

# -----------------------------
# Train / Validation / Test
# -----------------------------
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15