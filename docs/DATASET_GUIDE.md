# Dataset Guide

This document explains every dataset used in the TG-FEA project.

---

## processed_dataset.csv

Purpose

Master processed dataset.

Contains

- 115 feature columns
- device
- attack_type
- label

Usage

Used for verification and debugging.
Not used directly for federated model training.

---

## processed_dataset.pkl

Purpose

Binary version of the processed dataset.

Usage

Loads faster than CSV.
Used only if needed for debugging or analysis.

---

## X.npy

Purpose

Normalized feature matrix.

Contains only

115 normalized features.

Shape

(samples,115)

Used only for centralized training experiments.

---

## y.npy

Purpose

Target labels.

Shape

(samples,)

Contains

Attack class labels.

---

## scaler.pkl

Purpose

Stores the fitted StandardScaler.

Used when

- preprocessing new data
- testing the trained model
- inference after deployment

---

## label_mapping.json

Maps every integer label to its corresponding attack class.

Example

0 → Benign

1 → gafgyt_combo

...

10 → mirai_udpplain

---

# Federated Dataset

Location

data/federated/

Contains

client_1

client_2

...

client_9

Each client contains

- X.npy
- y.npy
- info.json

These files are the primary training data for Federated Learning.

The federated learning process loads data from each client separately.

The original processed dataset is no longer required during model training. 