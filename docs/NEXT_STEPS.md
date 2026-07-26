# TG-FEA Project Progress

## Project Title

Trust-Gated Byzantine-Robust Federated Hybrid Ensemble (TG-FEA) for IoT Intrusion Detection using the N-BaIoT Dataset

---

# Project Objective

The objective of this project is to develop a Federated Learning based Intrusion Detection System (IDS) for IoT devices. The system will train a hybrid deep learning model using distributed client data while remaining robust against Byzantine (malicious) clients through a Trust-Gated Aggregation mechanism.

---

# Current Progress

## Module 1: Dataset Discovery ✅

Completed Tasks

- Explored the N-BaIoT dataset.
- Identified all IoT devices.
- Counted CSV files.
- Counted samples.
- Verified feature count.
- Checked missing values.
- Checked duplicate rows.

Generated Files

- dataset_summary.csv
- dataset_statistics.json
- dataset_report.txt

---

## Module 2: Data Preprocessing ✅

Completed Tasks

- Merged all CSV files into one dataset.
- Ignored demonstrate_structure.csv.
- Added device column.
- Added attack_type column.
- Encoded attack labels.
- Generated label mapping.

Generated Files

- processed_dataset.csv
- processed_dataset.pkl
- label_mapping.json
- preprocessing_report.txt

---

## Module 3: Feature Normalization ✅

Completed Tasks

- Standardized all 115 numerical features.
- Saved normalized feature matrix.
- Saved labels separately.
- Saved fitted StandardScaler.

Generated Files

- X.npy
- y.npy
- scaler.pkl
- normalization_report.txt

---

## Module 4: Federated Client Split ✅

Completed Tasks

- Split the dataset into 9 federated clients.
- One client represents one IoT device.
- Generated metadata for every client.

Client Devices

1. Danmini_Doorbell
2. Ecobee_Thermostat
3. Ennio_Doorbell
4. Philips_B120N10_Baby_Monitor
5. Provision_PT_737E_Security_Camera
6. Provision_PT_838_Security_Camera
7. Samsung_SNH_1011_N_Webcam
8. SimpleHome_XCS7_1002_WHT_Security_Camera
9. SimpleHome_XCS7_1003_WHT_Security_Camera

Generated Files

Each client contains

- X.npy
- y.npy
- info.json

---

# Remaining Modules

- CNN Feature Extractor
- BiLSTM Model
- GRU Model
- Attention Layer
- Hybrid Model
- Federated Learning
- Byzantine Attack Simulation
- Trust-Gated Aggregation
- Experimental Evaluation

---

Current Project Completion

Approximately 45%

The complete data preparation pipeline is finished.
The remaining work focuses on model development, federated learning, experimentation, and evaluation.