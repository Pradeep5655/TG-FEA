# TG-FEA: Trust-Gated Byzantine-Robust Federated Hybrid Ensemble

## Overview

TG-FEA is a Federated Learning-based Intrusion Detection System (IDS) designed for IoT networks. The project combines a Hybrid Deep Learning model (CNN + BiLSTM + GRU + Attention) with a Trust-Gated Aggregation mechanism to improve robustness against Byzantine (malicious) clients during federated training.

The project uses the **N-BaIoT Dataset** for detecting IoT botnet attacks while preserving data privacy through Federated Learning.

---

# Objectives

- Detect IoT botnet attacks using Deep Learning.
- Train models without sharing raw client data.
- Simulate Byzantine attacks in a federated environment.
- Develop a Trust-Gated Aggregation algorithm for robust model aggregation.
- Compare the proposed method with standard Federated Learning techniques.

---

# Dataset

Dataset Used:
- N-BaIoT Dataset

Dataset Statistics

| Item | Value |
|------|------:|
| IoT Devices | 9 |
| CSV Files | 89 |
| Features | 115 |
| Total Samples | 7,062,606 |
| Attack Classes | 11 |

---

# Project Workflow

```text
Raw N-BaIoT Dataset
        │
        ▼
Dataset Discovery
        │
        ▼
CSV Inspection
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Normalization
        │
        ▼
Federated Client Split
        │
        ▼
CNN
        │
        ▼
BiLSTM
        │
        ▼
GRU
        │
        ▼
Attention
        │
        ▼
Federated Learning
        │
        ▼
Byzantine Attack Simulation
        │
        ▼
Trust-Gated Aggregation
        │
        ▼
Experimental Evaluation
```

---

# Project Structure

```text
TG-FEA/
│
├── attacks/
├── data/
├── dataset/
├── docs/
├── experiments/
├── federated/
├── figures/
├── models/
├── results/
├── utils/
├── config.py
├── requirements.txt
└── README.md
```

---

# Project Progress

- [x] Dataset Discovery
- [x] CSV Inspection
- [x] Dataset Report Generation
- [x] Data Preprocessing
- [x] Data Normalization
- [x] Client Split
- [ ] CNN Feature Extractor
- [ ] BiLSTM Model
- [ ] GRU Model
- [ ] Attention Layer
- [ ] Hybrid Model
- [ ] Federated Learning
- [ ] Byzantine Attack Simulation
- [ ] Trust-Gated Aggregation
- [ ] Experimental Evaluation

Current Progress: **~45% Complete**

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Matplotlib
- tqdm

---

# Installation

Clone the repository:

```bash
git clone <repository_url>
cd TG-FEA
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

The dataset is **not included** in this repository because of its size.

After downloading the shared project data, copy it into:

```text
TG-FEA/
└── data/
    ├── processed/
    └── federated/
```

The project expects this directory structure before running any scripts.

# License

This project is developed for academic and research purposes.