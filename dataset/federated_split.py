"""
=========================================================
TG-FEA Research Project

Module:
    Federated Data Splitting

Description:
    Split the processed N-BaIoT dataset into client-specific
    datasets for federated learning experiments.

The script reads the large processed CSV file in chunks,
 accumulates samples per device, and saves one X.npy, y.npy,
 and info.json file for each client.

Author:
    TG-FEA Research Team
=========================================================
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT as CONFIG_PROJECT_ROOT  # noqa: E402


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEDERATED_DIR = PROJECT_ROOT / "data" / "federated"
CSV_FILE = PROCESSED_DIR / "processed_dataset.csv"
CHUNK_SIZE = 100_000
FEATURE_COLUMNS = 115
DROP_COLUMNS = {"device", "attack_type", "label"}


def ensure_directories() -> None:
    """Create the processed and federated output directories if needed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEDERATED_DIR.mkdir(parents=True, exist_ok=True)


def get_client_folders() -> List[Path]:
    """Return the list of client folders inside the federated directory."""
    client_folders = sorted(FEDERATED_DIR.glob("client_*"))
    return [folder for folder in client_folders if folder.is_dir()]


def detect_devices() -> List[str]:
    """Detect all unique device names from the processed CSV file."""
    print("Loading CSV...")

    devices: set[str] = set()

    try:
        for chunk in tqdm(
            pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE),
            desc="Scanning devices",
        ):
            if "device" in chunk.columns:
                devices.update(chunk["device"].astype(str).tolist())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Processed dataset not found: {CSV_FILE}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to scan devices from {CSV_FILE}") from exc

    return sorted(devices)


def prepare_client_structure(devices: List[str]) -> Dict[str, Path]:
    """Create one folder per client and return mappings to their paths."""
    client_paths: Dict[str, Path] = {}

    for index, device in enumerate(devices, start=1):
        client_dir = FEDERATED_DIR / f"client_{index}"
        client_dir.mkdir(parents=True, exist_ok=True)
        client_paths[device] = client_dir

    return client_paths


def collect_client_data(devices: List[str], client_paths: Dict[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """Read CSV in chunks and accumulate feature/label arrays per client."""
    collected: Dict[str, Dict[str, np.ndarray]] = {}

    for device in devices:
        collected[device] = {"X": [], "y": []}

    try:
        for chunk_index, chunk in enumerate(
            tqdm(
                pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE),
                desc="Processing chunks",
            ),
            start=1,
        ):
            print(f"Processing chunk {chunk_index}...")

            if "device" not in chunk.columns or "label" not in chunk.columns:
                raise ValueError("The CSV file is missing required columns: device or label")

            for device in devices:
                device_chunk = chunk[chunk["device"].astype(str) == device]
                if device_chunk.empty:
                    continue

                feature_columns = [
                    col for col in device_chunk.columns if col not in DROP_COLUMNS
                ]

                if len(feature_columns) != FEATURE_COLUMNS:
                    raise ValueError(
                        f"Expected {FEATURE_COLUMNS} features, found {len(feature_columns)}"
                    )

                X_chunk = device_chunk[feature_columns].to_numpy(dtype=np.float32)
                y_chunk = device_chunk["label"].to_numpy(dtype=np.uint8)

                collected[device]["X"].append(X_chunk)
                collected[device]["y"].append(y_chunk)

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Processed dataset not found: {CSV_FILE}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to process chunks from {CSV_FILE}") from exc

    for device, arrays in collected.items():
        if arrays["X"]:
            arrays["X"] = np.concatenate(arrays["X"], axis=0)
            arrays["y"] = np.concatenate(arrays["y"], axis=0)
        else:
            arrays["X"] = np.empty((0, FEATURE_COLUMNS), dtype=np.float32)
            arrays["y"] = np.empty((0,), dtype=np.uint8)

    return collected


def save_client_artifacts(
    client_paths: Dict[str, Path],
    collected: Dict[str, Dict[str, np.ndarray]],
    devices: List[str],
) -> List[Dict[str, object]]:
    """Save X.npy, y.npy, and info.json for each client."""
    client_reports: List[Dict[str, object]] = []

    for client_id, device in enumerate(devices, start=1):
        client_dir = client_paths[device]
        X = collected[device]["X"]
        y = collected[device]["y"]

        print(f"Saving Client {client_id}...")

        np.save(client_dir / "X.npy", X)
        np.save(client_dir / "y.npy", y)

        info = {
            "client_id": client_id,
            "device": device,
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]) if X.ndim == 2 else 0,
            "classes": int(np.unique(y).size) if y.size > 0 else 0,
        }

        with open(client_dir / "info.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=4)

        client_reports.append(info)

    return client_reports


def write_federated_report(client_reports: List[Dict[str, object]]) -> None:
    """Generate the federated report file with client-level summary."""
    print("Generating report...")

    report_path = FEDERATED_DIR / "federated_report.txt"
    total_samples = int(sum(item["samples"] for item in client_reports))

    lines = [
        "=" * 36,
        "TG-FEA FEDERATED REPORT",
        "=" * 36,
        "",
    ]

    for item in client_reports:
        lines.extend(
            [
                f"Client {item['client_id']}",
                f"Device : {item['device']}",
                f"Samples : {item['samples']}",
                "",
            ]
        )

    lines.extend(
        [
            f"Total Clients : {len(client_reports)}",
            "",
            f"Total Samples : {total_samples}",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Main execution entry point."""
    ensure_directories()

    devices = detect_devices()
    if not devices:
        raise ValueError("No devices were detected in the processed dataset.")

    client_paths = prepare_client_structure(devices)
    collected = collect_client_data(devices, client_paths)
    client_reports = save_client_artifacts(client_paths, collected, devices)
    write_federated_report(client_reports)

    print("Done.")


if __name__ == "__main__":
    main()
