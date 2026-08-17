"""
download_data.py
----------------
Fetches the Heart Disease UCI dataset and writes a clean 14-column CSV to
data/heart_disease_raw.csv.

Primary source : UCI Machine Learning Repository (Cleveland processed subset).
Fallback source: a stable GitHub mirror of the same 303-row processed file,
                 used only if the UCI host is unreachable.

Canonical schema (14 columns):
    age, sex, cp, trestbps, chol, fbs, restecg, thalach,
    exang, oldpeak, slope, ca, thal, target

The original UCI 'num' target is 0-4 (0 = no disease, 1-4 = disease of
increasing severity). We binarise it to {0, 1} for this classification task.

Usage:
    python src/download_data.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "heart_disease_raw.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

# Canonical UCI source (raw, space/comma-separated 'processed' Cleveland file).
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Stable mirror already in the canonical 14-column, header-included CSV form.
MIRROR_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/"
    "Heart-UCI-Dataset/master/heart.csv"
)

TIMEOUT = 30


def _from_uci() -> pd.DataFrame:
    """Load the raw UCI file (no header, '?' for missing) and normalise it."""
    resp = requests.get(UCI_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), header=None, names=COLUMNS, na_values="?")
    # Binarise the 0-4 severity target to presence/absence.
    df["target"] = (df["target"].astype(float) > 0).astype(int)
    return df


def _from_mirror() -> pd.DataFrame:
    """Load the pre-assembled mirror CSV (already headered and binarised)."""
    resp = requests.get(MIRROR_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df[COLUMNS]  # enforce column order
    return df


def download() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Attempting canonical UCI source:\n  {UCI_URL}")
        df = _from_uci()
        print("  -> success (UCI).")
    except Exception as exc:  # noqa: BLE001 - we intentionally fall back
        print(f"  -> UCI unreachable ({exc.__class__.__name__}). Falling back to mirror.")
        df = _from_mirror()
        print("  -> success (mirror).")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {df.shape[0]} rows x {df.shape[1]} cols to {OUT_PATH}")
    print(f"Target balance: {df['target'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not obtain dataset: {exc}", file=sys.stderr)
        sys.exit(1)
