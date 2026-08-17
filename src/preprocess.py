"""
preprocess.py
-------------
Single source of truth for data cleaning and feature transformation.

Both the training script and the serving API import from here, so the exact
same transformations are applied at train time and inference time. This is what
guarantees reproducibility and prevents train/serve skew.

Feature groups (based on the UCI Cleveland schema):
    Numeric    : age, trestbps, chol, thalach, oldpeak
    Categorical: cp, restecg, slope, thal, ca       (one-hot encoded)
    Binary     : sex, fbs, exang                     (already 0/1, passed through)

Data-quality handling:
    - The pre-assembled CSV encodes some historically-missing values as
      out-of-range codes (ca == 4, thal == 0). We treat those as missing and
      impute them, rather than letting the model learn a spurious category.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


@dataclass
class CleanReport:
    """Small summary of what cleaning did - handy for the EDA/report."""
    n_rows_in: int = 0
    n_rows_out: int = 0
    coerced_ca: int = 0
    coerced_thal: int = 0
    notes: list[str] = field(default_factory=list)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
    """
    Apply deterministic cleaning that is NOT learned from data (so it is safe
    to run before any train/test split):
      - coerce known invalid codes (ca == 4, thal == 0) to NaN for later imputation
      - drop exact duplicate rows
    Statistical imputation/scaling happens inside the fitted pipeline instead.
    """
    report = CleanReport(n_rows_in=len(df))
    df = df.copy()

    if "ca" in df:
        mask = df["ca"] == 4
        report.coerced_ca = int(mask.sum())
        df.loc[mask, "ca"] = np.nan

    if "thal" in df:
        mask = df["thal"] == 0
        report.coerced_thal = int(mask.sum())
        df.loc[mask, "thal"] = np.nan

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        report.notes.append(f"Dropped {dropped} duplicate row(s).")

    report.n_rows_out = len(df)
    report.notes.append(f"Coerced ca==4 -> NaN: {report.coerced_ca}")
    report.notes.append(f"Coerced thal==0 -> NaN: {report.coerced_thal}")
    return df, report


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    The reusable, fittable feature transformer.
    - numeric      -> median impute + standard scale
    - categorical  -> most-frequent impute + one-hot (unknown-safe)
    - binary       -> passthrough (already 0/1)
    """
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def load_clean(csv_path: str) -> tuple[pd.DataFrame, pd.Series, CleanReport]:
    """Convenience loader used by training and tests."""
    raw = pd.read_csv(csv_path)
    cleaned, report = clean_data(raw)
    X, y = split_X_y(cleaned)
    return X, y, report


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/heart_disease_raw.csv"
    X, y, rep = load_clean(path)
    print("Rows in/out:", rep.n_rows_in, "->", rep.n_rows_out)
    for n in rep.notes:
        print(" -", n)
    print("Feature matrix:", X.shape, "| target balance:", y.value_counts().to_dict())
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    print("Transformed shape:", Xt.shape)
