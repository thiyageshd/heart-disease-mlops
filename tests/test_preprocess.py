"""Unit tests for the data cleaning + preprocessing pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import (  # noqa: E402
    ALL_FEATURES,
    build_preprocessor,
    clean_data,
    load_clean,
    split_X_y,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "heart_disease_raw.csv"


@pytest.fixture
def raw_df():
    return pd.read_csv(DATA)


def test_dataset_exists():
    assert DATA.exists(), "Run `python src/download_data.py` first."


def test_clean_coerces_invalid_codes(raw_df):
    cleaned, report = clean_data(raw_df)
    # ca==4 and thal==0 are invalid codes -> must become NaN
    assert not (cleaned["ca"] == 4).any()
    assert not (cleaned["thal"] == 0).any()
    assert report.coerced_ca >= 0
    assert report.coerced_thal >= 0


def test_clean_drops_duplicates(raw_df):
    cleaned, _ = clean_data(raw_df)
    assert cleaned.duplicated().sum() == 0


def test_split_shapes(raw_df):
    cleaned, _ = clean_data(raw_df)
    X, y = split_X_y(cleaned)
    assert list(X.columns) == ALL_FEATURES
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})


def test_preprocessor_is_fittable_and_expands(raw_df):
    cleaned, _ = clean_data(raw_df)
    X, y = split_X_y(cleaned)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    # one-hot encoding must expand the column count beyond the raw 13
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] > len(ALL_FEATURES)
    assert not np.isnan(Xt).any(), "Imputation should remove all NaNs"


def test_preprocessor_handles_missing_values():
    # inject NaNs and confirm the pipeline imputes them away
    df = pd.DataFrame([{
        "age": 55, "sex": 1, "cp": 0, "trestbps": np.nan, "chol": 240,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0,
        "slope": 1, "ca": np.nan, "thal": 2,
    }])
    pre = build_preprocessor()
    # fit on a tiny frame with two rows so imputers have something to learn
    df2 = pd.concat([df, df.fillna(0)], ignore_index=True)
    Xt = pre.fit_transform(df2[ALL_FEATURES])
    assert not np.isnan(Xt).any()


def test_load_clean_end_to_end():
    X, y, report = load_clean(str(DATA))
    assert X.shape[0] == y.shape[0]
    assert report.n_rows_out <= report.n_rows_in
