from pathlib import Path

import numpy as np

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
from src.preprocessing import fit_preprocessor
from src.splitting import create_stratified_split


ROOT = Path(__file__).resolve().parents[1]


def test_primary_preprocessor_produces_five_scaled_and_three_one_hot_features():
    frame = load_dataset(ROOT / "ai4i2020.csv")
    split = create_stratified_split(frame[TARGET_COLUMN], 42)
    fitted = fit_preprocessor(frame.iloc[split.train][FEATURE_COLUMNS])
    transformed = fitted.transform(frame.iloc[split.validation][FEATURE_COLUMNS])
    assert transformed.shape == (2_000, 8)
    assert np.isfinite(transformed).all()
    assert len(fitted.feature_names) == 8


def test_numeric_only_sensitivity_preprocessor():
    frame = load_dataset(ROOT / "ai4i2020.csv")
    fitted = fit_preprocessor(frame[FEATURE_COLUMNS], numeric_only=True)
    transformed = fitted.transform(frame.iloc[:5][FEATURE_COLUMNS])
    assert transformed.shape == (5, 5)

