from pathlib import Path

import pytest

from src.data import TARGET_COLUMN, load_dataset
from src.splitting import create_stratified_split


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_split_is_disjoint_complete_and_stratified():
    frame = load_dataset(ROOT / "ai4i2020.csv")
    target = frame[TARGET_COLUMN]
    split = create_stratified_split(target, random_state=42)
    split.validate(len(frame))
    assert len(split.train) == 6_000
    assert len(split.validation) == 2_000
    assert len(split.test) == 2_000
    overall = target.mean()
    assert frame.iloc[split.train][TARGET_COLUMN].mean() == pytest.approx(overall, abs=0.001)
    assert frame.iloc[split.validation][TARGET_COLUMN].mean() == pytest.approx(overall, abs=0.001)
    assert frame.iloc[split.test][TARGET_COLUMN].mean() == pytest.approx(overall, abs=0.001)


def test_split_is_reproducible():
    frame = load_dataset(ROOT / "ai4i2020.csv")
    first = create_stratified_split(frame[TARGET_COLUMN], 42)
    second = create_stratified_split(frame[TARGET_COLUMN], 42)
    assert (first.train == second.train).all()
    assert (first.validation == second.validation).all()
    assert (first.test == second.test).all()

