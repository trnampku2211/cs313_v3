from pathlib import Path

from src.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
    model_frame,
    quality_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_source_dataset_schema_and_quality():
    frame = load_dataset(ROOT / "ai4i2020.csv")
    report = quality_report(frame)
    assert report.rows == 10_000
    assert report.columns == 14
    assert report.missing_values == 0
    assert report.exact_duplicates == 0
    assert report.positive_count == 339


def test_model_frame_excludes_identifiers_and_labels():
    frame = load_dataset(ROOT / "ai4i2020.csv")
    features, target = model_frame(frame)
    assert list(features.columns) == FEATURE_COLUMNS
    assert target.name == TARGET_COLUMN
    assert "UDI" not in features
    assert "TWF" not in features

