from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ID_COLUMNS = ["UDI", "Product ID"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
EXPECTED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN] + FAILURE_MODE_COLUMNS


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    columns: int
    missing_values: int
    exact_duplicates: int
    positive_count: int
    positive_rate: float
    udi_unique: bool
    product_id_unique: bool


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the AI4I-shaped source table without mutating it."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    frame = pd.read_csv(source)
    validate_schema(frame)
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    if missing or extra:
        raise ValueError(f"Unexpected dataset schema. Missing={missing}; extra={extra}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    target_values = set(frame[TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1; found {sorted(target_values)}")
    invalid_types = set(frame["Type"].dropna().unique()) - {"L", "M", "H"}
    if invalid_types:
        raise ValueError(f"Unexpected Type values: {sorted(invalid_types)}")


def quality_report(frame: pd.DataFrame) -> DataQualityReport:
    return DataQualityReport(
        rows=len(frame),
        columns=len(frame.columns),
        missing_values=int(frame.isna().sum().sum()),
        exact_duplicates=int(frame.duplicated().sum()),
        positive_count=int(frame[TARGET_COLUMN].sum()),
        positive_rate=float(frame[TARGET_COLUMN].mean()),
        udi_unique=bool(frame["UDI"].is_unique),
        product_id_unique=bool(frame["Product ID"].is_unique),
    )


def model_frame(frame: pd.DataFrame, numeric_only: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    columns = NUMERIC_FEATURES if numeric_only else FEATURE_COLUMNS
    return frame.loc[:, columns].copy(), frame[TARGET_COLUMN].astype(int).copy()


def metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ID_COLUMNS + FAILURE_MODE_COLUMNS].copy()

