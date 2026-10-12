from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass
class FittedPreprocessor:
    transformer: ColumnTransformer
    feature_names: list[str]
    numeric_only: bool = False

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.transformer.transform(frame), dtype=float)


def build_preprocessor(numeric_only: bool = False) -> ColumnTransformer:
    transformers: list[tuple] = [
        ("numeric", Pipeline([("scaler", StandardScaler())]), NUMERIC_FEATURES)
    ]
    if not numeric_only:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        )
                    ]
                ),
                CATEGORICAL_FEATURES,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_preprocessor(frame: pd.DataFrame, numeric_only: bool = False) -> FittedPreprocessor:
    transformer = build_preprocessor(numeric_only=numeric_only)
    transformer.fit(frame)
    names = [str(name) for name in transformer.get_feature_names_out()]
    return FittedPreprocessor(transformer, names, numeric_only)

