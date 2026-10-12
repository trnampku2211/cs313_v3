from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def fit_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return model.fit(x_train, y_train)


def failure_probabilities(model: RandomForestClassifier, values: np.ndarray) -> np.ndarray:
    return model.predict_proba(values)[:, 1]
