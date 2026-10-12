from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(
    x_train: np.ndarray,
    n_estimators: int = 300,
    max_samples: str | int = "auto",
    random_state: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    return model.fit(x_train)


def anomaly_scores(model: IsolationForest, values: np.ndarray) -> np.ndarray:
    return -model.score_samples(values)

