from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def fit_lof(
    x_train: np.ndarray,
    n_neighbors: int = 20,
    metric: str = "minkowski",
) -> LocalOutlierFactor:
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        metric=metric,
        contamination="auto",
        novelty=True,
        n_jobs=-1,
    )
    return model.fit(x_train)


def anomaly_scores(model: LocalOutlierFactor, values: np.ndarray) -> np.ndarray:
    return -model.score_samples(values)

