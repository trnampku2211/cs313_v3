from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .dbscan_predictor import DBSCANOutOfSamplePredictor


@dataclass
class DBSCANFit:
    model: DBSCAN
    predictor: DBSCANOutOfSamplePredictor
    point_roles: np.ndarray

    @property
    def cluster_count(self) -> int:
        labels = set(self.model.labels_)
        return len(labels - {-1})

    @property
    def noise_count(self) -> int:
        return int(np.sum(self.model.labels_ == -1))


def fit_dbscan(x_train: np.ndarray, eps: float, min_samples: int) -> DBSCANFit:
    model = DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=-1)
    model.fit(x_train)
    roles = np.full(len(x_train), "border", dtype=object)
    roles[model.labels_ == -1] = "noise"
    roles[np.asarray(model.core_sample_indices_, dtype=int)] = "core"
    predictor = DBSCANOutOfSamplePredictor.from_fitted(model, x_train)
    return DBSCANFit(model=model, predictor=predictor, point_roles=roles)


def k_distances(x_train: np.ndarray, k: int) -> np.ndarray:
    if k < 2 or k > len(x_train):
        raise ValueError("k must be between 2 and the number of training rows.")
    neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(x_train)
    distances, _ = neighbors.kneighbors(x_train)
    return np.sort(distances[:, -1])

