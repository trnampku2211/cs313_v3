from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


@dataclass
class DBSCANOutOfSamplePredictor:
    eps: float
    core_samples: np.ndarray
    core_labels: np.ndarray
    _neighbors: NearestNeighbors | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_fitted(cls, model: DBSCAN, x_train: np.ndarray) -> "DBSCANOutOfSamplePredictor":
        indices = np.asarray(model.core_sample_indices_, dtype=int)
        return cls(
            eps=float(model.eps),
            core_samples=np.asarray(x_train)[indices].copy(),
            core_labels=np.asarray(model.labels_)[indices].astype(int).copy(),
        )

    @property
    def has_core_samples(self) -> bool:
        return len(self.core_samples) > 0

    def _index(self) -> NearestNeighbors:
        if not self.has_core_samples:
            raise ValueError("DBSCAN produced no core samples.")
        if self._neighbors is None:
            self._neighbors = NearestNeighbors(n_neighbors=1).fit(self.core_samples)
        return self._neighbors

    def nearest_core_distance(self, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(x_new, dtype=float)
        if not self.has_core_samples:
            return np.full(len(values), np.inf), np.full(len(values), -1, dtype=int)
        distances, indices = self._index().kneighbors(values)
        return distances[:, 0], indices[:, 0]

    def predict_clusters(self, x_new: np.ndarray) -> np.ndarray:
        distances, nearest = self.nearest_core_distance(x_new)
        predictions = np.full(len(distances), -1, dtype=int)
        reachable = distances <= self.eps
        if self.has_core_samples:
            predictions[reachable] = self.core_labels[nearest[reachable]]
        return predictions

    def predict_anomalies(self, x_new: np.ndarray) -> np.ndarray:
        return (self.predict_clusters(x_new) == -1).astype(int)

    def anomaly_score(self, x_new: np.ndarray) -> np.ndarray:
        distances, _ = self.nearest_core_distance(x_new)
        return distances / self.eps

    def score_is_consistent(self, x_new: np.ndarray) -> bool:
        scores = self.anomaly_score(x_new)
        if not np.all(np.isfinite(scores)):
            return False
        return bool(np.array_equal((scores > 1.0).astype(int), self.predict_anomalies(x_new)))


