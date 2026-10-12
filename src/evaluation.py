from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    average_precision: float | None
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    threshold: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_binary(
    y_true: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None = None,
    threshold: float | None = None,
) -> BinaryMetrics:
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    ap: float | None = None
    if scores is not None:
        score_values = np.asarray(scores, dtype=float)
        if np.all(np.isfinite(score_values)) and len(np.unique(truth)) == 2:
            ap = float(average_precision_score(truth, score_values))
    return BinaryMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        average_precision=ap,
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
        threshold=None if threshold is None else float(threshold),
    )


def optimize_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, BinaryMetrics]:
    truth = np.asarray(y_true, dtype=int)
    score_values = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(score_values)):
        raise ValueError("Threshold optimization requires finite scores.")
    precision, recall, thresholds = precision_recall_curve(truth, score_values)
    if len(thresholds) == 0:
        raise ValueError("No candidate thresholds were produced.")
    candidate_precision = precision[:-1]
    candidate_recall = recall[:-1]
    denominator = candidate_precision + candidate_recall
    f1 = np.divide(
        2 * candidate_precision * candidate_recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = max(
        range(len(thresholds)),
        key=lambda index: (
            float(f1[index]),
            float(candidate_recall[index]),
            float(candidate_precision[index]),
            float(thresholds[index]),
        ),
    )
    best_threshold = float(thresholds[best_index])
    predictions = (score_values >= best_threshold).astype(int)
    return best_threshold, evaluate_binary(truth, predictions, score_values, best_threshold)


def precision_recall_points(y_true: np.ndarray, scores: np.ndarray) -> dict[str, list[float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }

