import numpy as np

from src.evaluation import evaluate_binary, optimize_f1_threshold


def test_metrics_include_confusion_counts_and_average_precision():
    truth = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    predictions = (scores >= 0.5).astype(int)
    metrics = evaluate_binary(truth, predictions, scores, 0.5)
    assert metrics.f1 == 1.0
    assert metrics.average_precision == 1.0
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0


def test_threshold_is_selected_on_f1():
    truth = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.4, 0.8])
    threshold, metrics = optimize_f1_threshold(truth, scores)
    assert threshold == 0.4
    assert metrics.f1 == 1.0

