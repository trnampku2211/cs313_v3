import numpy as np

from src.models.dbscan_model import fit_dbscan


def test_out_of_sample_prediction_uses_training_cores_only():
    train = np.array([[0.0], [0.05], [0.1], [5.0], [5.05], [5.1]])
    fitted = fit_dbscan(train, eps=0.2, min_samples=2)
    new = np.array([[0.08], [5.08], [2.5]])
    clusters = fitted.predictor.predict_clusters(new)
    anomalies = fitted.predictor.predict_anomalies(new)
    assert clusters[0] >= 0
    assert clusters[1] >= 0
    assert clusters[2] == -1
    assert anomalies.tolist() == [0, 0, 1]


def test_derived_score_matches_noise_rule():
    train = np.array([[0.0], [0.05], [0.1], [1.0], [1.05], [1.1]])
    fitted = fit_dbscan(train, eps=0.2, min_samples=2)
    new = np.array([[0.15], [0.5], [1.15]])
    scores = fitted.predictor.anomaly_score(new)
    predictions = fitted.predictor.predict_anomalies(new)
    assert np.array_equal((scores > 1.0).astype(int), predictions)
    assert fitted.predictor.score_is_consistent(new)


def test_no_core_samples_are_handled_explicitly():
    fitted = fit_dbscan(np.array([[0.0], [10.0]]), eps=0.1, min_samples=2)
    new = np.array([[0.0], [4.0]])
    assert not fitted.predictor.has_core_samples
    assert fitted.predictor.predict_anomalies(new).tolist() == [1, 1]
    assert np.isinf(fitted.predictor.anomaly_score(new)).all()
    assert not fitted.predictor.score_is_consistent(new)

