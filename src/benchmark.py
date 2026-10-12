from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import ParameterSampler

from .artifacts import save_joblib, save_json
from .data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
from .evaluation import BinaryMetrics, evaluate_binary, optimize_f1_threshold, precision_recall_points
from .experiments import append_result
from .models.dbscan_model import DBSCANFit, fit_dbscan, k_distances
from .models.isolation_forest_model import anomaly_scores as if_scores
from .models.isolation_forest_model import fit_isolation_forest
from .models.lof_model import anomaly_scores as lof_scores
from .models.lof_model import fit_lof
from .models.random_forest_model import failure_probabilities, fit_random_forest
from .preprocessing import FittedPreprocessor, fit_preprocessor
from .splitting import SplitIndices, create_stratified_split, save_split


@dataclass
class BenchmarkPaths:
    root: Path

    @property
    def split(self) -> Path:
        return self.root / "artifacts" / "splits" / "official_split.json"

    @property
    def bundle(self) -> Path:
        return self.root / "artifacts" / "benchmark" / "official_benchmark.joblib"

    @property
    def results_json(self) -> Path:
        return self.root / "results" / "benchmark_results.json"

    @property
    def experiments_csv(self) -> Path:
        return self.root / "results" / "experiments.csv"


def _selection_key(metrics: BinaryMetrics) -> tuple[float, float, float]:
    return metrics.f1, metrics.recall, metrics.precision


def _timed_fit(function: Callable, *args, **kwargs) -> tuple[Any, float]:
    started = perf_counter()
    value = function(*args, **kwargs)
    return value, perf_counter() - started


def _result(
    name: str,
    learning_type: str,
    parameters: dict,
    validation_metrics: BinaryMetrics,
    test_metrics: BinaryMetrics,
    fit_time: float,
    inference_time: float,
    score_source: str,
    test_pr_curve: dict[str, list[float]] | None,
) -> dict:
    return {
        "run_id": uuid4().hex,
        "model": name,
        "learning_type": learning_type,
        "parameters": parameters,
        "validation_metrics": validation_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
        "fit_time_seconds": float(fit_time),
        "inference_time_seconds": float(inference_time),
        "score_source": score_source,
        "test_pr_curve": test_pr_curve,
    }


def _compact_pr_curve(y_true: np.ndarray, scores: np.ndarray, max_points: int = 500) -> dict[str, list[float]]:
    curve = precision_recall_points(y_true, scores)
    point_count = len(curve["precision"])
    if point_count <= max_points:
        return curve
    indices = np.unique(np.linspace(0, point_count - 1, max_points, dtype=int))
    return {
        "precision": [curve["precision"][index] for index in indices],
        "recall": [curve["recall"][index] for index in indices],
        "thresholds": [],
    }


def _select_dbscan(
    x_train: np.ndarray,
    y_validation: np.ndarray,
    x_validation: np.ndarray,
    min_samples_values: list[int],
    quantiles: list[float],
) -> tuple[DBSCANFit, dict, BinaryMetrics, float]:
    best: tuple[tuple, DBSCANFit, dict, BinaryMetrics, float] | None = None
    for min_samples in min_samples_values:
        distances = k_distances(x_train, min_samples)
        eps_values = sorted(set(float(np.quantile(distances, q)) for q in quantiles))
        for eps in eps_values:
            fitted, fit_time = _timed_fit(fit_dbscan, x_train, eps, min_samples)
            if not fitted.predictor.has_core_samples or fitted.cluster_count == 0:
                continue
            predictions = fitted.predictor.predict_anomalies(x_validation)
            scores = fitted.predictor.anomaly_score(x_validation)
            consistent = fitted.predictor.score_is_consistent(x_validation)
            metrics = evaluate_binary(
                y_validation,
                predictions,
                scores if consistent else None,
                threshold=1.0,
            )
            params = {
                "eps": eps,
                "min_samples": min_samples,
                "cluster_count": fitted.cluster_count,
                "training_noise_ratio": fitted.noise_count / len(x_train),
                "derived_score_consistent": consistent,
            }
            key = (*_selection_key(metrics), -abs(params["training_noise_ratio"] - 0.0339))
            if best is None or key > best[0]:
                best = (key, fitted, params, metrics, fit_time)
    if best is None:
        raise RuntimeError("No non-degenerate DBSCAN configuration was found.")
    return best[1], best[2], best[3], best[4]


def _select_isolation_forest(
    x_train: np.ndarray,
    y_validation: np.ndarray,
    x_validation: np.ndarray,
    estimators: list[int],
    max_samples_values: list[str | int],
    random_state: int,
):
    best = None
    for n_estimators in estimators:
        for max_samples in max_samples_values:
            model, fit_time = _timed_fit(
                fit_isolation_forest,
                x_train,
                n_estimators,
                max_samples,
                random_state,
            )
            scores = if_scores(model, x_validation)
            threshold, metrics = optimize_f1_threshold(y_validation, scores)
            params = {"n_estimators": n_estimators, "max_samples": max_samples}
            key = _selection_key(metrics)
            if best is None or key > best[0]:
                best = (key, model, params, threshold, metrics, fit_time)
    return best[1:]


def _select_lof(
    x_train: np.ndarray,
    y_validation: np.ndarray,
    x_validation: np.ndarray,
    neighbors: list[int],
    metrics_to_try: list[str],
):
    best = None
    for n_neighbors in neighbors:
        for metric in metrics_to_try:
            model, fit_time = _timed_fit(fit_lof, x_train, n_neighbors, metric)
            scores = lof_scores(model, x_validation)
            threshold, metrics = optimize_f1_threshold(y_validation, scores)
            params = {"n_neighbors": n_neighbors, "metric": metric, "novelty": True}
            key = _selection_key(metrics)
            if best is None or key > best[0]:
                best = (key, model, params, threshold, metrics, fit_time)
    return best[1:]


def _select_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    x_validation: np.ndarray,
    config: dict,
    random_state: int,
    iterations: int,
):
    space = {
        "n_estimators": config["random_forest_estimators"],
        "max_depth": config["random_forest_depths"],
        "min_samples_split": config["random_forest_min_samples_split"],
        "min_samples_leaf": config["random_forest_min_samples_leaf"],
        "class_weight": config["random_forest_class_weight"],
    }
    candidates = ParameterSampler(space, n_iter=iterations, random_state=random_state)
    best = None
    for params in candidates:
        model, fit_time = _timed_fit(
            fit_random_forest, x_train, y_train, random_state=random_state, **params
        )
        scores = failure_probabilities(model, x_validation)
        threshold, metrics = optimize_f1_threshold(y_validation, scores)
        key = _selection_key(metrics)
        if best is None or key > best[0]:
            best = (key, model, params, threshold, metrics, fit_time)
    return best[1:]


def run_official_benchmark(root: str | Path, config: dict, quick: bool = False) -> dict:
    root_path = Path(root).resolve()
    paths = BenchmarkPaths(root_path)
    data_path = root_path / config["data"]["path"]
    frame = load_dataset(data_path)
    target = frame[TARGET_COLUMN].astype(int)
    split = create_stratified_split(target, config["project"]["random_state"])
    split_path = (
        root_path / "artifacts" / "smoke" / "quick_split.json" if quick else paths.split
    )
    save_split(split, split_path)

    preprocessor = fit_preprocessor(frame.iloc[split.train][FEATURE_COLUMNS])
    x_train = preprocessor.transform(frame.iloc[split.train][FEATURE_COLUMNS])
    x_validation = preprocessor.transform(frame.iloc[split.validation][FEATURE_COLUMNS])
    x_test = preprocessor.transform(frame.iloc[split.test][FEATURE_COLUMNS])
    y_train = target.iloc[split.train].to_numpy()
    y_validation = target.iloc[split.validation].to_numpy()
    y_test = target.iloc[split.test].to_numpy()
    if quick:
        # Quick mode validates integration without creating or replacing official test artifacts.
        x_test = x_validation
        y_test = y_validation
    settings = config["benchmark"]

    models: dict[str, Any] = {}
    results: list[dict] = []

    db_min = [10, 20] if quick else settings["dbscan_min_samples"]
    db_quantiles = [0.90, 0.95, 0.98] if quick else settings["dbscan_k_distance_quantiles"]
    db_fit, db_params, db_validation, db_fit_time = _select_dbscan(
        x_train, y_validation, x_validation, db_min, db_quantiles
    )
    db_params["feature_configuration"] = "Five numerical features + one-hot Type"
    started = perf_counter()
    db_predictions = db_fit.predictor.predict_anomalies(x_test)
    db_scores = db_fit.predictor.anomaly_score(x_test)
    db_consistent = db_fit.predictor.score_is_consistent(x_test)
    db_inference = perf_counter() - started
    db_test = evaluate_binary(
        y_test, db_predictions, db_scores if db_consistent else None, threshold=1.0
    )
    db_params["derived_score_consistent_on_test"] = db_consistent
    results.append(
        _result(
            "DBSCAN",
            "Unsupervised (label-guided validation)",
            db_params,
            db_validation,
            db_test,
            db_fit_time,
            db_inference,
            "Derived nearest-core distance" if db_consistent else "N/A",
            _compact_pr_curve(y_test, db_scores) if db_consistent else None,
        )
    )
    models["DBSCAN"] = {"fit": db_fit, "threshold": 1.0}

    numeric_preprocessor = fit_preprocessor(
        frame.iloc[split.train][FEATURE_COLUMNS], numeric_only=True
    )
    x_train_numeric = numeric_preprocessor.transform(frame.iloc[split.train][FEATURE_COLUMNS])
    x_validation_numeric = numeric_preprocessor.transform(
        frame.iloc[split.validation][FEATURE_COLUMNS]
    )
    x_evaluation_numeric = numeric_preprocessor.transform(
        frame.iloc[split.validation if quick else split.test][FEATURE_COLUMNS]
    )
    sensitivity_fit, sensitivity_params, sensitivity_validation, sensitivity_fit_time = _select_dbscan(
        x_train_numeric,
        y_validation,
        x_validation_numeric,
        db_min,
        db_quantiles,
    )
    sensitivity_params["feature_configuration"] = "Five numerical features only"
    started = perf_counter()
    sensitivity_predictions = sensitivity_fit.predictor.predict_anomalies(x_evaluation_numeric)
    sensitivity_scores = sensitivity_fit.predictor.anomaly_score(x_evaluation_numeric)
    sensitivity_consistent = sensitivity_fit.predictor.score_is_consistent(x_evaluation_numeric)
    sensitivity_inference = perf_counter() - started
    sensitivity_test = evaluate_binary(
        y_test,
        sensitivity_predictions,
        sensitivity_scores if sensitivity_consistent else None,
        threshold=1.0,
    )
    sensitivity_params["derived_score_consistent_on_test"] = sensitivity_consistent
    sensitivity_result = _result(
        "DBSCAN — numeric-only sensitivity",
        "Unsupervised (label-guided validation)",
        sensitivity_params,
        sensitivity_validation,
        sensitivity_test,
        sensitivity_fit_time,
        sensitivity_inference,
        "Derived nearest-core distance" if sensitivity_consistent else "N/A",
        _compact_pr_curve(y_test, sensitivity_scores) if sensitivity_consistent else None,
    )

    if_estimators = [200] if quick else settings["isolation_forest_estimators"]
    if_max_samples = ["auto", 256] if quick else settings["isolation_forest_max_samples"]
    if_model, if_params, if_threshold, if_validation, if_fit_time = _select_isolation_forest(
        x_train,
        y_validation,
        x_validation,
        if_estimators,
        if_max_samples,
        config["project"]["random_state"],
    )
    started = perf_counter()
    scores = if_scores(if_model, x_test)
    predictions = (scores >= if_threshold).astype(int)
    inference = perf_counter() - started
    test_metrics = evaluate_binary(y_test, predictions, scores, if_threshold)
    results.append(
        _result(
            "Isolation Forest",
            "Unsupervised (label-guided validation)",
            if_params,
            if_validation,
            test_metrics,
            if_fit_time,
            inference,
            "Native anomaly score",
            _compact_pr_curve(y_test, scores),
        )
    )
    models["Isolation Forest"] = {"model": if_model, "threshold": if_threshold}

    lof_neighbors = [20, 50] if quick else settings["lof_neighbors"]
    lof_metrics_to_try = ["euclidean"] if quick else settings["lof_metrics"]
    lof_model, lof_params, lof_threshold, lof_validation, lof_fit_time = _select_lof(
        x_train, y_validation, x_validation, lof_neighbors, lof_metrics_to_try
    )
    started = perf_counter()
    scores = lof_scores(lof_model, x_test)
    predictions = (scores >= lof_threshold).astype(int)
    inference = perf_counter() - started
    test_metrics = evaluate_binary(y_test, predictions, scores, lof_threshold)
    results.append(
        _result(
            "Local Outlier Factor",
            "Unsupervised (label-guided validation)",
            lof_params,
            lof_validation,
            test_metrics,
            lof_fit_time,
            inference,
            "Native novelty score",
            _compact_pr_curve(y_test, scores),
        )
    )
    models["Local Outlier Factor"] = {"model": lof_model, "threshold": lof_threshold}

    rf_iterations = 4 if quick else 24
    rf_model, rf_params, rf_threshold, rf_validation, rf_fit_time = _select_random_forest(
        x_train,
        y_train,
        y_validation,
        x_validation,
        settings,
        config["project"]["random_state"],
        rf_iterations,
    )
    started = perf_counter()
    scores = failure_probabilities(rf_model, x_test)
    predictions = (scores >= rf_threshold).astype(int)
    inference = perf_counter() - started
    test_metrics = evaluate_binary(y_test, predictions, scores, rf_threshold)
    results.append(
        _result(
            "Random Forest",
            "Supervised",
            rf_params,
            rf_validation,
            test_metrics,
            rf_fit_time,
            inference,
            "Predicted probability",
            _compact_pr_curve(y_test, scores),
        )
    )
    models["Random Forest"] = {"model": rf_model, "threshold": rf_threshold}

    payload = {
        "protocol": {
            "split": "60/20/20 stratified",
            "random_state": config["project"]["random_state"],
            "threshold_objective": "Maximum validation F1",
            "test_policy": "Untouched until configurations and thresholds were selected",
            "refit_after_validation": False,
            "feature_columns": FEATURE_COLUMNS,
            "mode": "Quick integration smoke test" if quick else "Official benchmark",
            "evaluation_split": "Validation reuse (not official)" if quick else "Untouched test set",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "results": results,
        "sensitivity_results": [sensitivity_result],
    }
    bundle = {
        "preprocessor": preprocessor,
        "split": split,
        "models": models,
        "results": payload,
        "sensitivity": {
            "preprocessor": numeric_preprocessor,
            "DBSCAN": {"fit": sensitivity_fit, "threshold": 1.0},
            "result": sensitivity_result,
        },
    }
    if quick:
        save_joblib(bundle, root_path / "artifacts" / "smoke" / "quick_benchmark.joblib")
        save_json(payload, root_path / "results" / "quick_smoke_results.json")
    else:
        save_joblib(bundle, paths.bundle)
        save_json(payload, paths.results_json)
        for result in results:
            append_result(paths.experiments_csv, result, split="test")
    return payload
