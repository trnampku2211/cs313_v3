from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


RESULT_FIELDS = [
    "run_id",
    "created_at_utc",
    "model",
    "learning_type",
    "split",
    "precision",
    "recall",
    "f1",
    "average_precision",
    "false_positive",
    "false_negative",
    "fit_time_seconds",
    "inference_time_seconds",
    "threshold",
    "score_source",
    "parameters_json",
]


def append_result(path: str | Path, model_result: dict[str, Any], split: str = "test") -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics = model_result[f"{split}_metrics"]
    row = {
        "run_id": model_result.get("run_id", uuid4().hex),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_result["model"],
        "learning_type": model_result["learning_type"],
        "split": split,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "average_precision": metrics["average_precision"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "fit_time_seconds": model_result["fit_time_seconds"],
        "inference_time_seconds": model_result["inference_time_seconds"],
        "threshold": metrics["threshold"],
        "score_source": model_result["score_source"],
        "parameters_json": json.dumps(model_result["parameters"], sort_keys=True),
    }
    exists = destination.exists() and destination.stat().st_size > 0
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

