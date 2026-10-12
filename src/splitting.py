from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    random_state: int = 42

    def validate(self, total_rows: int) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("Split indices overlap.")
        combined = groups[0] | groups[1] | groups[2]
        if combined != set(range(total_rows)):
            raise ValueError("Split indices do not cover the dataset exactly once.")

    def to_dict(self) -> dict:
        return {
            "random_state": self.random_state,
            "train": self.train.tolist(),
            "validation": self.validation.tolist(),
            "test": self.test.tolist(),
        }


def create_stratified_split(
    target: pd.Series,
    random_state: int = 42,
) -> SplitIndices:
    indices = np.arange(len(target))
    train_val, test = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_state,
        stratify=target,
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=target.iloc[train_val],
    )
    result = SplitIndices(
        train=np.sort(train),
        validation=np.sort(validation),
        test=np.sort(test),
        random_state=random_state,
    )
    result.validate(len(target))
    return result


def save_split(split: SplitIndices, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")


def load_split(path: str | Path, total_rows: int | None = None) -> SplitIndices:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = SplitIndices(
        train=np.asarray(payload["train"], dtype=int),
        validation=np.asarray(payload["validation"], dtype=int),
        test=np.asarray(payload["test"], dtype=int),
        random_state=int(payload.get("random_state", 42)),
    )
    if total_rows is not None:
        split.validate(total_rows)
    return split

