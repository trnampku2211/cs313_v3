from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Configuration not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

