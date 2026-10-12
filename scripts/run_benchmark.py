from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import run_official_benchmark
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed official model benchmark.")
    parser.add_argument("--quick", action="store_true", help="Use a reduced search grid for smoke testing.")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    args = parser.parse_args()
    payload = run_official_benchmark(ROOT, load_config(args.config), quick=args.quick)
    summary = {
        "mode": payload["protocol"]["mode"],
        "evaluation_split": payload["protocol"]["evaluation_split"],
        "models": [
            {
                "model": result["model"],
                "f1": result["test_metrics"]["f1"],
                "average_precision": result["test_metrics"]["average_precision"],
            }
            for result in payload["results"] + payload.get("sensitivity_results", [])
        ],
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
