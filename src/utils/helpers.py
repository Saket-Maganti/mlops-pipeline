"""
Shared utilities for the MLOps pipeline.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def append_jsonl(record: Dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def get_latest_file(directory: str, pattern: str = "*.csv") -> str:
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return str(files[-1])


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
