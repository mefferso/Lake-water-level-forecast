from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config.yml") -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for path in [
        "data/raw/coops",
        "data/raw/synoptic",
        "data/processed",
        "models",
        "outputs/plots",
        "outputs/tables",
    ]:
        (REPO_ROOT / path).mkdir(parents=True, exist_ok=True)
