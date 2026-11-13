from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "data" / "config.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for name in ("projection_temperature", "smoothing_sigma", "ridge"):
        value = config[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {name}: {value}")
    for group in ("covariance", "visual", "prompt", "evidence"):
        if not isinstance(config[group], dict):
            raise ValueError(f"Invalid {group} settings")
    if not 0 <= config["visual"]["center_weight"] <= 1:
        raise ValueError("visual.center_weight must be in [0, 1]")
    if not 0 <= config["evidence"]["weight"]:
        raise ValueError("evidence.weight must be nonnegative")
    if config["prompt"]["normal_reference_count"] < 1:
        raise ValueError("prompt.normal_reference_count must be positive")
    return config
