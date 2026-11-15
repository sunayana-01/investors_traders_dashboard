"""Loads config/config.yaml.

The config is a normal nested dictionary, for example cfg["labels"]["horizons"].
"""
import hashlib
import json
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"

TRADING_DAYS = 252   # trading days in a year, used to turn daily numbers into yearly ones


class ConfigError(ValueError):
    pass


def check_config(cfg):
    """Stop early with a clear message if something important is missing or silly."""
    sections = ["project", "data", "portfolios", "labels", "validation",
                "calibration", "inference", "overlay", "trust"]
    for name in sections:
        if name not in cfg:
            raise ConfigError(f"The config is missing the '{name}' section")

    labels = cfg["labels"]
    if labels["primary_horizon"] not in labels["horizons"]:
        raise ConfigError("labels.primary_horizon must be one of labels.horizons")
    if not 0 < labels["quantile"] < 1:
        raise ConfigError("labels.quantile must be between 0 and 1")
    if not 0 < cfg["validation"]["calibration_fraction"] < 0.5:
        raise ConfigError("validation.calibration_fraction must be between 0 and 0.5")

    for pid, portfolio in cfg["portfolios"].items():
        if abs(sum(portfolio["weights"].values()) - 1) > 1e-6:
            raise ConfigError(f"The weights of portfolio {pid} must add up to 1")

    allowed = cfg["calibration"]["candidates"]
    chosen = cfg["calibration"].get("selected") or {}
    for model, method in chosen.items():
        if method not in allowed:
            raise ConfigError(f"calibration.selected.{model} = {method!r} is not in the candidates")


def load_config(path=None):
    """Read and check a config file. Order: given path, RISKML_CONFIG variable, default file."""
    path = Path(path or os.environ.get("RISKML_CONFIG", DEFAULT_CONFIG))
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    check_config(cfg)
    return cfg


_config = None
_config_time = None


def get_config():
    """Load the default config and reuse it. It is read again if config.yaml was edited."""
    global _config, _config_time
    saved_time = DEFAULT_CONFIG.stat().st_mtime
    if _config is None or saved_time != _config_time:
        _config = load_config()
        _config_time = saved_time
    return _config


def config_hash(cfg):
    """Short fingerprint of the whole config. It is saved with every result."""
    text = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def project_path(relative):
    """Full path for something inside the project. Makes the folder if it does not exist yet."""
    path = PROJECT_ROOT / relative
    folder = path if not path.suffix else path.parent
    folder.mkdir(parents=True, exist_ok=True)
    return path
