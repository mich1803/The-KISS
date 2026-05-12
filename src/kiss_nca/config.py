"""Configuration loading for scripted and notebook training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML training configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def validate_config(config: dict[str, Any]) -> None:
    """Validate project-specific configuration constraints."""

    if config["resolution"] != 64:
        raise ValueError("Only 64x64 training is supported by this project.")
    if config["neighborhood_size"] not in {3, 5}:
        raise ValueError("neighborhood_size must be 3 or 5.")
    if config["state_channels"] < 4:
        raise ValueError("state_channels must include RGB, alpha, and hidden channels.")
    if not config.get("paintings"):
        raise ValueError("At least one painting must be configured.")
