"""YAML config loader with tyro CLI override support.

Usage pattern:
    config = load_config(OptimizeExcitationConfig, config_path="configs/default.yaml")

This loads defaults from the YAML file, then lets tyro override any field via CLI flags.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TypeVar

import yaml

T = TypeVar("T")

_DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_yaml(path: str | Path | None = None) -> dict:
    """Load a YAML config file, returning an empty dict if the file doesn't exist."""
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def apply_yaml_defaults(dataclass_type: type[T], yaml_data: dict, prefix: str = "") -> dict:
    """Extract flat keyword arguments from nested YAML data for a dataclass.

    Maps nested YAML sections to flat dataclass fields using a naming convention.
    Returns a dict of {field_name: value} for fields present in the YAML.
    """
    if not is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type} is not a dataclass")

    result: dict = {}
    flat = _flatten_yaml(yaml_data)

    for f in fields(dataclass_type):
        # Try exact field name match in flat dict
        if f.name in flat:
            result[f.name] = flat[f.name]

    return result


def _flatten_yaml(data: dict, prefix: str = "") -> dict:
    """Flatten nested dict to single-level with dotless keys for common patterns."""
    result: dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Recurse and add both nested and flattened
            nested = _flatten_yaml(value, f"{prefix}{key}.")
            result.update(nested)
            # Also add child keys directly (for flat dataclass fields)
            for child_key, child_value in value.items():
                if not isinstance(child_value, dict):
                    result[child_key] = child_value
        else:
            result[f"{prefix}{key}"] = value
            result[key] = value
    return result
