"""YAML config loader with tyro CLI override support.

Usage pattern:
    config = load_config(OptimizeExcitationConfig, config_path="configs/default.yaml")

This loads defaults from the YAML file, then lets tyro override any field via CLI flags.
"""

from __future__ import annotations

import typing
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TypeVar, get_args, get_origin

import tyro
import yaml

T = TypeVar("T")

_DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(
    dataclass_type: type[T],
    config_path: str | Path | None = None,
) -> T:
    """Load config from YAML defaults, then apply tyro CLI overrides.

    1. Read YAML file (defaults to configs/default.yaml)
    2. Build a dataclass instance with YAML values as defaults
    3. Pass it to tyro.cli so CLI flags override YAML values
    """
    yaml_data = load_yaml(config_path)
    kwargs = apply_yaml_defaults(dataclass_type, yaml_data)
    default_instance = dataclass_type(**kwargs)
    return tyro.cli(dataclass_type, default=default_instance)


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
            value = flat[f.name]
            if _type_compatible(f.type, value):
                result[f.name] = value

    return result


def _resolve_type(annotation: str | type) -> type | None:
    """Resolve a string or type annotation to its origin type."""
    if isinstance(annotation, str):
        # Evaluate string annotations (from __future__ annotations)
        try:
            annotation = eval(annotation, {**typing.__dict__})
        except Exception:
            return None
    origin = get_origin(annotation)
    if origin is typing.Union:
        # For Optional/Union, check non-None args
        args = [a for a in get_args(annotation) if a is not type(None)]
        return args[0] if args else None
    return origin or annotation


def _type_compatible(annotation: str | type, value: object) -> bool:
    """Check if a YAML value is broadly compatible with a dataclass field type."""
    resolved = _resolve_type(annotation)
    if resolved is None:
        return True  # Can't resolve → accept
    # int/float are compatible with each other in YAML
    if resolved in (int, float) and isinstance(value, (int, float)):
        return True
    if resolved is str and isinstance(value, str):
        return True
    if resolved is bool and isinstance(value, bool):
        return True
    if resolved in (list, tuple) and isinstance(value, (list, tuple)):
        return True
    # Reject clear mismatches (e.g. list value for scalar field)
    if resolved in (int, float, str, bool) and isinstance(value, (list, dict)):
        return False
    return True  # Unknown types → accept


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
