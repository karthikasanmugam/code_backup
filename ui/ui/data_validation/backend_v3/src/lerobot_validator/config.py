from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import ValidationConfig


class ConfigError(ValueError):
    """Raised when the validator configuration is invalid."""


# Top-level sections expected by configs/default.yml.
REQUIRED_SECTIONS = {
    "dataset",
    "execution",
    "setup",
    "structural",
    "video",
    "actions",
    "episode",
    "dataset_quality",
    "scoring",
    "reporting",
    "logging",
}


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file and return its top-level mapping.

    Raises:
        FileNotFoundError: if the configuration file does not exist.
        ConfigError: if the YAML is malformed or does not contain a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if not path.is_file():
        raise ConfigError(f"Configuration path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML in configuration file: {path}"
        ) from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Top-level configuration must be a mapping, got "
            f"{type(data).__name__}"
        )

    return data


def validate_config(data: dict[str, Any]) -> None:
    """
    Validate the high-level structure of the configuration.

    This intentionally checks only the configuration contract that
    the project depends on. Individual field/value validation can be
    expanded later without changing the loader interface.
    """
    missing = REQUIRED_SECTIONS - data.keys()

    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ConfigError(
            f"Configuration is missing required sections: {missing_list}"
        )

    for section in REQUIRED_SECTIONS:
        if not isinstance(data[section], dict):
            raise ConfigError(
                f"Configuration section '{section}' must be a mapping, "
                f"got {type(data[section]).__name__}"
            )


def load_config(path: Path) -> ValidationConfig:
    """
    Load and validate a YAML configuration file.

    Args:
        path: Path to YAML configuration.

    Returns:
        ValidationConfig containing the raw validated configuration.
    """
    data = load_yaml(path)
    validate_config(data)

    return ValidationConfig(raw=data)