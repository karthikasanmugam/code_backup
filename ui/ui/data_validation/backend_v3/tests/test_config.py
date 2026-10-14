from pathlib import Path

import pytest

from lerobot_validator.config import (
    ConfigError,
    load_config,
    load_yaml,
    validate_config,
)


def make_valid_config() -> dict:
    """
    Minimal valid configuration matching the project contract.
    """
    sections = [
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
    ]

    return {section: {} for section in sections}


def write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_yaml_reads_mapping(tmp_path):
    config_path = tmp_path / "config.yml"

    write_yaml(
        config_path,
        """
dataset:
  format: lerobot_v2

execution:
  workers: 1
""",
    )

    data = load_yaml(config_path)

    assert data == {
        "dataset": {
            "format": "lerobot_v2",
        },
        "execution": {
            "workers": 1,
        },
    }


def test_load_yaml_empty_file_returns_empty_dict(tmp_path):
    config_path = tmp_path / "empty.yml"

    write_yaml(config_path, "")

    data = load_yaml(config_path)

    assert data == {}


def test_load_yaml_missing_file(tmp_path):
    config_path = tmp_path / "does_not_exist.yml"

    with pytest.raises(FileNotFoundError):
        load_yaml(config_path)


def test_load_yaml_rejects_directory(tmp_path):
    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()

    with pytest.raises(ConfigError, match="is not a file"):
        load_yaml(config_dir)


def test_load_yaml_rejects_invalid_yaml(tmp_path):
    config_path = tmp_path / "bad.yml"

    write_yaml(
        config_path,
        """
dataset:
  format: [lerobot_v2
""",
    )

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_yaml(config_path)


def test_load_yaml_rejects_non_mapping(tmp_path):
    config_path = tmp_path / "bad.yml"

    write_yaml(
        config_path,
        """
- dataset
- execution
""",
    )

    with pytest.raises(
        ConfigError,
        match="Top-level configuration must be a mapping",
    ):
        load_yaml(config_path)


def test_validate_config_accepts_valid_config():
    data = make_valid_config()

    validate_config(data)


def test_validate_config_detects_missing_sections():
    data = make_valid_config()

    del data["video"]
    del data["actions"]

    with pytest.raises(ConfigError, match="missing required sections"):
        validate_config(data)


def test_validate_config_reports_missing_section_names():
    data = make_valid_config()

    del data["video"]

    with pytest.raises(
        ConfigError,
        match=r"video",
    ):
        validate_config(data)


def test_validate_config_rejects_non_mapping_section():
    data = make_valid_config()

    data["video"] = []

    with pytest.raises(
        ConfigError,
        match="section 'video' must be a mapping",
    ):
        validate_config(data)


def test_load_config_returns_validation_config(tmp_path):
    config_path = tmp_path / "config.yml"

    write_yaml(
        config_path,
        """
dataset:
  format: lerobot_v2

execution:
  short_circuit_on_hard: true
  skip_expensive_after_hard: true

setup: {}
structural: {}
video: {}
actions: {}
episode: {}
dataset_quality: {}
scoring: {}
reporting: {}
logging: {}
""",
    )

    config = load_config(config_path)

    assert config.raw["dataset"]["format"] == "lerobot_v2"
    assert config.short_circuit_on_hard is True
    assert config.skip_expensive_after_hard is True


def test_load_config_preserves_nested_values(tmp_path):
    config_path = tmp_path / "config.yml"

    write_yaml(
        config_path,
        """
dataset:
  format: lerobot_v2
  episode_indices: [0, 2, 7]

execution:
  workers: 4
  verbose: false

setup:
  action_percentiles:
    low: 1.0
    high: 99.0

structural:
  min_episode_length: 20
  max_episode_length: 1000

video: {}
actions: {}
episode: {}
dataset_quality: {}
scoring: {}
reporting: {}
logging: {}
""",
    )

    config = load_config(config_path)

    assert config.raw["dataset"]["episode_indices"] == [0, 2, 7]
    assert config.raw["execution"]["workers"] == 4
    assert config.raw["execution"]["verbose"] is False
    assert config.raw["setup"]["action_percentiles"]["low"] == 1.0
    assert config.raw["structural"]["max_episode_length"] == 1000