from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lerobot_validator.deletion import _preserve_validation_reference_stats


def test_delete_freezes_validator_percentiles_not_lerobot_aggregate(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_meta = source_root / "meta"
    source_meta.mkdir(parents=True)
    (source_meta / "stats.json").write_text(
        json.dumps({"action": {"q01": [-2.0], "q99": [2.5]}})
    )
    parquet = source_root / "data" / "chunk-000" / "file-000.parquet"
    parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3],
            "action": [[0.0], [1.0], [1.0], [2.0]],
        }
    ).to_parquet(parquet)
    output_root = tmp_path / "rebuilt"

    _preserve_validation_reference_stats(source_meta, output_root)

    reference_path = output_root / "meta" / "validation_reference_action_percentiles.json"
    frozen = json.loads(reference_path.read_text())
    assert frozen["version"] == 1
    assert frozen["low"] == [0.03]
    assert frozen["high"][0] == pytest.approx(1.97)
    assert frozen["low"] != [-2.0]
    assert frozen["high"] != [2.5]


def test_subsequent_delete_keeps_the_original_validator_reference(tmp_path: Path) -> None:
    source_meta = tmp_path / "source" / "meta"
    source_meta.mkdir(parents=True)
    (source_meta / "validation_reference_action_percentiles.json").write_text(
        json.dumps({"version": 1, "low": [-1.0], "high": [1.0]})
    )
    (source_meta / "stats.json").write_text(
        json.dumps({"action": {"q01": [-0.2], "q99": [0.2]}})
    )
    output_root = tmp_path / "rebuilt"

    _preserve_validation_reference_stats(source_meta, output_root)

    frozen = json.loads(
        (output_root / "meta" / "validation_reference_action_percentiles.json").read_text()
    )
    assert frozen == {"version": 1, "low": [-1.0], "high": [1.0]}


def test_legacy_reference_is_recomputed_from_data(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_meta = source_root / "meta"
    source_meta.mkdir(parents=True)
    (source_meta / "validation_reference_action_percentiles.json").write_text(
        json.dumps({"low": [-200.0], "high": [200.0]})
    )
    parquet = source_root / "data" / "chunk-000" / "file-000.parquet"
    parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        {"episode_index": [0, 0, 0], "action": [[0.0], [1.0], [2.0]]}
    ).to_parquet(parquet)
    output_root = tmp_path / "rebuilt"

    _preserve_validation_reference_stats(source_meta, output_root)

    frozen = json.loads(
        (output_root / "meta" / "validation_reference_action_percentiles.json").read_text()
    )
    assert frozen["version"] == 1
    assert frozen["low"][0] == pytest.approx(0.02)
    assert frozen["high"][0] == pytest.approx(1.98)
