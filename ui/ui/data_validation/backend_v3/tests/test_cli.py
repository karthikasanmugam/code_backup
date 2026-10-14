from __future__ import annotations

import json
from pathlib import Path

from lerobot_validator.cli import main


def create_minimal_dataset(
    root: Path,
) -> None:
    info = {
        "fps": 30,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [3],
            },
            "action": {
                "dtype": "float32",
                "shape": [3],
            },
        },
    }

    meta = root / "meta"
    data = root / "data"

    meta.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        meta / "info.json"
    ).write_text(
        json.dumps(info),
        encoding="utf-8",
    )

    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "frame_index": np.arange(30),
            "timestamp": np.arange(30) / 30.0,
            "observation.state": [
                np.array(
                    [float(i), 0.0, 0.0],
                    dtype=np.float32,
                )
                for i in range(30)
            ],
            "action": [
                np.array(
                    [0.1, 0.1, 0.1],
                    dtype=np.float32,
                )
                for _ in range(30)
            ],
            "task_index": np.zeros(
                30,
                dtype=np.int64,
            ),
        }
    )

    df.to_parquet(
        data / "episode_000000.parquet",
        index=False,
    )


def test_cli_runs_with_dynamic_dataset_path(
    tmp_path,
):
    dataset_root = (
        tmp_path
        / "dataset"
    )

    create_minimal_dataset(
        dataset_root
    )

    config_path = (
        tmp_path
        / "config.yml"
    )

    config_path.write_text(
        """
dataset:
  format: lerobot_v2
  fps_override: null
  episode_indices: null

execution:
  short_circuit_on_hard: true
  skip_expensive_after_hard: true
  workers: 1
  verbose: false

setup:
  action_percentiles:
    low: 1.0
    high: 99.0

structural:
  required_columns:
    - frame_index
    - timestamp
    - observation.state
    - action
    - task_index
  min_episode_length: 20
  max_episode_length: 1000

video:
  enabled: false

actions:
  enabled: false

episode:
  enabled: false

dataset_quality:
  enabled: false

scoring:
  enabled: true
  weights: {}
  classification:
    reject_on_hard: true
    review_min_score: 0.70

reporting:
  output_dir: outputs

logging:
  level: INFO
""",
        encoding="utf-8",
    )

    output_dir = (
        tmp_path
        / "output"
    )

    result = main(
        [
            "--data-root",
            str(dataset_root),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0

    assert (
        output_dir
        / "flagged_episodes.log"
    ).exists()

    assert (
        output_dir
        / "flagged_episodes_ids.txt"
    ).exists()

    assert (
        output_dir
        / "kept_episodes_ids.txt"
    ).exists()

    assert (
        output_dir
        / "episode_scores.csv"
    ).exists()

    assert (
        output_dir
        / "validation_results.json"
    ).exists()