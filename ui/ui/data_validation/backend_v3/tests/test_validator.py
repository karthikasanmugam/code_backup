from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lerobot_validator.validator import (
    _derive_task_map_from_dataframe,
    _missing_camera_results,
    _resolve_episode_metadata,
    discover_episode_indices,
    load_episode_metadata,
    load_task_map,
)


def test_load_episode_metadata_from_parquet(tmp_path):
    meta_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    meta_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"episode_index": 0, "length": 500},
            {"episode_index": 7, "length": 420},
        ]
    ).to_parquet(meta_dir / "episodes.parquet")

    metadata = load_episode_metadata(tmp_path)

    assert metadata[0]["length"] == 500
    assert metadata[7]["length"] == 420
    assert _resolve_episode_metadata(metadata, 7)["length"] == 420


def test_load_episode_metadata_jsonl_fallback(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True)

    path = meta_dir / "episodes.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 2, "length": 300}),
                json.dumps({"episode_index": 5, "length": 275}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = load_episode_metadata(tmp_path)

    assert metadata[2]["length"] == 300
    assert metadata[5]["length"] == 275


def test_load_task_map_from_parquet(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "task_index": 0,
                "task": "pick up the brown block",
            }
        ]
    ).to_parquet(meta_dir / "tasks.parquet")

    tasks = load_task_map(tmp_path)

    assert tasks[0] == "pick up the brown block"


def test_load_task_map_jsonl_fallback(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True)

    path = meta_dir / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_index": 0,
                "instruction": "place the block in the tray",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tasks = load_task_map(tmp_path)

    assert tasks[0] == "place the block in the tray"


def test_derive_task_map_from_dataframe():
    df = pd.DataFrame(
        {
            "task_index": [0, 0, 1],
            "task": [
                "pick up the block",
                "pick up the block",
                "place it in the tray",
            ],
        }
    )

    tasks = _derive_task_map_from_dataframe(df)

    assert tasks == {
        0: "pick up the block",
        1: "place it in the tray",
    }


def test_missing_camera_results_are_dynamic():
    results = _missing_camera_results(
        expected_camera_keys=[
            "camera_a",
            "camera_b",
            "camera_c",
        ],
        video_paths={
            "camera_a": Path("/tmp/a.mp4"),
            "camera_c": Path("/tmp/c.mp4"),
        },
    )

    assert len(results) == 1
    assert results[0].check == "video_exists"
    assert results[0].severity == "HARD"
    assert "camera_b" in results[0].message


def test_missing_camera_results_empty_when_all_present():
    results = _missing_camera_results(
        expected_camera_keys=[
            "camera_a",
            "camera_b",
        ],
        video_paths={
            "camera_a": Path("/tmp/a.mp4"),
            "camera_b": Path("/tmp/b.mp4"),
        },
    )

    assert results == []


def test_episode_metadata_loader_returns_empty_when_absent(tmp_path):
    assert load_episode_metadata(tmp_path) == {}


def test_task_loader_returns_empty_when_absent(tmp_path):
    assert load_task_map(tmp_path) == {}


def test_load_v3_task_map_when_task_text_is_parquet_index(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    path = meta_dir / "tasks.parquet"
    pd.DataFrame(
        [{"task_index": 0, "task": "pick up the brown block"}]
    ).set_index("task").to_parquet(path)

    assert load_task_map(tmp_path) == {0: "pick up the brown block"}


def test_discover_v3_episode_indices_from_chunk_parquet(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {"episode_index": [0, 0, 7, 7, 42]}
    ).to_parquet(data_dir / "file-000.parquet")

    assert discover_episode_indices(tmp_path) == [0, 7, 42]
