from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lerobot_validator.dataset import (
    compute_dataset_stats,
    load_episode_dataframe,
    _infer_camera_hw,
    load_reference_action_percentiles,
    load_dataset_info,
    resolve_episode_parquet,
    resolve_episode_paths,
    resolve_episode_video,
)
from lerobot_validator.actions import check_action_range
from lerobot_validator.models import ValidationConfig
from lerobot_validator.validator import setup_dataset


@pytest.mark.parametrize(
    ("shape", "names", "expected"),
    [
        ([240, 320, 3], ["height", "width", "channels"], (240, 320)),
        ([3, 240, 320], ["channels", "height", "width"], (240, 320)),
        ([240, 320, 3], None, (240, 320)),
        ([240, 320, 1], None, (240, 320)),
    ],
)
def test_infer_camera_hw(shape, names, expected):
    feature = {"shape": shape}
    if names is not None:
        feature["names"] = names

    assert _infer_camera_hw(feature) == expected


def test_load_dataset_info_uses_declared_camera_keys_and_count(tmp_path: Path):
    camera_keys = [
        "observation.images.zed_right",
        "observation.images.zed_left",
        "observation.images.wrist_right",
    ]

    features = {
        "action": {"dtype": "float32", "shape": [16]},
        "observation.state": {"dtype": "float32", "shape": [16]},
    }

    for key in camera_keys:
        features[key] = {
            "dtype": "video",
            "shape": [240, 320, 3],
            "names": ["height", "width", "channels"],
        }

    # This is deliberately an unrelated image-like feature. The explicit
    # camera_keys list should be authoritative and keep it out of cameras.
    features["observation.images.extra"] = {
        "dtype": "video",
        "shape": [120, 160, 3],
        "names": ["height", "width", "channels"],
    }

    info = {
        "codebase_version": "custom_v3_local",
        "fps": 30,
        "camera_keys": camera_keys,
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1024,
    }

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))

    dataset_info = load_dataset_info(tmp_path)

    assert dataset_info.fps == 30.0
    assert dataset_info.state_dim == 16
    assert dataset_info.action_dim == 16
    assert dataset_info.camera_keys == camera_keys
    assert len(dataset_info.cameras) == 3

    for camera in dataset_info.cameras.values():
        assert camera.height == 240
        assert camera.width == 320


def test_resolve_episode_paths_from_info_templates(tmp_path: Path):
    camera_key = "observation.images.zed_right"
    info = {
        "fps": 30,
        "camera_keys": [camera_key],
        "features": {
            "action": {"dtype": "float32", "shape": [16]},
            "observation.state": {"dtype": "float32", "shape": [16]},
            camera_key: {
                "dtype": "video",
                "shape": [240, 320, 3],
                "names": ["height", "width", "channels"],
            },
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1024,
    }

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))

    parquet = tmp_path / "data" / "chunk-000" / "episode_000007.parquet"
    video = (
        tmp_path
        / "videos"
        / camera_key
        / "chunk-000"
        / "episode_000007.mp4"
    )

    parquet.parent.mkdir(parents=True)
    video.parent.mkdir(parents=True)
    parquet.touch()
    video.touch()

    assert resolve_episode_parquet(tmp_path, 7) == parquet
    assert resolve_episode_video(tmp_path, 7, camera_key) == video


def test_load_episode_dataframe_filters_rows_from_v3_chunk(tmp_path: Path):
    parquet = tmp_path / "file-000.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1, 1],
            "frame_index": [0, 1, 0, 1, 2],
            "action": [[0.0], [1.0], [2.0], [3.0], [4.0]],
        }
    ).to_parquet(parquet)

    episode = load_episode_dataframe(parquet, episode_index=1)

    assert episode["episode_index"].unique().tolist() == [1]
    assert episode["frame_index"].tolist() == [0, 1, 2]
    assert len(episode) == 3


def test_compute_dataset_stats_groups_v3_chunk_by_episode(tmp_path: Path):
    parquet = tmp_path / "file-000.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 1, 1],
            "frame_index": [0, 1, 2, 0, 1],
            "timestamp": [0.0, 1.0, 2.0, 0.0, 1.0],
            "action": [[0.0], [1.0], [2.0], [3.0], [4.0]],
            "observation.state": [[0.0], [1.0], [2.0], [3.0], [4.0]],
            "task_index": [0, 0, 0, 1, 1],
        }
    ).to_parquet(parquet)

    stats = compute_dataset_stats(
        parquet_files=[parquet, parquet],
        action_col="action",
        percentiles=(0, 100),
    )

    assert stats.median_episode_length == 2.5
    assert stats.action_percentiles["low"].tolist() == [0.0]
    assert stats.action_percentiles["high"].tolist() == [4.0]
    assert stats.task_counts == {0: 3, 1: 2}


def test_resolve_v3_paths_and_video_segment_from_episode_metadata(tmp_path: Path):
    camera_key = "observation.images.front"
    info = {
        "fps": 30,
        "chunks_size": 1000,
        "camera_keys": [camera_key],
        "features": {
            "action": {"dtype": "float32", "shape": [2]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            camera_key: {"dtype": "video", "shape": [240, 320, 3]},
        },
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps(info))

    data_file = tmp_path / "data/chunk-002/file-004.parquet"
    video_file = tmp_path / "videos" / camera_key / "chunk-003/file-006.mp4"
    data_file.parent.mkdir(parents=True)
    video_file.parent.mkdir(parents=True)
    data_file.touch()
    video_file.touch()
    episode_metadata = {
        "data/chunk_index": 2,
        "data/file_index": 4,
        f"videos/{camera_key}/chunk_index": 3,
        f"videos/{camera_key}/file_index": 6,
        f"videos/{camera_key}/from_timestamp": 12.5,
        f"videos/{camera_key}/to_timestamp": 15.0,
    }

    paths = resolve_episode_paths(
        tmp_path,
        ep_idx=42,
        camera_keys=[camera_key],
        episode_metadata=episode_metadata,
    )

    assert paths.parquet == data_file
    assert paths.videos[camera_key] == video_file
    assert paths.video_ranges[camera_key] == (12.5, 15.0)


def test_setup_dataset_uses_frozen_action_range_after_pruning(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "chunks_size": 1000,
                "features": {
                    "action": {"dtype": "float32", "shape": [1]},
                    "observation.state": {"dtype": "float32", "shape": [1]},
                },
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        )
    )
    (meta / "episodes").mkdir()
    (meta / "episodes" / "chunk-000").mkdir()
    pd.DataFrame({"episode_index": [0, 0, 0]}).to_parquet(
        meta / "episodes" / "chunk-000" / "episodes.parquet"
    )
    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3],
            "timestamp": [0.0, 1.0, 2.0, 3.0],
            "action": [[0.0], [1.0], [1.0], [2.0]],
            "observation.state": [[0.0], [1.0], [1.0], [2.0]],
            "task_index": [0, 0, 0, 0],
        }
    ).to_parquet(data_path)
    reference_path = meta / "validation_reference_action_percentiles.json"
    reference_path.write_text(json.dumps({"version": 1, "low": [-2.0], "high": [2.5]}))

    config = ValidationConfig(
        raw={"setup": {"action_percentiles": {"low": 1.0, "high": 99.0}}}
    )
    _, _, stats = setup_dataset(tmp_path, config)

    assert stats.action_percentiles["low"].tolist() == [-2.0]
    assert stats.action_percentiles["high"].tolist() == [2.5]
    assert check_action_range([[2.0]], stats.action_percentiles) == []

    recomputed = compute_dataset_stats(
        [data_path], action_col="action", percentiles=(1, 99)
    )
    assert check_action_range([[2.0]], recomputed.action_percentiles)


def test_load_reference_action_percentiles_only_applies_to_default_pair(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "validation_reference_action_percentiles.json").write_text(
        json.dumps({"version": 1, "low": [0.0], "high": [1.0]})
    )

    reference = load_reference_action_percentiles(tmp_path, (1, 99))
    assert reference is not None
    assert reference["high"].tolist() == [1.0]
    assert load_reference_action_percentiles(tmp_path, (5, 95)) is None

    (meta / "validation_reference_action_percentiles.json").write_text(
        json.dumps({"low": [-100.0], "high": [100.0]})
    )
    assert load_reference_action_percentiles(tmp_path, (1, 99)) is None
