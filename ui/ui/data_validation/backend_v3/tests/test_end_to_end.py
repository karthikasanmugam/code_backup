from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lerobot_validator.config import load_config
from lerobot_validator.reporting import write_reports
from lerobot_validator.validator import validate_dataset


# ============================================================
# Dummy dataset
# ============================================================

def create_info(
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
            "observation.images.front": {
                "dtype": "uint8",
                "shape": [3, 64, 64],
            },
        },
    }

    path = (
        root
        / "meta"
        / "info.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            info,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_episode_dataframe(
    n_frames: int = 30,
    bad_frame_index: bool = False,
) -> pd.DataFrame:
    states = []

    actions = []

    for index in range(n_frames):
        states.append(
            np.array(
                [
                    float(index),
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
        )

        actions.append(
            np.array(
                [
                    0.1,
                    0.1,
                    0.1,
                ],
                dtype=np.float32,
            )
        )

    frame_index = np.arange(
        n_frames
    )

    if bad_frame_index:
        frame_index[10] = 999

    return pd.DataFrame(
        {
            "frame_index": frame_index,
            "timestamp": np.arange(
                n_frames,
                dtype=np.float64,
            ) / 30.0,
            "observation.state": states,
            "action": actions,
            "task_index": np.zeros(
                n_frames,
                dtype=np.int64,
            ),
        }
    )


def create_dummy_video(
    path: Path,
    n_frames: int = 30,
    fps: float = 30.0,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 64),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create dummy video"
        )

    try:
        for index in range(n_frames):
            # Make every frame different so frozen-video detection
            # does not trigger.
            frame = np.zeros(
                (64, 64, 3),
                dtype=np.uint8,
            )

            value = (
                20
                + index * 5
            ) % 255

            frame[:, :, :] = value

            # Add a high-frequency pattern so image quality does
            # not interpret the frame as completely blurred.
            frame[::4, ::4] = 255

            writer.write(frame)
    finally:
        writer.release()


def create_dummy_dataset(
    root: Path,
) -> None:
    create_info(root)

    data_dir = root / "data"
    video_dir = (
        root
        / "videos"
        / "front"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_dummy_video(
        video_dir
        / "episode_000000.mp4"
    )

    create_dummy_video(
        video_dir
        / "episode_000001.mp4"
    )

    create_episode_dataframe(
        n_frames=30,
        bad_frame_index=False,
    ).to_parquet(
        data_dir
        / "episode_000000.parquet",
        index=False,
    )

    create_episode_dataframe(
        n_frames=30,
        bad_frame_index=True,
    ).to_parquet(
        data_dir
        / "episode_000001.parquet",
        index=False,
    )


# ============================================================
# Test
# ============================================================

def test_complete_dummy_validation_workflow(
    tmp_path,
):
    dataset_root = (
        tmp_path
        / "dummy_dataset"
    )

    create_dummy_dataset(
        dataset_root
    )

    # Use the project's actual default configuration.
    config = load_config(
        Path("configs/default.yml")
    )

    # Adjust optional checks for this deliberately tiny synthetic
    # dataset. The real project defaults remain unchanged.
# Disable expensive/optional checks for the tiny synthetic dataset.
# Use setdefault() so the test remains valid even when a custom
# default.yml omits an optional subsection.

    config.raw.setdefault(
        "video",
        {}
    )

    config.raw["video"].setdefault(
        "camera_coverage",
        {}
    )
    config.raw["video"]["camera_coverage"]["enabled"] = False

    config.raw["video"].setdefault(
        "image_quality",
        {}
    )
    config.raw["video"]["image_quality"]["enabled"] = False


    config.raw.setdefault(
        "actions",
        {}
    )

    for section in (
        "smoothness",
        "state_lag",
        "alignment",
        "gripper",
        "token_budget",
        "tokenizer_roundtrip",
    ):
        config.raw["actions"].setdefault(
            section,
            {}
        )
        config.raw["actions"][section]["enabled"] = False


    config.raw.setdefault(
        "episode",
        {}
    )

    config.raw["episode"].setdefault(
        "instruction_match",
        {}
    )
    config.raw["episode"]["instruction_match"]["enabled"] = False

    metadata = {
        0: {
            "length": 30,
            "success": True,
        },
        1: {
            "length": 30,
            "success": True,
        },
    }

    tasks = {
        0: {
            "instruction": "move the object forward",
        }
    }

    validation = validate_dataset(
        data_root=dataset_root,
        config=config,
        metadata_map=metadata,
        tasks_map=tasks,
    )

    assert validation.total_episodes == 2

    episode_0 = validation.episodes[0]
    episode_1 = validation.episodes[1]

    # Episode 0 should pass all enabled checks.
    assert episode_0.episode_index == 0
    assert episode_0.results == []
    assert episode_0.score == 1.0
    assert episode_0.classification == "KEEP"

    # Episode 1 has a structural HARD failure.
    assert episode_1.episode_index == 1
    assert any(
        result.check == "frame_index"
        for result in episode_1.results
    )

    assert episode_1.classification == "REJECT"

    assert validation.kept_episode_numbers == [0]
    assert validation.rejected_episode_numbers == [1]

    assert validation.flagged_episode_numbers == [1]

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    output_dir = (
        tmp_path
        / "reports"
    )

    paths = write_reports(
        validation=validation,
        output_dir=output_dir,
        data_root=dataset_root,
        fps=30.0,
        stuck_thresh=2.0,
    )

    for path in paths.values():
        assert path.exists()

    flagged_ids = (
        paths["flagged_ids"]
        .read_text(
            encoding="utf-8"
        )
    )

    kept_ids = (
        paths["kept_ids"]
        .read_text(
            encoding="utf-8"
        )
    )

    assert flagged_ids == "1\n"
    assert kept_ids == "0\n"

    log = (
        paths["flagged_log"]
        .read_text(
            encoding="utf-8"
        )
    )

    assert "FLAGGED EPISODES VALIDATION AUDIT" in log
    assert "[EPISODE 1]" in log
    assert "[HARD] frame_index" in log

    results = json.loads(
        paths["results_json"]
        .read_text(
            encoding="utf-8"
        )
    )

    assert results["total_episodes"] == 2
    assert results["kept_episode_numbers"] == [0]
    assert results["rejected_episode_numbers"] == [1]