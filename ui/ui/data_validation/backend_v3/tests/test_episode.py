from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lerobot_validator.episode import (
    check_duration_outlier,
    check_idle_ratio,
    check_instruction_match,
    check_leading_trailing_idle,
    check_path_efficiency,
    check_proprio_sync,
    check_task_instruction,
    compute_displacement,
    compute_idle_ratio,
    compute_path_length,
    find_leading_trailing_idle,
    run_episode_quality_checks,
)


# ============================================================
# Helpers
# ============================================================

def make_states(
    n_frames: int = 100,
    dims: int = 3,
) -> np.ndarray:
    return np.zeros(
        (n_frames, dims),
        dtype=np.float64,
    )


def make_timestamps(
    n_frames: int = 100,
    fps: float = 10.0,
) -> np.ndarray:
    return np.arange(
        n_frames,
        dtype=np.float64,
    ) / fps


def make_episode_df(
    n_frames: int = 30,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": np.arange(n_frames),
            "timestamp": np.arange(n_frames) / 10.0,
            "observation.state": [
                np.zeros(3)
                for _ in range(n_frames)
            ],
            "action": [
                np.zeros(3)
                for _ in range(n_frames)
            ],
            "task_index": np.zeros(
                n_frames,
                dtype=np.int64,
            ),
        }
    )


# ============================================================
# Idle ratio
# ============================================================

def test_compute_idle_ratio_all_idle():
    states = make_states(20)
    actions = make_states(20)

    ratio = compute_idle_ratio(
        states,
        actions,
    )

    assert ratio == 1.0


def test_compute_idle_ratio_active_episode():
    states = make_states(20)
    actions = make_states(20)

    actions[:, 0] = 1.0

    ratio = compute_idle_ratio(
        states,
        actions,
    )

    assert ratio == 0.0


def test_compute_idle_ratio_mixed():
    states = make_states(10)
    actions = make_states(10)

    actions[:5, 0] = 1.0

    ratio = compute_idle_ratio(
        states,
        actions,
    )

    assert 0.0 < ratio < 1.0


def test_check_idle_ratio_valid():
    states = make_states(20)
    actions = make_states(20)

    actions[:, 0] = 1.0

    results = check_idle_ratio(
        states,
        actions,
        max_idle_ratio=0.50,
    )

    assert results == []


def test_check_idle_ratio_flags():
    states = make_states(20)
    actions = make_states(20)

    results = check_idle_ratio(
        states,
        actions,
        max_idle_ratio=0.50,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"
    assert results[0].check == "idle_ratio"


# ============================================================
# Boundary idle
# ============================================================

def test_find_leading_trailing_idle():
    states = make_states(20)

    # Create movement in frames 5-14.
    for i in range(5, 15):
        states[i, 0] = float(i)

    timestamps = make_timestamps(
        n_frames=20,
        fps=10,
    )

    first, last = find_leading_trailing_idle(
        states,
        timestamps,
        vel_eps=0.1,
    )

    assert first == 5
    assert last >= 15


def test_boundary_idle_short_is_valid():
    states = make_states(20)

    for i in range(2, 18):
        states[i, 0] = float(i)

    timestamps = make_timestamps(
        n_frames=20,
        fps=10,
    )

    results = check_leading_trailing_idle(
        states,
        timestamps,
        vel_eps=0.1,
        max_idle_seconds=2.0,
    )

    assert results == []


def test_boundary_idle_long_leading():
    states = make_states(100)

    for i in range(50, 100):
        states[i, 0] = float(i)

    timestamps = make_timestamps(
        n_frames=100,
        fps=10,
    )

    results = check_leading_trailing_idle(
        states,
        timestamps,
        vel_eps=0.1,
        max_idle_seconds=2.0,
    )

    assert len(results) >= 1
    assert results[0].severity == "SOFT"


# ============================================================
# Duration
# ============================================================

def test_duration_outlier_normal():
    result = check_duration_outlier(
        duration=10.0,
        dataset_median=10.0,
        mad=1.0,
        mad_z_thresh=3.5,
    )

    assert result == []


def test_duration_outlier_detected():
    result = check_duration_outlier(
        duration=30.0,
        dataset_median=10.0,
        mad=1.0,
        mad_z_thresh=3.5,
    )

    assert len(result) == 1
    assert result[0].severity == "SOFT"


def test_duration_outlier_zero_mad_same_value():
    result = check_duration_outlier(
        duration=10.0,
        dataset_median=10.0,
        mad=0.0,
    )

    assert result == []


def test_duration_outlier_zero_mad_different_value():
    result = check_duration_outlier(
        duration=20.0,
        dataset_median=10.0,
        mad=0.0,
    )

    assert len(result) == 1


# ============================================================
# Path efficiency
# ============================================================

def test_compute_path_length():
    states = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    path = compute_path_length(
        states
    )

    assert path == pytest.approx(2.0)


def test_compute_displacement():
    states = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    displacement = compute_displacement(
        states
    )

    assert displacement == pytest.approx(
        np.sqrt(2.0)
    )


def test_path_efficiency_normal():
    states = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )

    results = check_path_efficiency(
        states,
        dataset_median_path=2.0,
        ratio_thresh=2.0,
    )

    assert results == []


def test_path_efficiency_outlier():
    states = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ]
    )

    results = check_path_efficiency(
        states,
        dataset_median_path=2.0,
        ratio_thresh=2.0,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"


def test_path_efficiency_position_dimensions():
    states = np.array(
        [
            [0.0, 100.0],
            [1.0, 100.0],
            [2.0, 100.0],
        ]
    )

    result = compute_path_length(
        states,
        position_dims=[0],
    )

    assert result == pytest.approx(2.0)


# ============================================================
# Task instruction
# ============================================================

def test_task_instruction_valid():
    df = make_episode_df()

    tasks = {
        0: {
            "instruction": "pick up the red block",
        }
    }

    results = check_task_instruction(
        df,
        tasks,
    )

    assert results == []


def test_task_instruction_missing_task():
    df = make_episode_df()

    tasks = {}

    results = check_task_instruction(
        df,
        tasks,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"


def test_task_instruction_short_text_is_valid():
    df = make_episode_df()

    tasks = {
        0: {
            "instruction": "pick block",
        }
    }

    results = check_task_instruction(
        df,
        tasks,
    )

    assert results == []


def test_task_instruction_missing_column():
    df = make_episode_df()
    df = df.drop(
        columns=["task_index"]
    )

    results = check_task_instruction(
        df,
        {0: "pick up block"},
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"


# ============================================================
# Instruction match
# ============================================================

def test_instruction_match_without_vlm_is_skipped():
    frames = np.zeros(
        (5, 32, 32, 3),
        dtype=np.uint8,
    )

    results = check_instruction_match(
        frames,
        "pick up the object",
        vlm=None,
    )

    assert results == []


def test_instruction_match_vlm_passes():
    frames = np.zeros(
        (5, 32, 32, 3),
        dtype=np.uint8,
    )

    def vlm(_frames, _instruction):
        return True

    results = check_instruction_match(
        frames,
        "pick up the object",
        vlm=vlm,
    )

    assert results == []


def test_instruction_match_vlm_fails():
    frames = np.zeros(
        (5, 32, 32, 3),
        dtype=np.uint8,
    )

    def vlm(_frames, _instruction):
        return False

    results = check_instruction_match(
        frames,
        "pick up the object",
        vlm=vlm,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"


# ============================================================
# Proprio sync
# ============================================================

def test_proprio_sync_valid():
    timestamps = np.arange(
        10,
        dtype=np.float64,
    ) / 30.0

    results = check_proprio_sync(
        parquet_ts=timestamps,
        video_ts=timestamps,
        tol_s=0.01,
    )

    assert results == []


def test_proprio_sync_mismatch():
    parquet_ts = np.arange(
        10,
        dtype=np.float64,
    ) / 30.0

    video_ts = (
        parquet_ts
        + np.array(
            [0.0] * 5
            + [0.1] * 5
        )
    )

    results = check_proprio_sync(
        parquet_ts=parquet_ts,
        video_ts=video_ts,
        tol_s=0.01,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"


def test_proprio_sync_length_mismatch():
    results = check_proprio_sync(
        parquet_ts=np.arange(10),
        video_ts=np.arange(9),
        tol_s=0.01,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"


# ============================================================
# Phase 4 runner
# ============================================================

def test_run_episode_quality_checks_valid():
    n = 30

    states = np.zeros(
        (n, 3)
    )

    states[:, 0] = np.arange(n)

    actions = np.ones(
        (n, 3)
    )

    timestamps = np.arange(n) / 10.0

    df = make_episode_df(
        n_frames=n
    )

    tasks = {
        0: {
            "instruction": "move the object forward",
        }
    }

    results = run_episode_quality_checks(
        states=states,
        actions=actions,
        timestamps=timestamps,
        df=df,
        task_map=tasks,
        episode_metadata={
            "success": False,
        },
        dataset_median_duration=2.9,
        dataset_duration_mad=0.5,
        dataset_median_path=29.0,
        vel_eps=1e-3,
        max_idle_ratio=0.50,
        max_idle_seconds=2.0,
        duration_mad_z_thresh=3.5,
        path_ratio_thresh=2.0,
        require_success=True,
    )

    # Success metadata is intentionally ignored because this dataset
    # does not define a success flag.
    assert results == []


def test_run_episode_quality_checks_without_metadata():
    n = 30

    states = np.zeros(
        (n, 3)
    )

    states[:, 0] = np.arange(n)

    actions = np.ones(
        (n, 3)
    )

    timestamps = np.arange(n) / 10.0

    df = make_episode_df(
        n_frames=n
    )

    results = run_episode_quality_checks(
        states=states,
        actions=actions,
        timestamps=timestamps,
        df=df,
        task_map=None,
        episode_metadata=None,
        dataset_median_duration=2.9,
        dataset_duration_mad=0.5,
        dataset_median_path=29.0,
    )

    assert results == []
