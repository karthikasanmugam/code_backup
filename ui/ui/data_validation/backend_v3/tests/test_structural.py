from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lerobot_validator.structural import (
    check_dimension_consistency,
    check_episode_length,
    check_frame_index,
    check_metadata_length,
    check_nan_inf,
    check_required_columns,
    run_structural_checks,
)


def make_dataframe(
    n_frames: int = 30,
    state_dim: int = 7,
    action_dim: int = 7,
) -> pd.DataFrame:
    """Create a valid dummy episode dataframe."""
    return pd.DataFrame(
        {
            "frame_index": np.arange(n_frames),
            "timestamp": np.arange(n_frames) / 30.0,
            "observation.state": [
                np.zeros(
                    state_dim,
                    dtype=np.float32,
                )
                for _ in range(n_frames)
            ],
            "action": [
                np.ones(
                    action_dim,
                    dtype=np.float32,
                )
                for _ in range(n_frames)
            ],
            "task_index": np.zeros(
                n_frames,
                dtype=np.int64,
            ),
        }
    )


# ============================================================
# check_metadata_length
# ============================================================

def test_metadata_length_valid():
    df = make_dataframe(n_frames=30)

    metadata = {
        5: {
            "length": 30,
        }
    }

    results = check_metadata_length(
        df=df,
        ep_idx=5,
        meta_dict=metadata,
    )

    assert results == []


def test_metadata_length_mismatch():
    df = make_dataframe(n_frames=30)

    metadata = {
        5: {
            "length": 40,
        }
    }

    results = check_metadata_length(
        df=df,
        ep_idx=5,
        meta_dict=metadata,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"
    assert results[0].check == "metadata_length"


def test_metadata_episode_missing():
    df = make_dataframe(n_frames=30)

    results = check_metadata_length(
        df=df,
        ep_idx=5,
        meta_dict={},
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"
    assert "missing" in results[0].message.lower()


def test_metadata_string_episode_key():
    df = make_dataframe(n_frames=30)

    metadata = {
        "5": {
            "length": 30,
        }
    }

    results = check_metadata_length(
        df=df,
        ep_idx=5,
        meta_dict=metadata,
    )

    assert results == []


def test_metadata_padded_episode_key():
    df = make_dataframe(n_frames=30)

    metadata = {
        "episode_000005": {
            "length": 30,
        }
    }

    results = check_metadata_length(
        df=df,
        ep_idx=5,
        meta_dict=metadata,
    )

    assert results == []


# ============================================================
# check_frame_index
# ============================================================

def test_frame_index_valid():
    df = make_dataframe(n_frames=30)

    results = check_frame_index(df)

    assert results == []


def test_frame_index_starts_at_one():
    df = make_dataframe(n_frames=30)
    df["frame_index"] = np.arange(1, 31)

    results = check_frame_index(df)

    assert len(results) == 1
    assert results[0].severity == "HARD"


def test_frame_index_has_gap():
    df = make_dataframe(n_frames=30)
    df.loc[10, "frame_index"] = 50

    results = check_frame_index(df)

    assert len(results) == 1
    assert results[0].check == "frame_index"
    assert results[0].frame_range is not None


def test_frame_index_column_missing():
    df = make_dataframe(n_frames=30)
    df = df.drop(columns=["frame_index"])

    results = check_frame_index(df)

    assert len(results) == 1
    assert results[0].severity == "HARD"


# ============================================================
# check_required_columns
# ============================================================

def test_required_columns_valid():
    df = make_dataframe()

    required = [
        "frame_index",
        "timestamp",
        "observation.state",
        "action",
        "task_index",
    ]

    results = check_required_columns(
        df=df,
        required=required,
    )

    assert results == []


def test_required_columns_detects_missing():
    df = make_dataframe()
    df = df.drop(columns=["action", "task_index"])

    required = [
        "frame_index",
        "timestamp",
        "observation.state",
        "action",
        "task_index",
    ]

    results = check_required_columns(
        df=df,
        required=required,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"
    assert "action" in results[0].message
    assert "task_index" in results[0].message


# ============================================================
# check_dimension_consistency
# ============================================================

def test_dimensions_valid():
    df = make_dataframe(
        state_dim=7,
        action_dim=8,
    )

    results = check_dimension_consistency(
        df=df,
        expected_state_dim=7,
        expected_action_dim=8,
    )

    assert results == []


def test_state_dimension_mismatch():
    df = make_dataframe(
        state_dim=6,
        action_dim=8,
    )

    results = check_dimension_consistency(
        df=df,
        expected_state_dim=7,
        expected_action_dim=8,
    )

    assert len(results) == 1
    assert results[0].check == "state_dimension"
    assert results[0].severity == "HARD"


def test_action_dimension_mismatch():
    df = make_dataframe(
        state_dim=7,
        action_dim=6,
    )

    results = check_dimension_consistency(
        df=df,
        expected_state_dim=7,
        expected_action_dim=8,
    )

    assert len(results) == 1
    assert results[0].check == "action_dimension"
    assert results[0].severity == "HARD"


def test_both_dimensions_mismatch():
    df = make_dataframe(
        state_dim=5,
        action_dim=6,
    )

    results = check_dimension_consistency(
        df=df,
        expected_state_dim=7,
        expected_action_dim=8,
    )

    assert len(results) == 2

    checks = {
        result.check
        for result in results
    }

    assert checks == {
        "state_dimension",
        "action_dimension",
    }


def test_missing_state_column():
    df = make_dataframe()
    df = df.drop(columns=["observation.state"])

    results = check_dimension_consistency(
        df=df,
        expected_state_dim=7,
        expected_action_dim=7,
    )

    assert any(
        result.check == "state_dimension"
        for result in results
    )


# ============================================================
# check_nan_inf
# ============================================================

def test_nan_inf_valid():
    states = np.zeros(
        (10, 7),
        dtype=np.float32,
    )

    actions = np.ones(
        (10, 7),
        dtype=np.float32,
    )

    results = check_nan_inf(
        {
            "state": states,
            "action": actions,
        }
    )

    assert results == []


def test_nan_detected():
    states = np.zeros(
        (10, 7),
        dtype=np.float32,
    )

    states[5, 2] = np.nan

    results = check_nan_inf(
        {
            "state": states,
        }
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"
    assert results[0].check == "nan_inf"
    assert results[0].frame_range == (5, 6)


def test_inf_detected():
    actions = np.ones(
        (10, 7),
        dtype=np.float32,
    )

    actions[3, 4] = np.inf

    results = check_nan_inf(
        {
            "action": actions,
        }
    )

    assert len(results) == 1
    assert results[0].frame_range == (3, 4)


def test_multiple_arrays_can_fail():
    states = np.zeros(
        (10, 7),
        dtype=np.float32,
    )

    actions = np.ones(
        (10, 7),
        dtype=np.float32,
    )

    states[2, 0] = np.nan
    actions[7, 1] = np.inf

    results = check_nan_inf(
        {
            "state": states,
            "action": actions,
        }
    )

    assert len(results) == 2


# ============================================================
# check_episode_length
# ============================================================

def test_episode_length_valid():
    assert check_episode_length(
        n_frames=100,
        min_len=20,
        max_len=1000,
    ) == []


def test_episode_length_too_short():
    results = check_episode_length(
        n_frames=10,
        min_len=20,
        max_len=1000,
    )

    assert len(results) == 1
    assert results[0].check == "episode_length"
    assert results[0].severity == "HARD"


def test_episode_length_too_long():
    results = check_episode_length(
        n_frames=1200,
        min_len=20,
        max_len=1000,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"


def test_episode_length_boundary_values_are_valid():
    assert check_episode_length(
        n_frames=20,
        min_len=20,
        max_len=1000,
    ) == []

    assert check_episode_length(
        n_frames=1000,
        min_len=20,
        max_len=1000,
    ) == []


def test_episode_length_invalid_configuration():
    with pytest.raises(ValueError):
        check_episode_length(
            n_frames=100,
            min_len=100,
            max_len=20,
        )


# ============================================================
# Integration of Phase 1
# ============================================================

def test_run_structural_checks_valid_episode():
    df = make_dataframe(
        n_frames=30,
        state_dim=7,
        action_dim=7,
    )

    metadata = {
        0: {
            "length": 30,
        }
    }

    results = run_structural_checks(
        df=df,
        ep_idx=0,
        meta_dict=metadata,
        required_columns=[
            "frame_index",
            "timestamp",
            "observation.state",
            "action",
            "task_index",
        ],
        expected_state_dim=7,
        expected_action_dim=7,
        min_len=20,
        max_len=1000,
    )

    assert results == []


def test_run_structural_checks_collects_multiple_failures():
    df = make_dataframe(
        n_frames=10,
        state_dim=6,
        action_dim=5,
    )

    df.loc[5, "frame_index"] = 99
    df.loc[3, "action"][2] = np.nan

    metadata = {
        0: {
            "length": 20,
        }
    }

    results = run_structural_checks(
        df=df,
        ep_idx=0,
        meta_dict=metadata,
        required_columns=[
            "frame_index",
            "timestamp",
            "observation.state",
            "action",
            "task_index",
        ],
        expected_state_dim=7,
        expected_action_dim=7,
        min_len=20,
        max_len=1000,
    )

    assert len(results) >= 4

    checks = {
        result.check
        for result in results
    }

    assert "metadata_length" in checks
    assert "frame_index" in checks
    assert "state_dimension" in checks
    assert "action_dimension" in checks
    assert "episode_length" in checks