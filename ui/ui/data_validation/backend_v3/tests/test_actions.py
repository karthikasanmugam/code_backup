from __future__ import annotations

import numpy as np
import pytest

from lerobot_validator.actions import (
    check_action_alignment_offset,
    check_action_range,
    check_action_smoothness,
    check_action_state_lag,
    check_gripper_transitions,
    check_token_budget,
    check_tokenizer_roundtrip,
)

# ============================================================
# Helpers
# ============================================================

def make_actions(
    n_frames: int = 100,
    n_dims: int = 7,
) -> np.ndarray:
    return np.zeros(
        (n_frames, n_dims),
        dtype=np.float64,
    )


class IdentityTokenizer:
    def encode(self, actions):
        return np.asarray(actions)

    def decode(self, encoded):
        return np.asarray(encoded)


class LossyTokenizer:
    def __init__(self, error: float):
        self.error = error

    def encode(self, actions):
        return np.asarray(actions)

    def decode(self, encoded):
        return np.asarray(encoded) + self.error


# ============================================================
# Action range
# ============================================================

def test_action_range_valid():
    actions = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
        ]
    )

    stats = {
        "low": np.array([-1.0, -1.0]),
        "high": np.array([1.0, 1.0]),
    }

    results = check_action_range(
        actions,
        stats,
    )

    assert results == []


def test_action_range_flags_excessive_outliers():
    actions = np.zeros(
        (100, 2)
    )

    actions[:20, 0] = 10.0

    stats = {
        "low": np.array([-1.0, -1.0]),
        "high": np.array([1.0, 1.0]),
    }

    results = check_action_range(
        actions,
        stats,
        max_clip_ratio=0.02,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"
    assert results[0].check == "action_range"


def test_action_range_allows_small_fraction():
    actions = np.zeros(
        (100, 2)
    )

    actions[0, 0] = 10.0

    stats = {
        "low": np.array([-1.0, -1.0]),
        "high": np.array([1.0, 1.0]),
    }

    results = check_action_range(
        actions,
        stats,
        max_clip_ratio=0.02,
    )

    assert results == []


def test_action_range_requires_matching_dimension():
    actions = make_actions(10, 7)

    with pytest.raises(ValueError):
        check_action_range(
            actions,
            {
                "low": np.zeros(6),
                "high": np.ones(6),
            },
        )


# ============================================================
# Smoothness
# ============================================================

def test_action_smoothness_constant_signal():
    actions = np.zeros(
        (100, 3)
    )

    results = check_action_smoothness(
        actions,
        fps=30,
    )

    assert results == []


def test_action_smoothness_short_episode():
    actions = np.zeros(
        (2, 3)
    )

    results = check_action_smoothness(
        actions,
        fps=30,
    )

    assert results == []


def test_action_smoothness_invalid_excluded_dimension():
    actions = make_actions(20, 3)

    with pytest.raises(ValueError):
        check_action_smoothness(
            actions,
            fps=30,
            exclude_dims=[3],
        )


# ============================================================
# Gripper
# ============================================================

def test_gripper_transitions_valid():
    gripper = np.concatenate(
        [
            np.zeros(10),
            np.ones(10),
            np.zeros(10),
        ]
    )

    results = check_gripper_transitions(
        gripper=gripper,
        open_close_thresh=0.5,
        max_toggles=6,
        min_hold_frames=5,
    )

    assert results == []


def test_gripper_too_many_toggles():
    gripper = np.array(
        [
            0, 1,
            0, 1,
            0, 1,
            0, 1,
            0, 1,
            0, 1,
            0,
        ],
        dtype=float,
    )

    results = check_gripper_transitions(
        gripper=gripper,
        open_close_thresh=0.5,
        max_toggles=3,
        min_hold_frames=1,
    )

    assert any(
        result.severity == "SOFT"
        for result in results
    )


def test_gripper_chatter_detected():
    gripper = np.array(
        [
            0, 0, 0,
            1,
            0,
            1,
            1, 1,
        ],
        dtype=float,
    )

    results = check_gripper_transitions(
        gripper=gripper,
        open_close_thresh=0.5,
        max_toggles=10,
        min_hold_frames=3,
    )

    assert len(results) >= 1
    assert all(
        result.severity == "SOFT"
        for result in results
    )


# ============================================================
# State/action lag
# ============================================================

def test_action_state_lag_too_short_returns_no_failure():
    states = np.zeros((2, 3))
    actions = np.zeros((2, 3))

    results = check_action_state_lag(
        states,
        actions,
    )

    assert results == []


def test_action_state_lag_detects_missing_positive_correlation():
    rng = np.random.default_rng(42)

    actions = rng.normal(
        size=(100, 2)
    )

    states = rng.normal(
        size=(100, 2)
    )

    results = check_action_state_lag(
        states,
        actions,
        max_lag=5,
        min_corr=0.5,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"


def test_action_state_lag_rejects_negative_max_lag():
    with pytest.raises(ValueError):
        check_action_state_lag(
            np.zeros((10, 2)),
            np.zeros((10, 2)),
            max_lag=-1,
        )


# ============================================================
# Alignment
# ============================================================

def test_action_alignment_same_stream_passes():
    t = np.linspace(
        0,
        10,
        100,
    )

    actions = np.stack(
        [
            np.sin(t),
            np.cos(t),
        ],
        axis=1,
    )

    states = actions.copy()

    results = check_action_alignment_offset(
        states,
        actions,
    )

    assert results == []


def test_action_alignment_short_data_is_safe():
    states = np.zeros((2, 2))
    actions = np.zeros((2, 2))

    results = check_action_alignment_offset(
        states,
        actions,
    )

    assert results == []


# ============================================================
# Token budget
# ============================================================

def test_token_budget_within_limit():
    results = check_token_budget(
        n_action_dims=7,
        chunk_len=10,
        tokens_per_dim=2,
        max_tokens=200,
    )

    assert results == []


def test_token_budget_exceeds_limit():
    results = check_token_budget(
        n_action_dims=8,
        chunk_len=20,
        tokens_per_dim=2,
        max_tokens=100,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"
    assert "320" in results[0].message


def test_token_budget_invalid_arguments():
    with pytest.raises(ValueError):
        check_token_budget(
            n_action_dims=0,
            chunk_len=10,
            tokens_per_dim=2,
            max_tokens=100,
        )


# ============================================================
# Tokenizer roundtrip
# ============================================================

def test_tokenizer_roundtrip_identity():
    actions = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    results = check_tokenizer_roundtrip(
        actions,
        IdentityTokenizer(),
        max_recon_err=1e-6,
    )

    assert results == []


def test_tokenizer_roundtrip_lossy():
    actions = np.zeros(
        (10, 3)
    )

    results = check_tokenizer_roundtrip(
        actions,
        LossyTokenizer(error=0.1),
        max_recon_err=0.01,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"


def test_tokenizer_roundtrip_disabled():
    actions = make_actions()

    results = check_tokenizer_roundtrip(
        actions,
        tokenizer=None,
        max_recon_err=0.01,
    )

    assert results == []


def test_tokenizer_roundtrip_shape_mismatch():
    class BadTokenizer:
        def encode(self, actions):
            return actions

        def decode(self, encoded):
            return np.zeros(
                (len(encoded), 1)
            )

    actions = np.zeros(
        (10, 3)
    )

    results = check_tokenizer_roundtrip(
        actions,
        BadTokenizer(),
        max_recon_err=0.01,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"