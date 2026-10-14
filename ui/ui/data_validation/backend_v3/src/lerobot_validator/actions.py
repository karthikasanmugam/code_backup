from __future__ import annotations

from typing import Any

import numpy as np

from .models import CheckResult


# ============================================================
# Helpers
# ============================================================

def _as_2d_array(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Convert an input array to shape (N, D).

    Scalar-per-frame arrays become (N, 1).
    """
    array = np.asarray(values)

    if array.ndim == 1:
        return array.reshape(-1, 1)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 1D or 2D, got shape {array.shape}"
        )

    return array


def _robust_zscore(
    values: np.ndarray,
) -> np.ndarray:
    """
    Robust z-score using median and MAD.

    MAD is preferred here because action jerk can contain genuine
    outliers that would distort a standard mean/std calculation.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return np.empty_like(values)

    median = np.median(values)
    mad = np.median(
        np.abs(values - median)
    )

    if mad < np.finfo(np.float64).eps:
        std = np.std(values)

        if std < np.finfo(np.float64).eps:
            return np.zeros_like(values)

        return (values - median) / std

    return 0.67448975 * (
        values - median
    ) / mad


def _normalize_excluded_dims(
    exclude_dims: list[int] | None,
    n_dims: int,
) -> set[int]:
    """Validate and normalize excluded action dimensions."""
    if not exclude_dims:
        return set()

    invalid = [
        dim
        for dim in exclude_dims
        if dim < 0 or dim >= n_dims
    ]

    if invalid:
        raise ValueError(
            f"Excluded dimensions out of range: {invalid}; "
            f"action dimension count is {n_dims}"
        )

    return set(exclude_dims)


# ============================================================
# Action range
# ============================================================

def check_action_range(
    actions: np.ndarray,
    stats: dict[str, np.ndarray],
    max_clip_ratio: float = 0.02,
) -> list[CheckResult]:
    """
    Check the fraction of action values outside dataset percentile
    bounds.

    stats must contain:
        "low":  shape (D,)
        "high": shape (D,)
    """
    if not 0.0 <= max_clip_ratio <= 1.0:
        raise ValueError(
            "max_clip_ratio must be between 0 and 1"
        )

    actions = _as_2d_array(
        actions,
        "actions",
    ).astype(np.float64, copy=False)

    if actions.size == 0:
        return []

    if "low" not in stats or "high" not in stats:
        raise ValueError(
            "stats must contain 'low' and 'high'"
        )

    low = np.asarray(
        stats["low"],
        dtype=np.float64,
    )

    high = np.asarray(
        stats["high"],
        dtype=np.float64,
    )

    if low.shape != (actions.shape[1],):
        raise ValueError(
            f"Low percentile shape {low.shape} does not match "
            f"action dimension {actions.shape[1]}"
        )

    if high.shape != (actions.shape[1],):
        raise ValueError(
            f"High percentile shape {high.shape} does not match "
            f"action dimension {actions.shape[1]}"
        )

    outside = (
        (actions < low)
        | (actions > high)
    )

    outside_ratio = (
        np.count_nonzero(outside)
        / actions.size
    )

    if outside_ratio <= max_clip_ratio:
        return []

    return [
        CheckResult(
            check="action_range",
            severity="HARD",
            message=(
                f"{outside_ratio:.2%} of action values are outside "
                f"the configured percentile bounds; allowed maximum "
                f"is {max_clip_ratio:.2%}."
            ),
        )
    ]


# ============================================================
# Action smoothness
# ============================================================

def check_action_smoothness(
    actions: np.ndarray,
    fps: float,
    jerk_z_thresh: float = 6.0,
    exclude_dims: list[int] | None = None,
) -> list[CheckResult]:
    """
    Detect large second-difference / jerk outliers.

    The second finite difference is scaled by fps^2, giving a
    discrete acceleration-like quantity. The robust z-score is
    then used to detect unusually sharp action changes.

    This is intentionally SOFT because legitimate manipulation
    actions can contain rapid changes.
    """
    if fps <= 0:
        raise ValueError(
            f"fps must be > 0, got {fps}"
        )

    if jerk_z_thresh <= 0:
        raise ValueError(
            f"jerk_z_thresh must be > 0, got {jerk_z_thresh}"
        )

    actions = _as_2d_array(
        actions,
        "actions",
    ).astype(np.float64, copy=False)

    if len(actions) < 3:
        return []

    excluded = _normalize_excluded_dims(
        exclude_dims,
        actions.shape[1],
    )

    dimensions = [
        dim
        for dim in range(actions.shape[1])
        if dim not in excluded
    ]

    if not dimensions:
        return []

    selected = actions[:, dimensions]

    second_diff = np.diff(
        selected,
        n=2,
        axis=0,
    ) * (fps ** 2)

    magnitude = np.linalg.norm(
        second_diff,
        axis=1,
    )

    z_scores = _robust_zscore(
        magnitude
    )

    bad = np.flatnonzero(
        np.abs(z_scores) > jerk_z_thresh
    )

    if len(bad) == 0:
        return []

    # np.diff(..., n=2) starts at original frame 2.
    start = int(bad.min()) + 2
    end = int(bad.max()) + 3

    return [
        CheckResult(
            check="action_smoothness",
            severity="SOFT",
            message=(
                f"Detected {len(bad)} unusually large "
                f"second-difference action transition(s)."
            ),
            frame_range=(start, end),
        )
    ]


# ============================================================
# Gripper transitions
# ============================================================

def _binary_gripper_state(
    gripper: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert continuous gripper values into open/closed states.

    State 1 = value >= threshold
    State 0 = value < threshold
    """
    return (
        np.asarray(gripper) >= threshold
    ).astype(np.int8)


def _run_lengths(
    states: np.ndarray,
) -> list[tuple[int, int, int]]:
    """
    Return contiguous state runs as:
        (start, end_exclusive, state)
    """
    states = np.asarray(states)

    if len(states) == 0:
        return []

    runs: list[tuple[int, int, int]] = []

    start = 0
    current = int(states[0])

    for index in range(1, len(states)):
        value = int(states[index])

        if value != current:
            runs.append(
                (start, index, current)
            )

            start = index
            current = value

    runs.append(
        (start, len(states), current)
    )

    return runs


def check_gripper_transitions(
    gripper: np.ndarray,
    open_close_thresh: float,
    max_toggles: int = 6,
    min_hold_frames: int = 5,
) -> list[CheckResult]:
    """
    Check gripper state transitions for chatter.

    A toggle is a transition between the two thresholded states.
    """
    if max_toggles < 0:
        raise ValueError(
            f"max_toggles must be >= 0, got {max_toggles}"
        )

    if min_hold_frames < 1:
        raise ValueError(
            f"min_hold_frames must be >= 1, got {min_hold_frames}"
        )

    gripper = np.asarray(
        gripper,
        dtype=np.float64,
    ).reshape(-1)

    if gripper.size == 0:
        return []

    states = _binary_gripper_state(
        gripper,
        open_close_thresh,
    )

    runs = _run_lengths(states)

    toggles = max(
        0,
        len(runs) - 1,
    )

    short_runs = [
        run
        for run in runs
        if (run[1] - run[0]) < min_hold_frames
    ]

    results: list[CheckResult] = []

    if toggles > max_toggles:
        results.append(
            CheckResult(
                check="gripper_transitions",
                severity="SOFT",
                message=(
                    f"Gripper toggled {toggles} times; "
                    f"maximum allowed is {max_toggles}."
                ),
            )
        )

    if short_runs:
        start = min(
            run[0]
            for run in short_runs
        )
        end = max(
            run[1]
            for run in short_runs
        )

        results.append(
            CheckResult(
                check="gripper_transitions",
                severity="SOFT",
                message=(
                    f"Detected {len(short_runs)} gripper state run(s) "
                    f"shorter than {min_hold_frames} frame(s)."
                ),
                frame_range=(start, end),
            )
        )

    return results


# ============================================================
# State/action lag
# ============================================================

def _mean_center(
    values: np.ndarray,
) -> np.ndarray:
    values = _as_2d_array(
        values,
        "values",
    ).astype(np.float64, copy=False)

    return values - np.mean(
        values,
        axis=0,
        keepdims=True,
    )


def _derivative(
    values: np.ndarray,
) -> np.ndarray:
    """
    First discrete difference along time.
    """
    values = _as_2d_array(
        values,
        "values",
    ).astype(np.float64, copy=False)

    if len(values) < 2:
        return np.empty(
            (0, values.shape[1]),
            dtype=np.float64,
        )

    return np.diff(
        values,
        axis=0,
    )


def _flatten_aligned(
    state_values: np.ndarray,
    action_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten feature dimensions while preserving frame alignment.
    """
    state_values = _as_2d_array(
        state_values,
        "states",
    )

    action_values = _as_2d_array(
        action_values,
        "actions",
    )

    n = min(
        len(state_values),
        len(action_values),
    )

    if n == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    state_flat = state_values[:n].reshape(-1)
    action_flat = action_values[:n].reshape(-1)

    return (
        state_flat.astype(np.float64, copy=False),
        action_flat.astype(np.float64, copy=False),
    )


def _lagged_correlation(
    states: np.ndarray,
    actions: np.ndarray,
    lag: int,
) -> float:
    """
    Correlation between action and state changes at a given lag.

    Positive lag means:
        action[t] -> state[t + lag]
    """
    states = _as_2d_array(
        states,
        "states",
    )

    actions = _as_2d_array(
        actions,
        "actions",
    )

    n = min(
        len(states),
        len(actions),
    )

    if n < 3:
        return float("nan")

    states = states[:n]
    actions = actions[:n]

    state_delta = _derivative(states)
    action_delta = _derivative(actions)

    # Derivative index k represents transition k -> k+1.
    if lag > 0:
        action_part = action_delta[:-lag]
        state_part = state_delta[lag:]
    elif lag < 0:
        shift = abs(lag)
        action_part = action_delta[shift:]
        state_part = state_delta[:-shift]
    else:
        action_part = action_delta
        state_part = state_delta

    if len(action_part) == 0:
        return float("nan")

    a, s = _flatten_aligned(
        action_part,
        state_part,
    )

    if len(a) < 2 or len(s) < 2:
        return float("nan")

    a = a - np.mean(a)
    s = s - np.mean(s)

    denom = np.linalg.norm(a) * np.linalg.norm(s)

    if denom <= np.finfo(np.float64).eps:
        return float("nan")

    return float(
        np.dot(a, s) / denom
    )


# ============================================================
# Action/state lag
# ============================================================

def check_action_state_lag(
    states: np.ndarray,
    actions: np.ndarray,
    max_lag: int = 5,
    min_corr: float = 0.5,
) -> list[CheckResult]:
    """
    Check whether state response is correlated with action at a
    small positive lag.

    This is SOFT because the physical relationship depends on the
    robot and control interface.
    """
    if max_lag < 0:
        raise ValueError(
            f"max_lag must be >= 0, got {max_lag}"
        )

    correlations = {
        lag: _lagged_correlation(
            states,
            actions,
            lag,
        )
        for lag in range(
            0,
            max_lag + 1,
        )
    }

    valid = {
        lag: corr
        for lag, corr in correlations.items()
        if np.isfinite(corr)
    }

    if not valid:
        return []

    best_lag = max(
        valid,
        key=valid.get,
    )

    best_corr = valid[best_lag]

    positive_lag_candidates = {
        lag: corr
        for lag, corr in valid.items()
        if lag > 0
    }

    positive_best = (
        max(
            positive_lag_candidates.values()
        )
        if positive_lag_candidates
        else float("-inf")
    )

    if (
        best_lag > 0
        and positive_best >= min_corr
    ):
        return []

    return [
        CheckResult(
            check="action_state_lag",
            severity="SOFT",
            message=(
                f"No sufficiently strong positive-lag "
                f"action/state correlation found. "
                f"Best lag={best_lag}, correlation={best_corr:.3f}; "
                f"required correlation >= {min_corr:.3f}."
            ),
        )
    ]


# ============================================================
# Action/state alignment offset
# ============================================================

def check_action_alignment_offset(
    states: np.ndarray,
    actions: np.ndarray,
) -> list[CheckResult]:
    """
    Compare correlations at lag -1, 0 and +1.

    Positive lag:
        action[t] compared with state[t + 1]

    A large improvement at +/-1 over lag 0 suggests a possible
    off-by-one alignment problem.
    """
    correlations = {
        lag: _lagged_correlation(
            states,
            actions,
            lag,
        )
        for lag in (-1, 0, 1)
    }

    valid = {
        lag: corr
        for lag, corr in correlations.items()
        if np.isfinite(corr)
    }

    if 0 not in valid:
        return []

    best_lag = max(
        valid,
        key=valid.get,
    )

    if best_lag == 0:
        return []

    best_corr = valid[best_lag]
    zero_corr = valid[0]

    improvement = best_corr - zero_corr

    # Require a meaningful difference rather than reacting to
    # tiny numerical variation.
    if improvement <= 0.10:
        return []

    direction = (
        "ahead"
        if best_lag < 0
        else "behind"
    )

    return [
        CheckResult(
            check="action_alignment_offset",
            severity="HARD",
            message=(
                f"Potential action/state off-by-one alignment: "
                f"best correlation occurs at lag {best_lag} "
                f"({best_corr:.3f}) versus lag 0 "
                f"({zero_corr:.3f}). State appears {direction} "
                f"relative to action."
            ),
        )
    ]


# ============================================================
# Token budget
# ============================================================

def check_token_budget(
    n_action_dims: int,
    chunk_len: int,
    tokens_per_dim: int,
    max_tokens: int,
) -> list[CheckResult]:
    """
    Verify the action representation fits within the model context.

    Current simple model:
        total_tokens =
            chunk_len * n_action_dims * tokens_per_dim
    """
    if n_action_dims <= 0:
        raise ValueError(
            "n_action_dims must be > 0"
        )

    if chunk_len <= 0:
        raise ValueError(
            "chunk_len must be > 0"
        )

    if tokens_per_dim <= 0:
        raise ValueError(
            "tokens_per_dim must be > 0"
        )

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be > 0"
        )

    total_tokens = (
        n_action_dims
        * chunk_len
        * tokens_per_dim
    )

    if total_tokens <= max_tokens:
        return []

    return [
        CheckResult(
            check="token_budget",
            severity="HARD",
            message=(
                f"Action chunk requires {total_tokens} tokens, "
                f"exceeding model context limit of {max_tokens}."
            ),
        )
    ]


# ============================================================
# Tokenizer roundtrip
# ============================================================

def _call_tokenizer_encode(
    tokenizer: Any,
    actions: np.ndarray,
) -> Any:
    """Support common encode APIs."""
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(actions)

    if callable(tokenizer):
        return tokenizer(actions)

    raise TypeError(
        "Tokenizer must provide .encode() or be callable"
    )


def _call_tokenizer_decode(
    tokenizer: Any,
    encoded: Any,
) -> np.ndarray:
    """Support common decode APIs."""
    if not hasattr(tokenizer, "decode"):
        raise TypeError(
            "Tokenizer must provide .decode()"
        )

    decoded = tokenizer.decode(encoded)

    return np.asarray(
        decoded,
        dtype=np.float64,
    )


def check_tokenizer_roundtrip(
    actions: np.ndarray,
    tokenizer: Any,
    max_recon_err: float,
) -> list[CheckResult]:
    """
    Encode then decode actions and verify reconstruction error.

    Skipping is represented by passing tokenizer=None.
    """
    if max_recon_err < 0:
        raise ValueError(
            "max_recon_err must be >= 0"
        )

    if tokenizer is None:
        return []

    actions = np.asarray(
        actions,
        dtype=np.float64,
    )

    if actions.size == 0:
        return []

    try:
        encoded = _call_tokenizer_encode(
            tokenizer,
            actions,
        )

        decoded = _call_tokenizer_decode(
            tokenizer,
            encoded,
        )
    except Exception as exc:
        return [
            CheckResult(
                check="tokenizer_roundtrip",
                severity="SOFT",
                message=(
                    f"Tokenizer roundtrip failed: {exc}"
                ),
            )
        ]

    if decoded.shape != actions.shape:
        return [
            CheckResult(
                check="tokenizer_roundtrip",
                severity="SOFT",
                message=(
                    f"Decoded action shape {decoded.shape} does not "
                    f"match original shape {actions.shape}."
                ),
            )
        ]

    errors = np.abs(
        decoded - actions
    )

    max_error = float(
        np.max(errors)
    )

    mean_error = float(
        np.mean(errors)
    )

    if max_error <= max_recon_err:
        return []

    bad = np.flatnonzero(
        np.any(
            errors > max_recon_err,
            axis=1,
        )
    ) if errors.ndim > 1 else np.flatnonzero(
        errors > max_recon_err
    )

    frame_range = None

    if len(bad):
        frame_range = (
            int(bad.min()),
            int(bad.max()) + 1,
        )

    return [
        CheckResult(
            check="tokenizer_roundtrip",
            severity="SOFT",
            message=(
                f"Maximum reconstruction error is {max_error:.6f} "
                f"(mean={mean_error:.6f}); allowed maximum is "
                f"{max_recon_err:.6f}."
            ),
            frame_range=frame_range,
        )
    ]