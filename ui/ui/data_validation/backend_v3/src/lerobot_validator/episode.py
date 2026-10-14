from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .models import CheckResult


# ============================================================
# Helpers
# ============================================================

def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)

    if array.ndim == 1:
        return array.reshape(-1, 1)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 1D or 2D, got shape {array.shape}"
        )

    return array.astype(np.float64, copy=False)


def _validate_matching_lengths(
    states: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = _as_2d(states, "states")
    actions = _as_2d(actions, "actions")

    if len(states) != len(actions):
        raise ValueError(
            f"states and actions must have the same length; "
            f"got {len(states)} and {len(actions)}"
        )

    return states, actions


def _safe_duration(
    timestamps: np.ndarray,
) -> float | None:
    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    if len(timestamps) < 2:
        return None

    duration = float(
        timestamps[-1] - timestamps[0]
    )

    if not np.isfinite(duration):
        return None

    return duration


def _mad_zscore(
    value: float,
    median: float,
    mad: float,
) -> float:
    if mad <= np.finfo(float).eps:
        return 0.0 if value == median else float("inf")

    return 0.67448975 * (
        value - median
    ) / mad


# ============================================================
# Idle analysis
# ============================================================

def compute_idle_ratio(
    states: np.ndarray,
    actions: np.ndarray,
    vel_eps: float = 1e-3,
) -> float:
    """
    Compute fraction of steps considered idle.

    A step is idle when both state motion and action magnitude
    are below the configured threshold.
    """
    states, actions = _validate_matching_lengths(
        states,
        actions,
    )

    if vel_eps < 0:
        raise ValueError(
            f"vel_eps must be >= 0, got {vel_eps}"
        )

    if len(states) == 0:
        return 0.0

    state_delta = np.zeros(
        len(states),
        dtype=np.float64,
    )

    if len(states) >= 2:
        state_delta[1:] = np.linalg.norm(
            np.diff(states, axis=0),
            axis=1,
        )

    action_mag = np.linalg.norm(
        actions,
        axis=1,
    )

    idle = (
        state_delta <= vel_eps
        ) & (
        action_mag <= vel_eps
    )

    return float(
        np.mean(idle)
    )


def find_idle_intervals(
    states: np.ndarray,
    actions: np.ndarray,
    vel_eps: float = 1e-3,
) -> list[tuple[int, int]]:
    """Return contiguous idle frame intervals."""
    states, actions = _validate_matching_lengths(
        states,
        actions,
    )

    if vel_eps < 0:
        raise ValueError(
            f"vel_eps must be >= 0, got {vel_eps}"
        )

    if len(states) == 0:
        return []

    state_delta = np.zeros(
        len(states),
        dtype=np.float64,
    )

    if len(states) >= 2:
        state_delta[1:] = np.linalg.norm(
            np.diff(states, axis=0),
            axis=1,
        )

    action_mag = np.linalg.norm(
        actions,
        axis=1,
    )

    idle = (
        state_delta <= vel_eps
    ) & (
        action_mag <= vel_eps
    )

    intervals: list[tuple[int, int]] = []
    start: int | None = None

    for index, value in enumerate(idle):
        if value and start is None:
            start = index

        elif not value and start is not None:
            intervals.append(
                (start, index)
            )
            start = None

    if start is not None:
        intervals.append(
            (start, len(idle))
        )

    return intervals


def check_idle_ratio(
    states: np.ndarray,
    actions: np.ndarray,
    vel_eps: float = 1e-3,
    max_idle_ratio: float = 0.50,
) -> list[CheckResult]:
    """
    SOFT check for excessive total idle time.

    Unlike a long-stuck detector, this also captures short pauses.
    """
    if not 0.0 <= max_idle_ratio <= 1.0:
        raise ValueError(
            "max_idle_ratio must be between 0 and 1"
        )

    ratio = compute_idle_ratio(
        states,
        actions,
        vel_eps,
    )

    if ratio <= max_idle_ratio:
        return []

    return [
        CheckResult(
            check="idle_ratio",
            severity="SOFT",
            message=(
                f"Episode idle ratio is {ratio:.2%}; "
                f"maximum allowed is {max_idle_ratio:.2%}."
            ),
        )
    ]


# ============================================================
# Leading/trailing idle
# ============================================================

def find_leading_trailing_idle(
    states: np.ndarray,
    timestamps: np.ndarray,
    vel_eps: float = 1e-3,
) -> tuple[int, int]:
    """
    Return trim indices:
        (first_non_idle, last_non_idle_exclusive)

    This does not itself flag the episode.
    """
    states = _as_2d(
        states,
        "states",
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    if len(states) != len(timestamps):
        raise ValueError(
            "states and timestamps must have the same length"
        )

    n = len(states)

    if n == 0:
        return 0, 0

    motion = np.zeros(
        n,
        dtype=np.float64,
    )

    if n >= 2:
        motion[1:] = np.linalg.norm(
            np.diff(states, axis=0),
            axis=1,
        )

    active = motion > vel_eps

    active_indices = np.flatnonzero(active)

    if len(active_indices) == 0:
        return 0, 0

    first_active = int(active_indices[0])
    last_active = int(active_indices[-1]) + 1

    return first_active, last_active


def check_leading_trailing_idle(
    states: np.ndarray,
    timestamps: np.ndarray,
    vel_eps: float = 1e-3,
    max_idle_seconds: float = 2.0,
) -> list[CheckResult]:
    """
    SOFT check for long idle regions at either episode boundary.
    """
    if max_idle_seconds < 0:
        raise ValueError(
            "max_idle_seconds must be >= 0"
        )

    states = _as_2d(
        states,
        "states",
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    if len(states) != len(timestamps):
        raise ValueError(
            "states and timestamps must have the same length"
        )

    if len(states) == 0:
        return []

    first_active, last_active = find_leading_trailing_idle(
        states,
        timestamps,
        vel_eps,
    )

    # Entire episode is idle.
    if first_active == 0 and last_active == 0:
        duration = _safe_duration(timestamps)

        if duration is not None and duration > max_idle_seconds:
            return [
                CheckResult(
                    check="leading_trailing_idle",
                    severity="SOFT",
                    message=(
                        f"Entire episode is idle for approximately "
                        f"{duration:.2f}s."
                    ),
                    frame_range=(0, len(states)),
                )
            ]

        return []

    results: list[CheckResult] = []

    leading_seconds = float(
        timestamps[first_active] - timestamps[0]
    ) if first_active > 0 else 0.0

    trailing_seconds = float(
        timestamps[-1] - timestamps[last_active - 1]
    ) if last_active < len(states) else 0.0

    if leading_seconds > max_idle_seconds:
        results.append(
            CheckResult(
                check="leading_trailing_idle",
                severity="SOFT",
                message=(
                    f"Leading idle period is "
                    f"{leading_seconds:.2f}s; "
                    f"maximum allowed is {max_idle_seconds:.2f}s."
                ),
                frame_range=(0, first_active),
            )
        )

    if trailing_seconds > max_idle_seconds:
        results.append(
            CheckResult(
                check="leading_trailing_idle",
                severity="SOFT",
                message=(
                    f"Trailing idle period is "
                    f"{trailing_seconds:.2f}s; "
                    f"maximum allowed is {max_idle_seconds:.2f}s."
                ),
                frame_range=(
                    last_active,
                    len(states),
                ),
            )
        )

    return results


# ============================================================
# Duration outliers
# ============================================================

def check_duration_outlier(
    duration: float,
    dataset_median: float,
    mad: float | None = None,
    mad_z_thresh: float = 3.5,
) -> list[CheckResult]:
    """
    Flag an episode whose duration is unusually far from the
    dataset median.

    If MAD is unavailable/zero, use a relative-deviation fallback.
    """
    if duration < 0:
        raise ValueError(
            f"duration must be >= 0, got {duration}"
        )

    if dataset_median < 0:
        raise ValueError(
            f"dataset_median must be >= 0, got {dataset_median}"
        )

    if mad_z_thresh <= 0:
        raise ValueError(
            "mad_z_thresh must be > 0"
        )

    if not np.isfinite(duration):
        return [
            CheckResult(
                check="duration_outlier",
                severity="SOFT",
                message="Episode duration is not finite.",
            )
        ]

    if not np.isfinite(dataset_median):
        return []

    if mad is not None:
        if mad < 0:
            raise ValueError(
                "mad must be >= 0"
            )

        z = _mad_zscore(
            duration,
            dataset_median,
            mad,
        )

        if abs(z) <= mad_z_thresh:
            return []

        return [
            CheckResult(
                check="duration_outlier",
                severity="SOFT",
                message=(
                    f"Duration {duration:.2f}s is a MAD-z-score of "
                    f"{z:.2f} relative to dataset median "
                    f"{dataset_median:.2f}s."
                ),
            )
        ]

    # Fallback when MAD is not available.
    if dataset_median == 0:
        return []

    relative_difference = (
        abs(duration - dataset_median)
        / dataset_median
    )

    # 100% deviation is used only as a conservative fallback.
    if relative_difference <= 1.0:
        return []

    return [
        CheckResult(
            check="duration_outlier",
            severity="SOFT",
            message=(
                f"Duration {duration:.2f}s differs from dataset "
                f"median {dataset_median:.2f}s by "
                f"{relative_difference:.2%}."
            ),
        )
    ]


# ============================================================
# Path efficiency
# ============================================================

def compute_path_length(
    states: np.ndarray,
    position_dims: list[int] | None = None,
) -> float:
    """
    Compute trajectory path length from selected state dimensions.
    """
    states = _as_2d(
        states,
        "states",
    )

    if len(states) < 2:
        return 0.0

    if position_dims is None:
        selected = states
    else:
        invalid = [
            dim
            for dim in position_dims
            if dim < 0 or dim >= states.shape[1]
        ]

        if invalid:
            raise ValueError(
                f"Position dimensions out of range: {invalid}"
            )

        selected = states[:, position_dims]

    return float(
        np.sum(
            np.linalg.norm(
                np.diff(selected, axis=0),
                axis=1,
            )
        )
    )


def compute_displacement(
    states: np.ndarray,
    position_dims: list[int] | None = None,
) -> float:
    """Compute straight-line start-to-end displacement."""
    states = _as_2d(
        states,
        "states",
    )

    if len(states) < 2:
        return 0.0

    if position_dims is None:
        selected = states
    else:
        invalid = [
            dim
            for dim in position_dims
            if dim < 0 or dim >= states.shape[1]
        ]

        if invalid:
            raise ValueError(
                f"Position dimensions out of range: {invalid}"
            )

        selected = states[:, position_dims]

    return float(
        np.linalg.norm(
            selected[-1] - selected[0]
        )
    )


def check_path_efficiency(
    states: np.ndarray,
    dataset_median_path: float,
    ratio_thresh: float = 2.0,
    position_dims: list[int] | None = None,
) -> list[CheckResult]:
    """
    Flag trajectories whose path length is unusually large relative
    to the dataset median.

    This captures excessive wandering/retries without assuming a
    particular robot state layout.
    """
    if dataset_median_path < 0:
        raise ValueError(
            "dataset_median_path must be >= 0"
        )

    if ratio_thresh <= 0:
        raise ValueError(
            "ratio_thresh must be > 0"
        )

    path_length = compute_path_length(
        states,
        position_dims,
    )

    if dataset_median_path <= np.finfo(float).eps:
        return []

    ratio = (
        path_length
        / dataset_median_path
    )

    if ratio <= ratio_thresh:
        return []

    return [
        CheckResult(
            check="path_efficiency",
            severity="SOFT",
            message=(
                f"Trajectory path length is {path_length:.4f}, "
                f"which is {ratio:.2f}x the dataset median "
                f"{dataset_median_path:.4f}."
            ),
        )
    ]


# ============================================================
# Task instruction
# ============================================================

def check_task_instruction(
    df: pd.DataFrame,
    tasks_map: dict[Any, Any],
    task_index_column: str = "task_index",
) -> list[CheckResult]:
    """
    Verify that task_index exists, resolves in tasks_map and maps
    to a non-empty instruction.
    """
    if task_index_column not in df.columns:
        return [
            CheckResult(
                check="task_instruction",
                severity="HARD",
                message=(
                    f"Required column '{task_index_column}' is missing."
                ),
            )
        ]

    if len(df) == 0:
        return []

    values = df[task_index_column].to_numpy()

    unique_indices = list(
        dict.fromkeys(values.tolist())
    )

    results: list[CheckResult] = []

    for task_index in unique_indices:
        task = tasks_map.get(task_index)

        if task is None:
            task = tasks_map.get(
                str(task_index)
            )

        if task is None:
            results.append(
                CheckResult(
                    check="task_instruction",
                    severity="HARD",
                    message=(
                        f"task_index {task_index!r} does not "
                        "resolve in the task map."
                    ),
                )
            )
            continue

        if isinstance(task, dict):
            instruction = (
                task.get("instruction")
                or task.get("task")
                or task.get("text")
            )
        else:
            instruction = task

        if not isinstance(instruction, str):
            results.append(
                CheckResult(
                    check="task_instruction",
                    severity="HARD",
                    message=(
                        f"task_index {task_index!r} resolves to a "
                        "non-text instruction."
                    ),
                )
            )
            continue

        instruction = instruction.strip()

        if not instruction:
            results.append(
                CheckResult(
                    check="task_instruction",
                    severity="HARD",
                    message=(
                        f"task_index {task_index!r} has an empty "
                        "instruction."
                    ),
                )
            )
            continue

    return results


# ============================================================
# Instruction/video semantic match
# ============================================================

def check_instruction_match(
    frames: np.ndarray,
    instruction: str,
    vlm: Callable[[np.ndarray, str], bool | dict[str, Any]] | None = None,
) -> list[CheckResult]:
    """
    Optional semantic video/instruction verification.

    The VLM callable is intentionally generic.

    Expected return:
        True
        False
        {"match": True/False}
    """
    if not isinstance(instruction, str) or not instruction.strip():
        return [
            CheckResult(
                check="instruction_match",
                severity="SOFT",
                message="Instruction is empty or invalid.",
            )
        ]

    if vlm is None:
        return []

    frames = np.asarray(frames)

    if frames.size == 0:
        return []

    try:
        result = vlm(
            frames,
            instruction,
        )
    except Exception as exc:
        return [
            CheckResult(
                check="instruction_match",
                severity="SOFT",
                message=f"VLM instruction check failed: {exc}",
            )
        ]

    if isinstance(result, dict):
        matched = bool(
            result.get("match", False)
        )
    else:
        matched = bool(result)

    if matched:
        return []

    return [
        CheckResult(
            check="instruction_match",
            severity="SOFT",
            message=(
                "Video was not judged to match the task instruction."
            ),
        )
    ]


# ============================================================
# Proprioception/video synchronization
# ============================================================

def check_proprio_sync(
    parquet_ts: np.ndarray,
    video_ts: np.ndarray,
    tol_s: float,
) -> list[CheckResult]:
    """
    Verify state/proprioception timestamps align with video timestamps.

    Both streams must have the same number of timestamps.
    """
    if tol_s < 0:
        raise ValueError(
            "tol_s must be >= 0"
        )

    parquet_ts = np.asarray(
        parquet_ts,
        dtype=np.float64,
    )

    video_ts = np.asarray(
        video_ts,
        dtype=np.float64,
    )

    if len(parquet_ts) != len(video_ts):
        return [
            CheckResult(
                check="proprio_sync",
                severity="HARD",
                message=(
                    f"Proprioception stream has {len(parquet_ts)} "
                    f"timestamps while video has {len(video_ts)}."
                ),
            )
        ]

    if len(parquet_ts) == 0:
        return []

    parquet_relative = (
        parquet_ts - parquet_ts[0]
    )

    video_relative = (
        video_ts - video_ts[0]
    )

    difference = np.abs(
        parquet_relative - video_relative
    )

    bad = np.flatnonzero(
        difference > tol_s
    )

    if len(bad) == 0:
        return []

    return [
        CheckResult(
            check="proprio_sync",
            severity="HARD",
            message=(
                f"{len(bad)} timestamp(s) differ from video by "
                f"more than {tol_s:.6f}s."
            ),
            frame_range=(
                int(bad.min()),
                int(bad.max()) + 1,
            ),
        )
    ]


# ============================================================
# Convenience runner
# ============================================================

def run_episode_quality_checks(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    df: pd.DataFrame,
    task_map: dict[Any, Any] | None = None,
    episode_metadata: dict[str, Any] | None = None,
    dataset_median_duration: float | None = None,
    dataset_duration_mad: float | None = None,
    dataset_median_path: float | None = None,
    vel_eps: float = 1e-3,
    max_idle_ratio: float = 0.50,
    max_idle_seconds: float = 2.0,
    duration_mad_z_thresh: float = 3.5,
    path_ratio_thresh: float = 2.0,
    task_index_column: str = "task_index",
    require_success: bool = True,
    video_timestamps: np.ndarray | None = None,
    proprio_sync_tol_s: float | None = None,
    position_dims: list[int] | None = None,
) -> list[CheckResult]:
    """
    Run the deterministic Phase 4 checks.

    Optional semantic instruction checking is intentionally left out
    because it requires video frames and an external VLM.

    ``episode_metadata`` and ``require_success`` are retained only
    for temporary API compatibility with the current validator.
    This dataset has no success field, so they are intentionally ignored.
    """
    # These parameters remain temporarily so validator.py can be updated
    # in its own step without breaking this module first.
    _ = episode_metadata
    _ = require_success

    results: list[CheckResult] = []

    # Idle ratio.
    results.extend(
        check_idle_ratio(
            states=states,
            actions=actions,
            vel_eps=vel_eps,
            max_idle_ratio=max_idle_ratio,
        )
    )

    # Episode boundary idle.
    results.extend(
        check_leading_trailing_idle(
            states=states,
            timestamps=timestamps,
            vel_eps=vel_eps,
            max_idle_seconds=max_idle_seconds,
        )
    )

    # Duration outlier.
    if dataset_median_duration is not None:
        duration = _safe_duration(timestamps)

        if duration is not None:
            results.extend(
                check_duration_outlier(
                    duration=duration,
                    dataset_median=dataset_median_duration,
                    mad=dataset_duration_mad,
                    mad_z_thresh=duration_mad_z_thresh,
                )
            )

    # Path efficiency.
    if dataset_median_path is not None:
        results.extend(
            check_path_efficiency(
                states=states,
                dataset_median_path=dataset_median_path,
                ratio_thresh=path_ratio_thresh,
                position_dims=position_dims,
            )
        )

    # Task instruction.
    if task_map is not None:
        results.extend(
            check_task_instruction(
                df=df,
                tasks_map=task_map,
                task_index_column=task_index_column,
            )
        )

    # Proprioception/video timestamp alignment.
    if (
        video_timestamps is not None
        and proprio_sync_tol_s is not None
    ):
        results.extend(
            check_proprio_sync(
                parquet_ts=timestamps,
                video_ts=video_timestamps,
                tol_s=proprio_sync_tol_s,
            )
        )

    return results
