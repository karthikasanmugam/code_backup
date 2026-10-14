from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dataset import column_to_numpy
from .models import CheckResult


# ============================================================
# Helpers
# ============================================================

def _find_episode_metadata(
    meta_dict: dict[Any, Any],
    ep_idx: int,
) -> dict[str, Any] | None:
    """
    Resolve episode metadata from several common key formats.

    Supported examples:
        {0: {"length": 100}}
        {"0": {"length": 100}}
        {"episode_0": {"length": 100}}
        {"episode_000000": {"length": 100}}
    """
    candidates = (
        ep_idx,
        str(ep_idx),
        f"episode_{ep_idx}",
        f"episode_{ep_idx:06d}",
    )

    for key in candidates:
        value = meta_dict.get(key)
        if isinstance(value, dict):
            return value

    return None


def _get_column_dimension(
    df: pd.DataFrame,
    column: str,
) -> tuple[int, ...] | None:
    """Return the vector shape represented by a dataframe column."""
    if column not in df.columns:
        return None

    if len(df) == 0:
        return None

    values = df[column].to_numpy()

    if values.size == 0:
        return None

    first = values[0]

    if isinstance(first, (list, tuple, np.ndarray)):
        try:
            return tuple(np.asarray(first).shape)
        except Exception:
            return None

    # Scalar feature.
    return ()


def _check_vector_column(
    df: pd.DataFrame,
    column: str,
    expected_dim: int,
    check_name: str,
) -> list[CheckResult]:
    """
    Verify every row of a vector column has the expected dimension.
    """
    if column not in df.columns:
        return [
            CheckResult(
                check=check_name,
                severity="HARD",
                message=f"Required column '{column}' is missing.",
            )
        ]

    try:
        values = df[column].to_numpy()
    except Exception as exc:
        return [
            CheckResult(
                check=check_name,
                severity="HARD",
                message=(
                    f"Could not read column '{column}': {exc}"
                ),
            )
        ]

    if len(values) == 0:
        return []

    expected_shape = (expected_dim,)

    bad_frames: list[int] = []

    for frame_idx, value in enumerate(values):
        try:
            shape = np.asarray(value).shape
        except Exception:
            bad_frames.append(frame_idx)
            continue

        if shape != expected_shape:
            bad_frames.append(frame_idx)

    if not bad_frames:
        return []

    return [
        CheckResult(
            check=check_name,
            severity="HARD",
            message=(
                f"Column '{column}' contains {len(bad_frames)} "
                f"rows with dimension different from expected "
                f"{expected_dim}."
            ),
            frame_range=(
                min(bad_frames),
                max(bad_frames) + 1,
            ),
        )
    ]


# ============================================================
# Phase 1 - Structural integrity
# ============================================================

def check_metadata_length(
    df: pd.DataFrame,
    ep_idx: int,
    meta_dict: dict[Any, Any],
) -> list[CheckResult]:
    """
    Verify parquet row count matches episode metadata length.

    Also flags the episode when metadata is missing.
    """
    episode_meta = _find_episode_metadata(
        meta_dict,
        ep_idx,
    )

    if episode_meta is None:
        return [
            CheckResult(
                check="metadata_length",
                severity="HARD",
                message=(
                    f"Episode {ep_idx} is missing from episode metadata."
                ),
            )
        ]

    if "length" not in episode_meta:
        return [
            CheckResult(
                check="metadata_length",
                severity="HARD",
                message=(
                    f"Episode {ep_idx} metadata does not contain "
                    "'length'."
                ),
            )
        ]

    try:
        expected_length = int(episode_meta["length"])
    except (TypeError, ValueError):
        return [
            CheckResult(
                check="metadata_length",
                severity="HARD",
                message=(
                    f"Episode {ep_idx} metadata has invalid length: "
                    f"{episode_meta['length']!r}"
                ),
            )
        ]

    actual_length = len(df)

    if actual_length == expected_length:
        return []

    return [
        CheckResult(
            check="metadata_length",
            severity="HARD",
            message=(
                f"Metadata length is {expected_length}, "
                f"but parquet contains {actual_length} rows."
            ),
        )
    ]


def check_frame_index(
    df: pd.DataFrame,
) -> list[CheckResult]:
    """
    Verify:
        frame_index == [0, 1, ..., n-1]
    """
    if "frame_index" not in df.columns:
        return [
            CheckResult(
                check="frame_index",
                severity="HARD",
                message="Required column 'frame_index' is missing.",
            )
        ]

    frame_index = np.asarray(
        df["frame_index"].to_numpy()
    )

    expected = np.arange(
        len(df),
        dtype=frame_index.dtype
        if np.issubdtype(frame_index.dtype, np.integer)
        else np.int64,
    )

    if np.array_equal(frame_index, expected):
        return []

    mismatch = np.flatnonzero(
        frame_index != expected
    )

    if len(mismatch) == 0:
        return []

    start = int(mismatch[0])
    end = int(mismatch[-1]) + 1

    return [
        CheckResult(
            check="frame_index",
            severity="HARD",
            message=(
                f"frame_index is not contiguous from 0 to "
                f"{len(df) - 1}."
            ),
            frame_range=(start, end),
        )
    ]


def check_required_columns(
    df: pd.DataFrame,
    required: list[str],
) -> list[CheckResult]:
    """
    Verify all required columns exist.
    """
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if not missing:
        return []

    return [
        CheckResult(
            check="required_columns",
            severity="HARD",
            message=(
                "Missing required columns: "
                + ", ".join(missing)
            ),
        )
    ]


def check_dimension_consistency(
    df: pd.DataFrame,
    expected_state_dim: int,
    expected_action_dim: int,
    state_column: str = "observation.state",
    action_column: str = "action",
) -> list[CheckResult]:
    """
    Verify state/action vector dimensions match info.json.

    Every row must contain vectors of the expected dimensions.
    """
    results: list[CheckResult] = []

    results.extend(
        _check_vector_column(
            df=df,
            column=state_column,
            expected_dim=expected_state_dim,
            check_name="state_dimension",
        )
    )

    results.extend(
        _check_vector_column(
            df=df,
            column=action_column,
            expected_dim=expected_action_dim,
            check_name="action_dimension",
        )
    )

    return results


def check_nan_inf(
    arrays: dict[str, np.ndarray],
) -> list[CheckResult]:
    """
    Verify supplied numeric arrays contain no NaN or Inf.
    """
    results: list[CheckResult] = []

    for name, array in arrays.items():
        values = np.asarray(array)

        if not np.issubdtype(values.dtype, np.number):
            continue

        bad_mask = ~np.isfinite(values)

        if not np.any(bad_mask):
            continue

        if values.ndim == 1:
            bad_indices = np.flatnonzero(bad_mask)
        else:
            bad_indices = np.flatnonzero(
                np.any(
                    bad_mask.reshape(
                        values.shape[0],
                        -1,
                    ),
                    axis=1,
                )
            )

        if len(bad_indices) == 0:
            continue

        results.append(
            CheckResult(
                check="nan_inf",
                severity="HARD",
                message=(
                    f"{name} contains {len(bad_indices)} "
                    "frames with NaN or Inf."
                ),
                frame_range=(
                    int(bad_indices.min()),
                    int(bad_indices.max()) + 1,
                ),
            )
        )

    return results


def check_episode_length(
    n_frames: int,
    min_len: int = 20,
    max_len: int = 1000,
) -> list[CheckResult]:
    """
    Verify episode length falls within the configured bounds.
    """
    if min_len < 0:
        raise ValueError(
            f"min_len must be >= 0, got {min_len}"
        )

    if max_len < min_len:
        raise ValueError(
            f"max_len must be >= min_len, got "
            f"{min_len}, {max_len}"
        )

    if min_len <= n_frames <= max_len:
        return []

    return [
        CheckResult(
            check="episode_length",
            severity="HARD",
            message=(
                f"Episode has {n_frames} frames; "
                f"expected {min_len} to {max_len}."
            ),
        )
    ]


# ============================================================
# Convenience function
# ============================================================

def run_structural_checks(
    df: pd.DataFrame,
    ep_idx: int,
    meta_dict: dict[Any, Any],
    required_columns: list[str],
    expected_state_dim: int,
    expected_action_dim: int,
    min_len: int = 20,
    max_len: int = 1000,
    state_column: str = "observation.state",
    action_column: str = "action",
) -> list[CheckResult]:
    """
    Run all Phase 1 structural checks for one episode.

    This is useful for the pipeline while each individual function
    remains independently testable.
    """
    results: list[CheckResult] = []

    results.extend(
        check_metadata_length(
            df=df,
            ep_idx=ep_idx,
            meta_dict=meta_dict,
        )
    )

    results.extend(
        check_frame_index(df)
    )

    results.extend(
        check_required_columns(
            df=df,
            required=required_columns,
        )
    )

    # Avoid duplicate-looking dimension errors when the required
    # columns themselves are missing.
    if (
        state_column in df.columns
        and action_column in df.columns
    ):
        results.extend(
            check_dimension_consistency(
                df=df,
                expected_state_dim=expected_state_dim,
                expected_action_dim=expected_action_dim,
                state_column=state_column,
                action_column=action_column,
            )
        )

        try:
            state = column_to_numpy(
                df,
                state_column,
            )

            action = column_to_numpy(
                df,
                action_column,
            )

            results.extend(
                check_nan_inf(
                    {
                        state_column: state,
                        action_column: action,
                    }
                )
            )
        except (KeyError, ValueError):
            # Dimension/column checks already report the structural
            # problem. Do not duplicate it here.
            pass

    results.extend(
        check_episode_length(
            n_frames=len(df),
            min_len=min_len,
            max_len=max_len,
        )
    )

    return results