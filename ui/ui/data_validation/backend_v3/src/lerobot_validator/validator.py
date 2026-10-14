from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from .scoring import score_and_classify_episode
from .actions import (
    check_action_alignment_offset,
    check_action_range,
    check_action_smoothness,
    check_action_state_lag,
    check_gripper_transitions,
    check_token_budget,
    check_tokenizer_roundtrip,
)
from .dataset import (
    compute_dataset_stats,
    extract_action,
    extract_state,
    extract_timestamps,
    load_dataset_info,
    load_raw_info,
    load_episode_dataframe,
    load_reference_action_percentiles,
    resolve_episode_paths,
)
from .dataset_quality import run_dataset_quality_checks
from .episode import run_episode_quality_checks
from .models import (
    CheckResult,
    DatasetInfo,
    DatasetStats,
    DatasetValidation,
    EpisodeData,
    EpisodeMetadata,
    EpisodePaths,
    EpisodeValidation,
    ValidationConfig,
)
from .structural import run_structural_checks
from .video import (
    check_frozen_video,
    extract_video_timestamps,
    run_video_checks,
    sample_video_frames,
)


# ============================================================
# Progress logging
# ============================================================

def _log_validation_progress(message: str) -> None:
    """Write timestamped progress directly to CLI/server stdout."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[validation {timestamp}] {message}", flush=True)


# ============================================================
# Generic config helpers
# ============================================================

def _section(
    config: ValidationConfig,
    name: str,
) -> dict[str, Any]:
    value = config.raw.get(name, {})

    if not isinstance(value, dict):
        return {}

    return value


def _enabled(
    config: ValidationConfig,
    section_name: str,
    default: bool = True,
) -> bool:
    section = _section(
        config,
        section_name,
    )

    value = section.get(
        "enabled",
        default,
    )

    return bool(value)


def _get(
    config: ValidationConfig,
    section_name: str,
    key: str,
    default: Any = None,
) -> Any:
    section = _section(
        config,
        section_name,
    )

    return section.get(
        key,
        default,
    )


def _nested_get(
    config: ValidationConfig,
    section_name: str,
    subsection: str,
    key: str,
    default: Any = None,
) -> Any:
    section = _section(
        config,
        section_name,
    )

    sub = section.get(
        subsection,
        {},
    )

    if not isinstance(sub, dict):
        return default

    return sub.get(
        key,
        default,
    )


# ============================================================
# HARD failure / phase helpers
# ============================================================

def has_hard_failure(
    results: list,
) -> bool:
    """Return True when any result has HARD severity."""
    return any(
        result.severity == "HARD"
        for result in results
    )


def extend_results(
    destination: list,
    source: list,
) -> bool:
    """
    Add results and return whether a HARD failure exists.
    """
    destination.extend(source)
    return has_hard_failure(source)


# ============================================================
# Dataset episode discovery
# ============================================================

def _extract_episode_index(
    path: Path,
) -> int | None:
    """
    Extract an episode index from common filename conventions.

    Examples:
        episode_000123.parquet
        episode-123.parquet
        000123.parquet
        ep_123.parquet
    """
    stem = path.stem.lower()

    patterns = (
        r"(?:episode|ep)[-_]?(\d+)",
        r"^(\d+)$",
        r".*?(\d+)$",
    )

    for pattern in patterns:
        match = re.fullmatch(
            pattern,
            stem,
        )

        if match:
            try:
                return int(
                    match.group(1)
                )
            except ValueError:
                pass

    return None


def discover_episode_indices(
    data_root: Path,
) -> list[int]:
    """
    Discover episode IDs dynamically from parquet files.

    No dataset root or fixed directory structure is hard-coded.
    """
    data_root = Path(data_root)

    raw = load_raw_info(data_root)
    data_template = raw.get("data_path")
    if isinstance(data_template, str) and "{episode_index" not in data_template:
        template = data_template
        for token in ("{chunk_index:03d}", "{episode_chunk:03d}", "{chunk_index}", "{episode_chunk}", "{file_index:03d}", "{file_index}"):
            template = template.replace(token, "*")
        chunk_files = sorted(data_root.glob(template))
        if chunk_files:
            indices: set[int] = set()
            for path in chunk_files:
                try:
                    values = pd.read_parquet(path, columns=["episode_index"])["episode_index"]
                    indices.update(int(value) for value in values.dropna().unique())
                except (KeyError, OSError, ValueError):
                    continue
            if indices:
                return sorted(indices)

    parquet_files = sorted(data_root.rglob("*.parquet"))

    indices: set[int] = set()

    for path in parquet_files:
        episode_index = _extract_episode_index(
            path
        )

        if episode_index is not None:
            indices.add(
                episode_index
            )

    return sorted(indices)


# ============================================================
# Dataset metadata loaders
# ============================================================

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file while ignoring blank lines."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

            if isinstance(value, dict):
                records.append(value)

    return records


def _record_episode_index(record: dict[str, Any]) -> int | None:
    """Extract an episode index from common episode metadata keys."""
    for key in (
        "episode_index",
        "episode_idx",
        "episode",
        "index",
    ):
        if key not in record:
            continue

        try:
            return int(record[key])
        except (TypeError, ValueError):
            continue

    return None


def _normalise_metadata_records(
    records: Iterable[dict[str, Any]],
) -> dict[Any, Any]:
    """
    Normalize episode metadata records to an index-keyed mapping.

    Both integer and string lookup are supported without requiring the
    caller to know which representation the source file used.
    """
    result: dict[Any, Any] = {}

    for record in records:
        ep_idx = _record_episode_index(record)

        if ep_idx is None:
            continue

        result[ep_idx] = record

    return result


def load_episode_metadata(
    data_root: Path,
) -> dict[Any, Any]:
    """
    Load per-episode metadata from this dataset's metadata files.

    Preferred source:
        meta/episodes/chunk-*/episodes.parquet

    Fallback:
        meta/episodes.jsonl

    An absent metadata source is represented by an empty mapping so the
    rest of validation can continue with checks supported by frame data.
    """
    data_root = Path(data_root)

    parquet_paths = sorted(
        (data_root / "meta" / "episodes").glob(
            "chunk-*/*.parquet"
        )
    )

    if parquet_paths:
        records: list[dict[str, Any]] = []

        for path in parquet_paths:
            df = pd.read_parquet(path)

            if len(df):
                records.extend(
                    df.to_dict(orient="records")
                )

        return _normalise_metadata_records(records)

    jsonl_path = data_root / "meta" / "episodes.jsonl"

    if jsonl_path.exists():
        return _normalise_metadata_records(
            _read_jsonl(jsonl_path)
        )

    return {}


def _task_index_from_record(record: dict[str, Any]) -> int | None:
    """Extract a task index/id from a task metadata record."""
    for key in (
        "task_index",
        "task_id",
        "id",
        "index",
    ):
        if key not in record:
            continue

        try:
            return int(record[key])
        except (TypeError, ValueError):
            continue

    return None


def _instruction_from_task_record(
    record: dict[str, Any],
) -> str | None:
    """Extract task text from common task metadata keys."""
    for key in (
        "instruction",
        "task",
        "text",
        "name",
    ):
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _normalise_task_records(
    records: Iterable[dict[str, Any]],
) -> dict[Any, Any]:
    """Normalize task records into an index -> instruction mapping."""
    result: dict[Any, Any] = {}

    for record in records:
        task_idx = _task_index_from_record(record)
        instruction = _instruction_from_task_record(record)

        if task_idx is None or instruction is None:
            continue

        result[task_idx] = instruction

    return result


def load_task_map(
    data_root: Path,
) -> dict[Any, Any]:
    """
    Load task instructions from this dataset's task metadata.

    Preferred source:
        meta/tasks.parquet

    Fallback:
        meta/tasks.jsonl
    """
    data_root = Path(data_root)

    parquet_path = data_root / "meta" / "tasks.parquet"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if not any(key in df.columns for key in ("task", "instruction", "text", "name")):
            df = df.reset_index()

        return _normalise_task_records(
            df.to_dict(orient="records")
        )

    jsonl_path = data_root / "meta" / "tasks.jsonl"

    if jsonl_path.exists():
        return _normalise_task_records(
            _read_jsonl(jsonl_path)
        )

    return {}


def _derive_task_map_from_dataframe(
    df: pd.DataFrame,
    task_index_column: str = "task_index",
    task_column: str = "task",
) -> dict[Any, Any]:
    """
    Fallback task mapping from the episode frame table itself.

    This is used only when task metadata files do not provide a mapping.
    """
    if (
        task_index_column not in df.columns
        or task_column not in df.columns
    ):
        return {}

    result: dict[Any, Any] = {}

    for task_idx, task_value in zip(
        df[task_index_column].tolist(),
        df[task_column].tolist(),
    ):
        try:
            task_id = int(task_idx)
        except (TypeError, ValueError):
            continue

        if not isinstance(task_value, str):
            continue

        instruction = task_value.strip()

        if not instruction:
            continue

        result[task_id] = instruction

    return result


def _missing_camera_results(
    expected_camera_keys: Iterable[str],
    video_paths: dict[str, Path],
) -> list[CheckResult]:
    """Return HARD results for dataset-declared cameras without files."""
    present = set(video_paths)
    results: list[CheckResult] = []

    for camera_key in expected_camera_keys:
        if camera_key in present:
            continue

        results.append(
            CheckResult(
                check="video_exists",
                severity="HARD",
                message=(
                    f"Video file for declared camera '{camera_key}' "
                    "could not be resolved."
                ),
            )
        )

    return results


# ============================================================
# Phase 0 - Dataset setup
# ============================================================

def setup_dataset(
    data_root: Path,
    config: ValidationConfig,
    metadata_map: dict[Any, Any] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> tuple[
    DatasetInfo,
    dict[int, Path],
    DatasetStats,
]:
    """
    Load dataset info, dynamically resolve episode parquet paths,
    and compute reusable dataset statistics.
    """
    data_root = Path(data_root)

    if cancel_check is not None:
        cancel_check()

    dataset_info = load_dataset_info(
        data_root
    )

    episode_indices = discover_episode_indices(
        data_root
    )

    configured_indices = _get(
        config,
        "dataset",
        "episode_indices",
        None,
    )

    if configured_indices is not None:
        episode_indices = [
            int(index)
            for index in configured_indices
            if int(index) in set(episode_indices)
        ]

    parquet_files: dict[int, Path] = {}

    for episode_index in episode_indices:
        if cancel_check is not None:
            cancel_check()
        try:
            paths = resolve_episode_paths(
                data_root=data_root,
                ep_idx=episode_index,
                camera_keys=[],
                episode_metadata=(metadata_map or {}).get(episode_index, {}),
            )

            parquet_files[episode_index] = (
                paths.parquet
            )
        except FileNotFoundError:
            # Discovery found an episode but the resolver cannot
            # locate it. The actual episode-level validator will
            # report what it can when invoked directly.
            continue

    stats_config = _section(
        config,
        "setup",
    )

    percentile_config = stats_config.get(
        "action_percentiles",
        {},
    )

    low = float(
        percentile_config.get(
            "low",
            1.0,
        )
    )

    high = float(
        percentile_config.get(
            "high",
            99.0,
        )
    )

    action_column = "action"

    if cancel_check is not None:
        cancel_check()

    stats = compute_dataset_stats(
        parquet_files=list(
            parquet_files.values()
        ),
        action_col=action_column,
        percentiles=(low, high),
        episode_indices=episode_indices,
        cancel_check=cancel_check,
    )

    if cancel_check is not None:
        cancel_check()

    reference_percentiles = load_reference_action_percentiles(
        data_root, (low, high)
    )
    if (
        reference_percentiles is not None
        and all(
            bounds.shape == (dataset_info.action_dim,)
            for bounds in reference_percentiles.values()
        )
    ):
        stats.action_percentiles = reference_percentiles

    return (
        dataset_info,
        parquet_files,
        stats,
    )


# ============================================================
# Phase 3 helper
# ============================================================

def run_action_checks(
    actions: np.ndarray,
    states: np.ndarray,
    dataset_stats: DatasetStats,
    dataset_info: DatasetInfo,
    config: ValidationConfig,
) -> list:
    """
    Run configured action/token checks.
    """
    results = []

    action_config = _section(
        config,
        "actions",
    )

    # --------------------------------------------------------
    # Action range
    # --------------------------------------------------------

    range_config = action_config.get(
        "range",
        {},
    )

    if (
        isinstance(range_config, dict)
        and range_config.get("enabled", True)
    ):
        action_percentiles = (
            dataset_stats.action_percentiles
        )

        if (
            "low" in action_percentiles
            and "high" in action_percentiles
        ):
            results.extend(
                check_action_range(
                    actions=actions,
                    stats=action_percentiles,
                    max_clip_ratio=float(
                        range_config.get(
                            "max_clip_ratio",
                            0.02,
                        )
                    ),
                )
            )

    # --------------------------------------------------------
    # Smoothness
    # --------------------------------------------------------

    smoothness = action_config.get(
        "smoothness",
        {},
    )

    if (
        isinstance(smoothness, dict)
        and smoothness.get("enabled", True)
    ):
        results.extend(
            check_action_smoothness(
                actions=actions,
                fps=dataset_info.fps,
                jerk_z_thresh=float(
                    smoothness.get(
                        "jerk_z_threshold",
                        6.0,
                    )
                ),
                exclude_dims=smoothness.get(
                    "exclude_dimensions",
                    [],
                ),
            )
        )

    # --------------------------------------------------------
    # Gripper
    # --------------------------------------------------------

    gripper = action_config.get(
        "gripper",
        {},
    )

    if (
        isinstance(gripper, dict)
        and gripper.get("enabled", True)
    ):
        threshold = gripper.get(
            "open_close_threshold"
        )

        # Do not guess the gripper dimension or threshold.
        # This must be explicitly configured for the custom dataset.
        gripper_dim = gripper.get(
            "dimension"
        )

        if (
            threshold is not None
            and gripper_dim is not None
        ):
            gripper_dim = int(
                gripper_dim
            )

            if not (
                0 <= gripper_dim < actions.shape[1]
            ):
                results.append(
                    # This is a configuration/data contract error.
                    # It is not silently ignored.
                    __import__(
                        "lerobot_validator.models",
                        fromlist=["CheckResult"],
                    ).CheckResult(
                        check="gripper_transitions",
                        severity="HARD",
                        message=(
                            f"Configured gripper dimension "
                            f"{gripper_dim} is outside action "
                            f"dimension {actions.shape[1]}."
                        ),
                    )
                )
            else:
                results.extend(
                    check_gripper_transitions(
                        gripper=actions[:, gripper_dim],
                        open_close_thresh=float(
                            threshold
                        ),
                        max_toggles=int(
                            gripper.get(
                                "max_toggles",
                                6,
                            )
                        ),
                        min_hold_frames=int(
                            gripper.get(
                                "min_hold_frames",
                                5,
                            )
                        ),
                    )
                )

    # --------------------------------------------------------
    # State/action lag
    # --------------------------------------------------------

    lag = action_config.get(
        "state_lag",
        {},
    )

    if (
        isinstance(lag, dict)
        and lag.get("enabled", True)
    ):
        results.extend(
            check_action_state_lag(
                states=states,
                actions=actions,
                max_lag=int(
                    lag.get(
                        "max_lag_frames",
                        5,
                    )
                ),
                min_corr=float(
                    lag.get(
                        "min_correlation",
                        0.5,
                    )
                ),
            )
        )

    # --------------------------------------------------------
    # Alignment offset
    # --------------------------------------------------------

    alignment = action_config.get(
        "alignment",
        {},
    )

    if (
        isinstance(alignment, dict)
        and alignment.get("enabled", True)
    ):
        results.extend(
            check_action_alignment_offset(
                states=states,
                actions=actions,
            )
        )

    # --------------------------------------------------------
    # Token budget
    # --------------------------------------------------------

    token_budget = action_config.get(
        "token_budget",
        {},
    )

    if (
        isinstance(token_budget, dict)
        and token_budget.get("enabled", True)
    ):
        chunk_len = token_budget.get(
            "chunk_length"
        )
        tokens_per_dim = token_budget.get(
            "tokens_per_dimension"
        )
        max_tokens = token_budget.get(
            "max_tokens"
        )

        if (
            chunk_len is not None
            and tokens_per_dim is not None
            and max_tokens is not None
        ):
            results.extend(
                check_token_budget(
                    n_action_dims=dataset_info.action_dim,
                    chunk_len=int(chunk_len),
                    tokens_per_dim=int(tokens_per_dim),
                    max_tokens=int(max_tokens),
                )
            )

    return results


# ============================================================
# Phase 2 frozen-video helper
# ============================================================

def _run_frozen_video_checks(
    video_paths: dict[str, Path],
    video_ranges: dict[str, tuple[float, float]],
    actions: np.ndarray,
    config: ValidationConfig,
    fps: float,
    sampled_video: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list:
    """
    Sample video frames and compare the corresponding action rows.

    Sampling uses the original frame indices returned by the video
    sampler, so we do not assume all frames were loaded into memory.
    """
    frozen_config = _section(
        config,
        "video",
    ).get(
        "frozen_video",
        {},
    )

    if not isinstance(
        frozen_config,
        dict,
    ):
        return []

    if not frozen_config.get(
        "enabled",
        True,
    ):
        return []

    results = []

    diff_eps = float(
        frozen_config.get(
            "diff_epsilon",
            1e-3,
        )
    )

    act_eps = float(
        frozen_config.get(
            "action_epsilon",
            0.03,
        )
    )

    min_run_s = float(
        frozen_config.get(
            "min_run_seconds",
            2.0,
        )
    )

    sample_every = int(
        _get(
            config,
            "video",
            "image_quality",
            {},
        ).get(
            "sample_every",
            15,
        )
        if isinstance(
            _get(
                config,
                "video",
                "image_quality",
                {},
            ),
            dict,
        )
        else 15
    )

    for camera_key, video_path in video_paths.items():
        try:
            if sampled_video is not None and camera_key in sampled_video:
                frames, indices = sampled_video[camera_key]
            else:
                time_range = video_ranges.get(camera_key)
                frames, indices = sample_video_frames(
                    video_path=video_path,
                    sample_every=sample_every,
                    start_time_s=time_range[0] if time_range else None,
                    end_time_s=time_range[1] if time_range else None,
                )
        except Exception:
            continue

        if len(frames) < 2:
            continue

        # We can only compare with action rows when the video index
        # falls inside the parquet action stream.
        valid = indices < len(actions)

        if not np.all(valid):
            continue

        sampled_actions = actions[
            indices
        ]

        camera_results = check_frozen_video(
            frames=frames,
            actions=sampled_actions,
            diff_eps=diff_eps,
            act_eps=act_eps,
            min_run_s=min_run_s,
            fps=fps / sample_every,
        )

        for result in camera_results:
            result.message = (
                f"[{camera_key}] {result.message}"
            )

        results.extend(
            camera_results
        )

    return results


# ============================================================
# Single episode validation
# ============================================================
def _resolve_episode_metadata(
    metadata_map: dict[Any, Any] | None,
    ep_idx: int,
) -> dict[str, Any]:
    """
    Resolve metadata for one episode without assuming one exact
    metadata-file representation.
    """
    if not metadata_map:
        return {}

    candidates = (
        ep_idx,
        str(ep_idx),
        f"episode_{ep_idx}",
        f"episode_{ep_idx:06d}",
    )

    for key in candidates:
        value = metadata_map.get(key)

        if isinstance(value, dict):
            return value

    return {}


def validate_episode(
    ep_idx: int,
    data_root: Path,
    dataset_info: DatasetInfo,
    stats: DatasetStats,
    config: ValidationConfig,
    episode_metadata: dict[str, Any] | None = None,
    metadata_map: dict[Any, Any] | None = None,
    tasks_map: dict[Any, Any] | None = None,
    tokenizer: Any | None = None,
    instruction_vlm: Callable | None = None,
    camera_coverage_detector: Callable | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> EpisodeValidation:
    """
    Validate one episode through Phases 1-4.

    HARD failure in an early phase short-circuits later expensive
    phases.
    """
    results = []
    if cancel_check is not None:
        cancel_check()
    episode_meta = episode_metadata or _resolve_episode_metadata(metadata_map, ep_idx)

    # --------------------------------------------------------
    # Resolve files dynamically.
    # --------------------------------------------------------

    paths = resolve_episode_paths(
        data_root=data_root,
        ep_idx=ep_idx,
        camera_keys=dataset_info.camera_keys,
        episode_metadata=episode_meta,
    )

    # --------------------------------------------------------
    # Load parquet.
    # --------------------------------------------------------

    try:
        df = load_episode_dataframe(
            paths.parquet,
            episode_index=ep_idx,
        )
    except Exception as exc:
        from .models import CheckResult

        return EpisodeValidation(
            episode_index=ep_idx,
            results=[
                CheckResult(
                    check="episode_load",
                    severity="HARD",
                    message=str(exc),
                )
            ],
        )

    if cancel_check is not None:
        cancel_check()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    # Structural metadata checks need the full episode-indexed map.
    structural_metadata_map = metadata_map or {}

    if not structural_metadata_map and episode_meta:
        structural_metadata_map = {
            ep_idx: episode_meta,
            str(ep_idx): episode_meta,
        }

    # Use task metadata supplied by the caller when present. Otherwise
    # fall back to task text stored directly in the episode dataframe.
    effective_tasks_map = tasks_map

    # --------------------------------------------------------
    # Phase 1 - Structural
    # --------------------------------------------------------

    structural_config = _section(
        config,
        "structural",
    )

    required_columns = structural_config.get(
        "required_columns",
        [
            "frame_index",
            "timestamp",
            "observation.state",
            "action",
            "task_index",
        ],
    )

    if cancel_check is not None:
        cancel_check()

    results.extend(
        run_structural_checks(
            df=df,
            ep_idx=ep_idx,
            meta_dict=structural_metadata_map,
            required_columns=required_columns,
            expected_state_dim=dataset_info.state_dim,
            expected_action_dim=dataset_info.action_dim,
            min_len=int(
                structural_config.get(
                    "min_episode_length",
                    20,
                )
            ),
            max_len=int(
                structural_config.get(
                    "max_episode_length",
                    1000,
                )
            ),
        )
    )

    if cancel_check is not None:
        cancel_check()

    if (
        has_hard_failure(results)
        and config.skip_expensive_after_hard
    ):
        return EpisodeValidation(
            episode_index=ep_idx,
            results=results,
        )

    # --------------------------------------------------------
    # Load arrays after structural checks.
    # --------------------------------------------------------

    if effective_tasks_map is None:
        effective_tasks_map = _derive_task_map_from_dataframe(df)

    try:
        states = extract_state(df)
        actions = extract_action(df)
        timestamps = extract_timestamps(df)
    except Exception as exc:
        from .models import CheckResult

        results.append(
            CheckResult(
                check="feature_extraction",
                severity="HARD",
                message=str(exc),
            )
        )

        return EpisodeValidation(
            episode_index=ep_idx,
            results=results,
        )

    # --------------------------------------------------------
    # Phase 2 - Video / vision
    # --------------------------------------------------------

    if _enabled(
        config,
        "video",
        True,
    ):
        expected_resolutions = {
            camera_key: (
                camera.height,
                camera.width,
            )
            for camera_key, camera
            in dataset_info.cameras.items()
            if camera.height is not None
            and camera.width is not None
        }

        video_config = _section(
            config,
            "video",
        )

        image_quality_config = video_config.get(
            "image_quality",
            {},
        )
        video_samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        frozen_video_config = video_config.get("frozen_video", {})
        frozen_video_enabled = (
            bool(frozen_video_config.get("enabled", True))
            if isinstance(frozen_video_config, dict)
            else False
        )

        results.extend(
            _missing_camera_results(
                expected_camera_keys=dataset_info.camera_keys,
                video_paths=paths.videos,
            )
        )

        if has_hard_failure(results) and config.skip_expensive_after_hard:
            return EpisodeValidation(
                episode_index=ep_idx,
                results=results,
            )

        if cancel_check is not None:
            cancel_check()

        results.extend(
            run_video_checks(
                video_paths={
                    camera_key: path
                    for camera_key, path
                    in paths.videos.items()
                },
                n_rows=len(df),
                expected_resolutions=expected_resolutions,
                parquet_timestamps=timestamps,
                video_ranges=paths.video_ranges,
                fps=dataset_info.fps,
                sample_every=int(
                    image_quality_config.get(
                        "sample_every",
                        15,
                    )
                )
                if isinstance(
                    image_quality_config,
                    dict,
                )
                else 15,
                image_quality_enabled=bool(
                    image_quality_config.get(
                        "enabled",
                        True,
                    )
                )
                if isinstance(
                    image_quality_config,
                    dict,
                )
                else True,
                frozen_video_enabled=frozen_video_enabled,
                sampled_video_out=video_samples,
                multiview_sync_enabled=bool(
                    video_config.get(
                        "multiview_sync",
                        {},
                    ).get(
                        "enabled",
                        True,
                    )
                ),
                frame_count_tolerance=int(
                    video_config.get(
                        "frame_count_tolerance",
                        0,
                    )
                ),
                timestamp_tolerance_s=(
                    float(video_config["timestamp_tolerance_s"])
                    if video_config.get("timestamp_tolerance_s") is not None
                    else None
                ),
                multiview_tolerance_frames=int(
                    video_config.get(
                        "multiview_sync",
                        {},
                    ).get(
                        "tolerance_frames",
                        1,
                    )
                ),
                reference_camera=str(
                    video_config.get(
                        "multiview_sync",
                        {},
                    ).get(
                        "reference_camera",
                        "auto",
                    )
                ),
                camera_coverage_detector=camera_coverage_detector
                if bool(
                    video_config.get(
                        "camera_coverage",
                        {},
                    ).get(
                        "enabled",
                        False,
                    )
                )
                else None,
            )
        )

        if cancel_check is not None:
            cancel_check()

        # Frozen video requires both sampled video and actions.
        if not has_hard_failure(results):
            results.extend(
                _run_frozen_video_checks(
                    video_paths=paths.videos,
                    video_ranges=paths.video_ranges,
                    actions=actions,
                    config=config,
                    fps=dataset_info.fps,
                    sampled_video=video_samples,
                )
            )

    if (
        has_hard_failure(results)
        and config.skip_expensive_after_hard
    ):
        return EpisodeValidation(
            episode_index=ep_idx,
            results=results,
        )

    # --------------------------------------------------------
    # Phase 3 - Actions / tokens
    # --------------------------------------------------------

    if cancel_check is not None:
        cancel_check()

    if _enabled(
        config,
        "actions",
        True,
    ):
        results.extend(
            run_action_checks(
                actions=actions,
                states=states,
                dataset_stats=stats,
                dataset_info=dataset_info,
                config=config,
            )
        )

    if (
        has_hard_failure(results)
        and config.skip_expensive_after_hard
    ):
        return EpisodeValidation(
            episode_index=ep_idx,
            results=results,
        )

    # --------------------------------------------------------
    # Phase 3b - tokenizer roundtrip
    # --------------------------------------------------------

    if cancel_check is not None:
        cancel_check()

    tokenizer_config = _nested_get(
        config,
        "actions",
        "tokenizer_roundtrip",
        "enabled",
        False,
    )

    if (
        tokenizer_config
        and tokenizer is not None
    ):
        max_error = float(
            _nested_get(
                config,
                "actions",
                "tokenizer_roundtrip",
                "max_reconstruction_error",
                0.01,
            )
        )

        results.extend(
            check_tokenizer_roundtrip(
                actions=actions,
                tokenizer=tokenizer,
                max_recon_err=max_error,
            )
        )

    # --------------------------------------------------------
    # Phase 4 - Episode quality
    # --------------------------------------------------------

    if cancel_check is not None:
        cancel_check()

    reference_video_timestamps = None
    proprio_sync_tolerance = None

    if paths.videos:
        reference_camera_key = next(
            (
                key
                for key in dataset_info.camera_keys
                if key in paths.videos
            ),
            next(iter(paths.videos)),
        )

        try:
            video_range = paths.video_ranges.get(reference_camera_key)
            reference_video_timestamps = extract_video_timestamps(
                paths.videos[reference_camera_key],
                *video_range if video_range else (None, None),
            )

            proprio_config = _section(
                config,
                "episode",
            ).get(
                "proprio_sync",
                {},
            )

            configured_tolerance = (
                proprio_config.get("tolerance_seconds")
                if isinstance(proprio_config, dict)
                else None
            )

            proprio_sync_tolerance = (
                float(configured_tolerance)
                if configured_tolerance is not None
                else 0.5 / dataset_info.fps
            )
        except Exception:
            reference_video_timestamps = None
            proprio_sync_tolerance = None

    if _enabled(
        config,
        "episode",
        True,
    ):
        episode_config = _section(
            config,
            "episode",
        )

        idle_config = episode_config.get(
            "idle",
            {},
        )

        boundary_config = episode_config.get(
            "boundary_idle",
            {},
        )

        duration_config = episode_config.get(
            "duration",
            {},
        )

        path_config = episode_config.get(
            "path_efficiency",
            {},
        )

        proprio_config = episode_config.get(
            "proprio_sync",
            {},
        )

        results.extend(
            run_episode_quality_checks(
                states=states,
                actions=actions,
                timestamps=timestamps,
                df=df,
                task_map=effective_tasks_map,
                episode_metadata=episode_meta,
                dataset_median_duration=(
                    stats.median_duration
                ),
                dataset_duration_mad=(
                    stats.extra.get(
                        "duration_mad"
                    )
                ),
                dataset_median_path=(
                    stats.median_path_length
                ),
                vel_eps=float(
                    idle_config.get(
                        "velocity_epsilon",
                        1e-3,
                    )
                ),
                max_idle_ratio=float(
                    idle_config.get(
                        "max_idle_ratio",
                        0.50,
                    )
                ),
                max_idle_seconds=float(
                    boundary_config.get(
                        "max_idle_seconds",
                        2.0,
                    )
                ),
                duration_mad_z_thresh=float(
                    duration_config.get(
                        "mad_z_threshold",
                        3.5,
                    )
                ),
                path_ratio_thresh=float(
                    path_config.get(
                        "ratio_threshold",
                        2.0,
                    )
                ),
                video_timestamps=reference_video_timestamps,
                proprio_sync_tol_s=proprio_sync_tolerance,
            )
        )

    # --------------------------------------------------------
    # Optional instruction/VLM check
    # --------------------------------------------------------

    if cancel_check is not None:
        cancel_check()

    if (
        instruction_vlm is not None
        and effective_tasks_map is not None
        and len(paths.videos) > 0
    ):
        instruction = None

        if "task_index" in df.columns and len(df):
            task_index = df["task_index"].iloc[0]

            task_value = effective_tasks_map.get(
                task_index
            )

            if task_value is None:
                task_value = effective_tasks_map.get(
                    str(task_index)
                )

            if isinstance(task_value, dict):
                instruction = (
                    task_value.get("instruction")
                    or task_value.get("task")
                    or task_value.get("text")
                )
            elif isinstance(task_value, str):
                instruction = task_value

        if instruction:
            from .episode import check_instruction_match

            # Use the first available camera as the semantic sample.
            first_camera_path = next(
                iter(
                    paths.videos.values()
                )
            )

            try:
                first_camera_key = next(iter(paths.videos))
                video_range = paths.video_ranges.get(first_camera_key)
                frames, _ = sample_video_frames(
                    first_camera_path,
                    sample_every=15,
                    max_frames=32,
                    start_time_s=video_range[0] if video_range else None,
                    end_time_s=video_range[1] if video_range else None,
                )

                results.extend(
                    check_instruction_match(
                        frames=frames,
                        instruction=instruction,
                        vlm=instruction_vlm,
                    )
                )
            except Exception:
                pass

    if cancel_check is not None:
        cancel_check()

    return EpisodeValidation(
        episode_index=ep_idx,
        results=results,
    )


# ============================================================
# Dataset validation
# ============================================================

def validate_dataset(
    data_root: Path,
    config: ValidationConfig,
    metadata_map: dict[Any, Any] | None = None,
    tasks_map: dict[Any, Any] | None = None,
    tokenizer: Any | None = None,
    instruction_vlm: Callable | None = None,
    camera_coverage_detector: Callable | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> DatasetValidation:
    """
    Validate the complete dataset.

    Dataset root is supplied dynamically at runtime.
    """
    data_root = Path(data_root)

    if cancel_check is not None:
        cancel_check()

    if metadata_map is None:
        metadata_map = load_episode_metadata(
            data_root
        )

    if tasks_map is None:
        tasks_map = load_task_map(
            data_root
        )

    (
        dataset_info,
        parquet_paths,
        stats,
    ) = setup_dataset(
        data_root=data_root,
        config=config,
        metadata_map=metadata_map,
        cancel_check=cancel_check,
    )

    if cancel_check is not None:
        cancel_check()

    episode_indices = sorted(
        parquet_paths.keys()
    )

    total_episodes = len(episode_indices)
    execution_config = _section(config, "execution")
    worker_count = max(1, int(execution_config.get("workers", 1)))
    _log_validation_progress(
        f"Starting {total_episodes} episode(s) in {data_root} "
        f"with {worker_count} worker(s)"
    )

    def validate_one(position: int, ep_idx: int) -> tuple[int, EpisodeValidation]:
        if cancel_check is not None:
            cancel_check()
        episode_started = time.perf_counter()
        _log_validation_progress(
            f"Episode {ep_idx} ({position}/{total_episodes}) started"
        )
        try:
            validation = validate_episode(
                ep_idx=ep_idx,
                data_root=data_root,
                dataset_info=dataset_info,
                stats=stats,
                config=config,
                episode_metadata=None,
                metadata_map=metadata_map,
                tasks_map=tasks_map,
                tokenizer=tokenizer,
                instruction_vlm=instruction_vlm,
                camera_coverage_detector=camera_coverage_detector,
                cancel_check=cancel_check,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - episode_started
            _log_validation_progress(
                f"Episode {ep_idx} ({position}/{total_episodes}) failed after "
                f"{elapsed:.1f}s: {type(exc).__name__}: {exc}"
            )
            raise

        hard_count = len(validation.hard_failures)
        soft_count = len(validation.soft_failures)
        elapsed = time.perf_counter() - episode_started
        _log_validation_progress(
            f"Episode {ep_idx} ({position}/{total_episodes}) completed in "
            f"{elapsed:.1f}s; hard={hard_count}, soft={soft_count}"
        )
        if progress_callback is not None:
            progress_callback({
                "event": "episode_completed",
                "episode_index": ep_idx,
                "position": position,
                "total_episodes": total_episodes,
                "hard_failures": hard_count,
                "soft_warnings": soft_count,
            })
        return position, validation

    results_by_position: dict[int, EpisodeValidation] = {}
    if worker_count == 1 or total_episodes < 2:
        for position, ep_idx in enumerate(episode_indices, start=1):
            result_position, validation = validate_one(position, ep_idx)
            results_by_position[result_position] = validation
    else:
        with ThreadPoolExecutor(
            max_workers=min(worker_count, total_episodes),
            thread_name_prefix="dataset-validator",
        ) as executor:
            futures = {
                executor.submit(validate_one, position, ep_idx): position
                for position, ep_idx in enumerate(episode_indices, start=1)
            }
            for future in as_completed(futures):
                position, validation = future.result()
                results_by_position[position] = validation

    # Keep output ordering deterministic even when workers finish out of order.
    episode_results = [
        results_by_position[position]
        for position in range(1, total_episodes + 1)
    ]

    # --------------------------------------------------------
    # Score and classify every episode
    # --------------------------------------------------------

    scoring_config = _section(
        config,
        "scoring",
    )

    weights = scoring_config.get(
        "weights",
        {},
    )

    classification_config = scoring_config.get(
        "classification",
        {},
    )

    min_score = float(
        classification_config.get(
            "review_min_score",
            0.70,
        )
    )

    reject_on_hard = bool(
        classification_config.get(
            "reject_on_hard",
            True,
        )
    )

    for episode in episode_results:
        if cancel_check is not None:
            cancel_check()
        score, classification = score_and_classify_episode(
            results=episode.results,
            weights=weights,
            min_score=min_score,
            reject_on_hard=reject_on_hard,
        )

        episode.score = score
        episode.classification = classification

    dataset_results = []

    # Phase 5 is intentionally optional until embeddings/tasks are
    # explicitly prepared by the caller.
    if _enabled(
        config,
        "dataset_quality",
        False,
    ):
        dataset_quality_config = _section(
            config,
            "dataset_quality",
        )

        # No embeddings are invented here.
        # A future embedding stage should populate these inputs.
        duplicate_config = dataset_quality_config.get(
            "duplicates",
            {},
        )

        task_balance_config = dataset_quality_config.get(
            "task_balance",
            {},
        )

        if (
            tasks_map is not None
            and isinstance(
                task_balance_config,
                dict,
            )
            and task_balance_config.get(
                "enabled",
                False,
            )
        ):
            task_counts = dict(stats.task_counts)

            dataset_results.extend(
                run_dataset_quality_checks(
                    task_counts=task_counts,
                    min_per_task=int(
                        task_balance_config.get(
                            "min_episodes_per_task",
                            1,
                        )
                    ),
                )
            )

    return DatasetValidation(
        total_episodes=len(episode_results),
        episodes=episode_results,
        dataset_results=dataset_results,
    )