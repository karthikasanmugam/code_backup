from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .models import (
    CameraInfo,
    DatasetInfo,
    DatasetStats,
    EpisodePaths,
    FeatureInfo,
)


# ============================================================
# Constants
# ============================================================

INFO_JSON_CANDIDATES = (
    "meta/info.json",
    "info.json",
)

PARQUET_ROOT_CANDIDATES = (
    "data",
    "parquet",
    "data/chunk-000",
)

VIDEO_ROOT_CANDIDATES = (
    "videos",
    "video",
    "videos/chunk-000",
)


@lru_cache(maxsize=2)
def _read_parquet_cached(
    path_string: str,
    modified_ns: int,
    file_size: int,
) -> pd.DataFrame:
    """Keep a small cache of chunk tables used across episode validation."""
    del modified_ns, file_size  # Included in the cache key for change invalidation.
    return pd.read_parquet(path_string)


@lru_cache(maxsize=4)
def _episode_row_indices_cached(
    path_string: str,
    modified_ns: int,
    file_size: int,
) -> dict[int, np.ndarray]:
    """Index episode rows once per parquet version instead of rescanning each chunk per episode."""
    frame = _read_parquet_cached(path_string, modified_ns, file_size)
    if "episode_index" not in frame.columns:
        return {}
    return {
        int(episode_index): np.asarray(positions, dtype=np.intp)
        for episode_index, positions in frame.groupby("episode_index", sort=False).indices.items()
    }


# ============================================================
# Generic helpers
# ============================================================

def _first_existing_path(
    data_root: Path,
    candidates: Iterable[str],
) -> Path | None:
    """Return the first existing candidate path."""
    for candidate in candidates:
        path = data_root / candidate
        if path.exists():
            return path
    return None


def _normalise_shape(value: Any) -> tuple[int, ...]:
    """Convert a JSON shape representation into a tuple of ints."""
    if value is None:
        return ()

    if isinstance(value, tuple):
        return tuple(int(x) for x in value)

    if isinstance(value, list):
        return tuple(int(x) for x in value)

    if isinstance(value, int):
        return (value,)

    return ()


def _feature_shape(feature: dict[str, Any]) -> tuple[int, ...]:
    """
    Extract a feature shape from several common representations.
    """
    if "shape" in feature:
        return _normalise_shape(feature["shape"])

    info = feature.get("info")

    if isinstance(info, dict) and "shape" in info:
        return _normalise_shape(info["shape"])

    return ()


def _feature_dtype(feature: dict[str, Any]) -> str | None:
    dtype = feature.get("dtype")

    if dtype is None:
        return None

    return str(dtype)


def _infer_camera_hw(
    feature: dict[str, Any],
) -> tuple[int | None, int | None]:
    """Infer camera height/width from dataset metadata.

    Prefer explicit ``names`` metadata, then support the common
    ``[H, W, C]`` and ``[C, H, W]`` conventions.
    """
    shape = _feature_shape(feature)

    if len(shape) != 3:
        return None, None

    names = feature.get("names")

    if isinstance(names, list) and len(names) == 3:
        lowered = [str(name).lower() for name in names]

        try:
            height_index = lowered.index("height")
            width_index = lowered.index("width")
            return shape[height_index], shape[width_index]
        except ValueError:
            pass

    # H, W, C
    if shape[-1] in (1, 3, 4):
        return shape[0], shape[1]

    # C, H, W
    if shape[0] in (1, 3, 4):
        return shape[1], shape[2]

    return None, None


def _camera_keys_from_info(
    raw: dict[str, Any],
    raw_features: dict[str, Any],
) -> list[str]:
    """Return dataset-declared camera keys, with feature fallback."""
    declared = raw.get("camera_keys")

    if isinstance(declared, list):
        keys = [
            str(key)
            for key in declared
            if str(key) in raw_features
            and isinstance(raw_features[str(key)], dict)
        ]

        if keys:
            return keys

    return [
        name
        for name, feature in raw_features.items()
        if isinstance(feature, dict)
        and _is_camera_feature(name, feature)
    ]


def _episode_chunk(
    raw: dict[str, Any],
    ep_idx: int,
) -> int:
    """Compute the dataset chunk number for an episode."""
    try:
        chunks_size = int(raw.get("chunks_size", 1024))
    except (TypeError, ValueError):
        chunks_size = 1024

    if chunks_size <= 0:
        chunks_size = 1024

    return int(ep_idx) // chunks_size


def _format_dataset_template(
    template: str,
    *,
    ep_idx: int,
    camera_key: str | None = None,
    chunks_size: int = 1024,
) -> Path:
    """Format a dataset path template from info.json."""
    episode_chunk = int(ep_idx) // int(chunks_size)

    values = {
        "episode_index": int(ep_idx),
        "episode_chunk": episode_chunk,
        "video_key": camera_key or "",
    }

    return Path(template.format(**values))


def _is_camera_feature(name: str, feature: dict[str, Any]) -> bool:
    """
    Detect image/video features without requiring one exact schema.

    We prefer explicit video/image metadata, then fall back to common
    observation.images naming.
    """
    dtype = str(feature.get("dtype", "")).lower()
    feature_type = str(
        feature.get("type", feature.get("modality", ""))
    ).lower()

    if feature_type in {"video", "image", "camera"}:
        return True

    if "video" in feature_type or "image" in feature_type:
        return True

    if name.startswith("observation.images."):
        return True

    # Typical uint8 image tensor: C,H,W.
    shape = _feature_shape(feature)

    if dtype in {"uint8", "image"} and len(shape) == 3:
        return True

    return False


def _infer_vector_dim(
    features: dict[str, FeatureInfo],
    candidates: Iterable[str],
) -> int:
    """Infer a vector feature dimension from candidate feature names."""
    for name in candidates:
        feature = features.get(name)

        if feature is None:
            continue

        if not feature.shape:
            continue

        if len(feature.shape) == 1:
            return feature.shape[0]

    return 0


# ============================================================
# info.json
# ============================================================

def find_info_json(data_root: Path) -> Path:
    """
    Locate info.json.

    The custom dataset may place info.json at the dataset root or
    under meta/.
    """
    data_root = Path(data_root)

    path = _first_existing_path(
        data_root,
        INFO_JSON_CANDIDATES,
    )

    if path is None:
        raise FileNotFoundError(
            f"Could not find info.json under dataset root: {data_root}"
        )

    return path


def load_raw_info(data_root: Path) -> dict[str, Any]:
    """Load the raw info.json dictionary."""
    info_path = find_info_json(data_root)

    try:
        with info_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {info_path}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected info.json to contain an object, got "
            f"{type(data).__name__}"
        )

    return data


def load_dataset_info(data_root: Path) -> DatasetInfo:
    """
    Read and normalize dataset information from info.json.

    Dataset-defined parameters such as FPS, camera keys and vector
    dimensions are taken from the dataset rather than hard-coded.
    """
    raw = load_raw_info(data_root)

    if "fps" not in raw:
        raise ValueError("info.json does not contain 'fps'")

    try:
        fps = float(raw["fps"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid fps value in info.json: {raw['fps']!r}"
        ) from exc

    if fps <= 0:
        raise ValueError(f"FPS must be > 0, got {fps}")

    raw_features = raw.get("features", {})

    if not isinstance(raw_features, dict):
        raise ValueError(
            "info.json 'features' must be a mapping"
        )

    features: dict[str, FeatureInfo] = {}

    for name, value in raw_features.items():
        if not isinstance(value, dict):
            continue

        features[name] = FeatureInfo(
            name=name,
            dtype=_feature_dtype(value),
            shape=_feature_shape(value),
            info=value,
        )

    cameras: dict[str, CameraInfo] = {}

    for name in _camera_keys_from_info(raw, raw_features):
        feature = raw_features.get(name)

        if not isinstance(feature, dict):
            continue

        height, width = _infer_camera_hw(feature)

        # CameraInfo uses its shape for spatial dimensions. Preserve the
        # original H/W/C shape in FeatureInfo above, while normalizing the
        # camera shape to (height, width) for resolution checks.
        camera_shape = (height, width) if height is not None and width is not None else ()

        cameras[name] = CameraInfo(
            key=name,
            dtype=_feature_dtype(feature),
            shape=camera_shape,
        )

    # Prefer the exact names used by your validator/data.
    # Fall back to the common names only when necessary.
    state_dim = _infer_vector_dim(
        features,
        (
            "observation.state",
            "state",
        ),
    )

    action_dim = _infer_vector_dim(
        features,
        (
            "action",
            "actions",
        ),
    )

    return DatasetInfo(
        fps=fps,
        state_dim=state_dim,
        action_dim=action_dim,
        features=features,
        cameras=cameras,
        raw=raw,
    )


# ============================================================
# Episode file resolution
# ============================================================

def _episode_tokens(ep_idx: int) -> list[str]:
    """
    Generate numeric forms commonly used in episode filenames.
    """
    return [
        str(ep_idx),
        f"{ep_idx:04d}",
        f"{ep_idx:05d}",
        f"{ep_idx:06d}",
    ]


def _matches_episode(path: Path, ep_idx: int) -> bool:
    """
    Return True when a path appears to refer to the requested episode.

    Matching is intentionally filename-based to avoid accidentally
    selecting a file belonging to another episode.
    """
    stem = path.stem.lower()

    tokens = _episode_tokens(ep_idx)

    patterns = [
        rf"(?:episode|ep)[-_]?{re.escape(token)}(?:\D|$)"
        for token in tokens
    ]

    if any(re.search(pattern, stem) for pattern in patterns):
        return True

    # Also support names such as:
    # 000123.parquet
    # episode_000123
    # 123.mp4
    return any(
        re.fullmatch(rf".*{re.escape(token)}", stem)
        for token in tokens
    )


def resolve_episode_parquet(
    data_root: Path,
    ep_idx: int,
    episode_metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Resolve the parquet file for one episode.

    When available, the dataset's own ``data_path`` and ``chunks_size``
    metadata are authoritative. Generic fallback resolution is retained
    for compatibility with earlier custom datasets.
    """
    data_root = Path(data_root)

    raw = load_raw_info(data_root)
    data_template = raw.get("data_path")

    if isinstance(data_template, str) and data_template:
        try:
            chunks_size = int(raw.get("chunks_size", 1024))
            if chunks_size <= 0:
                raise ValueError

            # LeRobot v3 uses chunk/file placeholders and stores multiple
            # episodes in each file. Resolve the containing chunk directly.
            if "{episode_index" not in data_template:
                metadata = episode_metadata or {}
                chunk_index = int(metadata.get("data/chunk_index", ep_idx // chunks_size))
                file_index = metadata.get("data/file_index")
                values = {
                    "episode_index": ep_idx,
                    "episode_chunk": chunk_index,
                    "chunk_index": chunk_index,
                    "file_index": int(file_index) if file_index is not None else 0,
                }
                try:
                    candidate = data_root / data_template.format(**values)
                    if candidate.exists():
                        return candidate
                except (KeyError, ValueError):
                    pass
                template = data_template
                for token in ("{chunk_index:03d}", "{episode_chunk:03d}"):
                    template = template.replace(token, f"{chunk_index:03d}")
                for token in ("{chunk_index}", "{episode_chunk}"):
                    template = template.replace(token, str(chunk_index))
                for token in ("{file_index:03d}", "{file_index}"):
                    template = template.replace(token, "*")
                matches = sorted(data_root.glob(template))
                for match in matches:
                    try:
                        found = pd.read_parquet(match, columns=["episode_index"])["episode_index"]
                        if int(ep_idx) in set(found.astype("int64").tolist()):
                            return match
                    except (KeyError, OSError, ValueError):
                        continue
            else:
                candidate = data_root / _format_dataset_template(
                    data_template,
                    ep_idx=ep_idx,
                    chunks_size=chunks_size,
                )
                if candidate.exists():
                    return candidate
        except (TypeError, ValueError, KeyError):
            pass

    candidate_roots = [
        data_root / relative
        for relative in PARQUET_ROOT_CANDIDATES
    ]

    for root in candidate_roots:
        if not root.exists():
            continue

        matches = [
            path
            for path in root.rglob("*.parquet")
            if _matches_episode(path, ep_idx)
        ]

        if matches:
            return sorted(matches)[0]

    matches = [
        path
        for path in data_root.rglob("*.parquet")
        if _matches_episode(path, ep_idx)
    ]

    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(
        f"Could not resolve parquet file for episode {ep_idx}"
    )


def _camera_filename_candidates(
    camera_key: str,
    ep_idx: int,
) -> list[str]:
    """
    Generate common filename patterns for a camera/episode.
    """
    clean_key = camera_key.replace(".", "_")
    tokens = _episode_tokens(ep_idx)

    candidates: list[str] = []

    for token in tokens:
        candidates.extend(
            [
                f"{camera_key}_{token}.mp4",
                f"{camera_key}-{token}.mp4",
                f"{clean_key}_{token}.mp4",
                f"{clean_key}-{token}.mp4",
                f"episode_{token}.mp4",
                f"episode-{token}.mp4",
                f"{token}.mp4",
            ]
        )

    return candidates


def resolve_episode_video(
    data_root: Path,
    ep_idx: int,
    camera_key: str,
    episode_metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Resolve one episode's video for a dataset-declared camera.

    The dataset's ``video_path`` template is tried first, followed by the
    existing generic fallbacks for compatibility.
    """
    data_root = Path(data_root)

    raw = load_raw_info(data_root)
    video_template = raw.get("video_path")

    if isinstance(video_template, str) and video_template:
        try:
            chunks_size = int(raw.get("chunks_size", 1024))
            if chunks_size <= 0:
                raise ValueError

            metadata = episode_metadata or {}
            prefix = f"videos/{camera_key}/"
            chunk_index = int(metadata.get(f"{prefix}chunk_index", ep_idx // chunks_size))
            file_index = metadata.get(f"{prefix}file_index")
            values = {
                "episode_index": ep_idx,
                "episode_chunk": chunk_index,
                "chunk_index": chunk_index,
                "file_index": int(file_index) if file_index is not None else 0,
                "video_key": camera_key,
            }
            candidate = data_root / video_template.format(**values)
            if candidate.exists():
                return candidate
        except (TypeError, ValueError, KeyError):
            pass

    candidate_roots = [
        data_root / relative
        for relative in VIDEO_ROOT_CANDIDATES
    ]

    filenames = _camera_filename_candidates(
        camera_key,
        ep_idx,
    )

    for root in candidate_roots:
        if not root.exists():
            continue

        for filename in filenames:
            matches = list(root.rglob(filename))

            if matches:
                return sorted(matches)[0]

    for root in candidate_roots:
        if not root.exists():
            continue

        matches = [
            path
            for path in root.rglob("*.mp4")
            if _matches_episode(path, ep_idx)
            and (
                camera_key in str(path)
                or camera_key.replace(".", "_") in str(path)
            )
        ]

        if matches:
            return sorted(matches)[0]

    for root in candidate_roots:
        if not root.exists():
            continue

        matches = [
            path
            for path in root.rglob("*.mp4")
            if _matches_episode(path, ep_idx)
            and camera_key.split(".")[-1].lower() in str(path).lower()
        ]

        if matches:
            return sorted(matches)[0]

    raise FileNotFoundError(
        f"Could not resolve video for episode {ep_idx}, "
        f"camera '{camera_key}'"
    )


def resolve_episode_paths(
    data_root: Path,
    ep_idx: int,
    camera_keys: list[str],
    episode_metadata: dict[str, Any] | None = None,
) -> EpisodePaths:
    """
    Resolve all files associated with an episode.
    """
    parquet = resolve_episode_parquet(
        data_root=data_root,
        ep_idx=ep_idx,
        episode_metadata=episode_metadata,
    )

    videos: dict[str, Path] = {}
    video_ranges: dict[str, tuple[float, float]] = {}

    for camera_key in camera_keys:
        try:
            videos[camera_key] = resolve_episode_video(
                data_root=data_root,
                ep_idx=ep_idx,
                camera_key=camera_key,
                episode_metadata=episode_metadata,
            )
            metadata = episode_metadata or {}
            prefix = f"videos/{camera_key}/"
            start = metadata.get(f"{prefix}from_timestamp")
            end = metadata.get(f"{prefix}to_timestamp")
            if start is not None and end is not None:
                video_ranges[camera_key] = (float(start), float(end))
        except (FileNotFoundError, KeyError, ValueError):
            # Missing camera files are intentionally left out here.
            # check_video_exists() will report the actual validation error.
            continue

    return EpisodePaths(
        episode_index=ep_idx,
        parquet=parquet,
        videos=videos,
        video_ranges=video_ranges,
    )


# ============================================================
# Parquet loading
# ============================================================

def load_episode_dataframe(
    parquet_path: Path,
    episode_index: int | None = None,
) -> pd.DataFrame:
    """Load one episode, including from a multi-episode v3 chunk parquet."""
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file does not exist: {parquet_path}"
        )

    if parquet_path.stat().st_size == 0:
        raise ValueError(
            f"Parquet file is empty: {parquet_path}"
        )

    try:
        stat = parquet_path.stat()
        dataframe = _read_parquet_cached(
            str(parquet_path.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
        )
        if episode_index is not None and "episode_index" in dataframe.columns:
            positions = _episode_row_indices_cached(
                str(parquet_path.resolve()), stat.st_mtime_ns, stat.st_size
            ).get(int(episode_index))
            dataframe = (
                dataframe.iloc[positions].copy()
                if positions is not None
                else dataframe.iloc[0:0].copy()
            )
            if "frame_index" in dataframe.columns:
                dataframe = dataframe.sort_values("frame_index", kind="stable")
            dataframe.reset_index(drop=True, inplace=True)
        return dataframe
    except Exception as exc:
        raise ValueError(
            f"Could not read parquet file: {parquet_path}"
        ) from exc


# ============================================================
# Feature extraction
# ============================================================

def column_to_numpy(
    df: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """
    Convert a dataframe feature column into a numeric ndarray.

    Handles:
        scalar columns
        lists/tuples
        numpy arrays
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataframe"
        )

    values = df[column].to_numpy()

    if len(values) == 0:
        return np.empty((0,), dtype=np.float32)

    first = values[0]

    if isinstance(first, (list, tuple, np.ndarray)):
        try:
            array = np.stack(
                [
                    np.asarray(value)
                    for value in values
                ]
            )
        except ValueError as exc:
            raise ValueError(
                f"Column '{column}' contains inconsistent "
                f"array shapes"
            ) from exc

        return array

    return np.asarray(values)


def extract_state(
    df: pd.DataFrame,
    state_column: str = "observation.state",
) -> np.ndarray:
    """Extract observation state from an episode dataframe."""
    return column_to_numpy(df, state_column)


def extract_action(
    df: pd.DataFrame,
    action_column: str = "action",
) -> np.ndarray:
    """Extract actions from an episode dataframe."""
    return column_to_numpy(df, action_column)


def extract_timestamps(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> np.ndarray:
    """Extract timestamps as float64 values."""
    values = column_to_numpy(
        df,
        timestamp_column,
    )

    return values.astype(np.float64, copy=False)


# ============================================================
# Dataset statistics
# ============================================================

def _stack_action_arrays(
    parquet_files: list[Path],
    action_col: str,
) -> tuple[np.ndarray, list[int], list[float]]:
    """
    Collect action arrays plus per-episode lengths and durations.
    """
    all_actions: list[np.ndarray] = []
    lengths: list[int] = []
    durations: list[float] = []

    for parquet_file in parquet_files:
        df = load_episode_dataframe(parquet_file)

        actions = extract_action(
            df,
            action_column=action_col,
        )

        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        if actions.ndim != 2:
            raise ValueError(
                f"Action column '{action_col}' must produce a "
                f"2D array, got shape {actions.shape}"
            )

        all_actions.append(
            actions.astype(np.float64, copy=False)
        )

        lengths.append(len(df))

        if "timestamp" in df.columns and len(df) >= 2:
            timestamps = extract_timestamps(df)
            durations.append(
                float(timestamps[-1] - timestamps[0])
            )
        else:
            durations.append(float("nan"))

    if not all_actions:
        return (
            np.empty((0, 0), dtype=np.float64),
            lengths,
            durations,
        )

    return (
        np.concatenate(all_actions, axis=0),
        lengths,
        durations,
    )

def _compute_mad(values: list[float]) -> float | None:
    """
    Compute median absolute deviation (MAD).

    Returns None when there are no finite values.
    """
    finite = np.asarray(
        [
            value
            for value in values
            if np.isfinite(value)
        ],
        dtype=np.float64,
    )

    if len(finite) == 0:
        return None

    median = float(np.median(finite))

    return float(
        np.median(
            np.abs(finite - median)
        )
    )


def _compute_path_length(
    states: np.ndarray,
) -> float:
    """
    Compute total trajectory path length from observation.state.

    For a state sequence:
        s0 -> s1 -> ... -> sn

    path length is:
        sum(||s[i] - s[i-1]||)
    """
    states = np.asarray(
        states,
        dtype=np.float64,
    )

    if states.ndim == 1:
        states = states.reshape(-1, 1)

    if states.ndim != 2:
        raise ValueError(
            f"States must be 1D or 2D, got shape {states.shape}"
        )

    if len(states) < 2:
        return 0.0

    return float(
        np.sum(
            np.linalg.norm(
                np.diff(states, axis=0),
                axis=1,
            )
        )
    )


def _extract_task_counts(
    df: pd.DataFrame,
    task_column: str = "task_index",
) -> dict[int, int]:
    """
    Count task occurrences in one episode.

    The count is by rows/frames here. The dataset-level task-balance
    check can later convert this into episode-level counts if needed.
    """
    if task_column not in df.columns:
        return {}

    counts: dict[int, int] = {}

    for value in df[task_column].tolist():
        try:
            task_id = int(value)
        except (TypeError, ValueError):
            continue

        counts[task_id] = (
            counts.get(task_id, 0) + 1
        )


    return counts


def load_reference_action_percentiles(
    data_root: Path,
    percentiles: tuple[float, float],
) -> dict[str, np.ndarray] | None:
    """Load frozen 1st/99th action bounds preserved by episode deletion.

    Other percentile pairs remain data-relative because the preserved
    metadata currently represents only the validator's default 1/99 bounds.
    """
    if tuple(map(float, percentiles)) != (1.0, 99.0):
        return None

    path = Path(data_root) / "meta" / "validation_reference_action_percentiles.json"
    if not path.is_file():
        return None

    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
        if reference.get("version") != 1:
            return None
        low = np.asarray(reference["low"], dtype=np.float64).reshape(-1)
        high = np.asarray(reference["high"], dtype=np.float64).reshape(-1)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

    if low.size == 0 or low.shape != high.shape:
        return None
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        return None

    return {"low": low, "high": high}


def compute_dataset_stats(
    parquet_files: list[Path],
    action_col: str,
    percentiles: tuple[float, float] = (1, 99),
    state_col: str = "observation.state",
    timestamp_col: str = "timestamp",
    task_col: str = "task_index",
    episode_indices: Iterable[int] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> DatasetStats:
    """
    Compute reusable dataset-level statistics.

    Statistics currently collected:

        action_percentiles
            Per-action-dimension low/high percentile bounds.

        median_episode_length
            Median number of parquet rows per episode.

        median_duration
            Median episode duration.

        duration_mad
            Median absolute deviation of episode durations.

        median_path_length
            Median trajectory path length.

        task_counts
            Aggregate task/frame counts.

    Dataset paths are supplied dynamically.
    """
    if len(percentiles) != 2:
        raise ValueError(
            "percentiles must contain exactly two values"
        )

    low, high = map(
        float,
        percentiles,
    )

    if not 0 <= low <= 100:
        raise ValueError(
            f"Invalid lower percentile: {low}"
        )

    if not 0 <= high <= 100:
        raise ValueError(
            f"Invalid upper percentile: {high}"
        )

    if low >= high:
        raise ValueError(
            "Lower percentile must be smaller than "
            "upper percentile"
        )

    all_actions: list[np.ndarray] = []
    episode_lengths: list[int] = []
    durations: list[float] = []
    path_lengths: list[float] = []
    task_counts: dict[int, int] = {}
    selected_episode_indices = (
        {int(index) for index in episode_indices}
        if episode_indices is not None
        else None
    )

    for parquet_file in dict.fromkeys(Path(path) for path in parquet_files):
        if cancel_check is not None:
            cancel_check()
        chunk_df = load_episode_dataframe(parquet_file)
        episode_frames = (
            group
            for _, group in chunk_df.groupby("episode_index", sort=False)
        ) if "episode_index" in chunk_df.columns else (chunk_df,)

        for df in episode_frames:
            if cancel_check is not None:
                cancel_check()
            if (
                selected_episode_indices is not None
                and "episode_index" in df.columns
                and int(df["episode_index"].iloc[0]) not in selected_episode_indices
            ):
                continue
            if "frame_index" in df.columns:
                df = df.sort_values("frame_index", kind="stable")

            # ----------------------------------------------------
            # Episode length
            # ----------------------------------------------------

            episode_lengths.append(
                len(df)
            )

            # ----------------------------------------------------
            # Actions
            # ----------------------------------------------------

            actions = extract_action(
                df,
                action_column=action_col,
            )

            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)

            if actions.ndim != 2:
                raise ValueError(
                    f"Action column '{action_col}' must produce "
                    f"a 2D array, got shape {actions.shape}"
                )

            if len(actions):
                all_actions.append(
                    actions.astype(
                        np.float64,
                        copy=False,
                    )
                )

            # ----------------------------------------------------
            # Duration
            # ----------------------------------------------------

            if timestamp_col in df.columns and len(df) >= 2:
                timestamps = extract_timestamps(
                    df,
                    timestamp_column=timestamp_col,
                )

                duration = float(
                    timestamps[-1] - timestamps[0]
                )

                if np.isfinite(duration):
                    durations.append(
                        duration
                    )

            # ----------------------------------------------------
            # Path length
            # ----------------------------------------------------

            if state_col in df.columns:
                states = extract_state(
                    df,
                    state_column=state_col,
                )

                try:
                    path_length = _compute_path_length(
                        states
                    )

                    if np.isfinite(path_length):
                        path_lengths.append(
                            path_length
                        )
                except ValueError:
                    # Structural validation will report malformed
                    # state data. Do not make statistics calculation
                    # fail solely because one episode is malformed.
                    pass

            # ----------------------------------------------------
            # Task counts
            # ----------------------------------------------------

            episode_task_counts = _extract_task_counts(
                df,
                task_column=task_col,
            )

            for task_id, count in episode_task_counts.items():
                task_counts[task_id] = (
                    task_counts.get(task_id, 0)
                    + count
                )

    # --------------------------------------------------------
    # Action percentile statistics
    # --------------------------------------------------------

    if all_actions:
        action_matrix = np.concatenate(
            all_actions,
            axis=0,
        )

        bounds = np.percentile(
            action_matrix,
            [low, high],
            axis=0,
        )

        action_percentiles = {
            "low": bounds[0],
            "high": bounds[1],
        }
    else:
        action_percentiles = {}

    # --------------------------------------------------------
    # Duration statistics
    # --------------------------------------------------------

    median_duration = (
        float(np.median(durations))
        if durations
        else None
    )

    duration_mad = _compute_mad(
        durations
    )

    # --------------------------------------------------------
    # Path statistics
    # --------------------------------------------------------

    median_path_length = (
        float(np.median(path_lengths))
        if path_lengths
        else None
    )

    return DatasetStats(
        action_percentiles=action_percentiles,
        median_episode_length=(
            float(np.median(episode_lengths))
            if episode_lengths
            else None
        ),
        median_duration=median_duration,
        median_path_length=median_path_length,
        task_counts=task_counts,
        extra={
            "duration_mad": duration_mad,
            "duration_values": durations,
            "path_length_values": path_lengths,
        },
    )