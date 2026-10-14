

"""
convert_to_lerobot_v3_fixed.py

python3 v3_convertor.py \
  --root /home/thunder/MD/Dinesh/custom/dataset/pick_box_2_train_source \
  --output /home/thunder/MD/Dinesh/dataset/pick_v3_train





Converts a v2.1-style LeRobot dataset into a v3.0 dataset.

IMPORTANT:
    This version intentionally RE-ENCODES videos.
    It does NOT use ffmpeg stream-copy concatenation.

    

Main fixes:
    1. Exact per-frame timestamps: frame_index / FPS.
    2. Exact video CFR timestamps.
    3. No stream-copy PTS drift.
    4. Episode video boundaries use frame_count / FPS.
    5. Surviving episodes are automatically renumbered.
    6. Generated video PTS are validated before completion.

Expected source structure:

    SOURCE/
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       ├── episode_000001.parquet
    │       └── ...
    ├── videos/
    │   ├── observation.images.world/
    │   │   └── chunk-000/
    │   │       ├── episode_000000.mp4
    │   │       └── ...
    │   ├── observation.images.wrist_left/
    │   └── observation.images.wrist_right/
    └── meta/
        ├── info.json
        ├── episodes.jsonl
        ├── tasks.jsonl
        └── ...

Usage:

    python convert_to_lerobot_v3_fixed.py \
        --root /path/to/source_dataset \
        --output /path/to/output_dataset_v3

Example:

    python convert_to_lerobot_v3_fixed.py \
        --root /home/thunder/MD/Dinesh/dataset/pick \
        --output /home/thunder/MD/Dinesh/dataset/pick_v3_final

Optional:

    --data-file-size-in-mb 100
    --video-file-size-in-mb 200
    --chunk-size 1000
    --episodes-per-meta-file 1000000
    --fps 30
    --crf 18
    --preset medium

Requirements:

    pandas
    pyarrow
    numpy
    ffmpeg
    ffprobe
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"

DEFAULT_DATA_PATH = (
    "data/" + CHUNK_FILE_PATTERN + ".parquet"
)

DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
)

DEFAULT_EPISODES_PATH = (
    "meta/episodes/" + CHUNK_FILE_PATTERN + ".parquet"
)

DEFAULT_TASKS_PATH = "meta/tasks.parquet"

EP_IDX_RE = re.compile(r"(\d+)$")

ADMIN_COLUMNS = {
    "episode_index",
    "frame_index",
    "index",
    "timestamp",
    "task_index",
}


# --------------------------------------------------------------------------- #
# General helpers
# --------------------------------------------------------------------------- #

def run_command(command, description=None):
    if description:
        print(f"[ffmpeg] {description}")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("\nCOMMAND FAILED:")
        print(" ".join(str(x) for x in command))
        print("\nFFMPEG/FFPROBE OUTPUT:")
        print(result.stderr)
        raise RuntimeError(
            f"Command failed with return code {result.returncode}"
        )

    return result


def update_chunk_file_indices(
    chunk_idx: int,
    file_idx: int,
    chunks_size: int,
):
    if file_idx == chunks_size - 1:
        return chunk_idx + 1, 0

    return chunk_idx, file_idx + 1


def episode_index_from_path(path: Path) -> int:
    match = EP_IDX_RE.search(path.stem)

    if not match:
        raise ValueError(
            f"Could not parse episode index from filename: {path}"
        )

    return int(match.group(1))


def glob_sorted_by_episode(
    root: Path,
    pattern: str,
):
    paths = list(root.glob(pattern))

    return sorted(
        paths,
        key=episode_index_from_path,
    )


def get_file_size_in_mb(path: Path):
    return path.stat().st_size / (1024 * 1024)


def to_jsonable(value):
    """Recursively convert parquet/pandas/numpy values into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    return value


def _write_json_atomically(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(to_jsonable(value), indent=2) + "\n")
    temporary.replace(path)


def _write_jsonl_atomically(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _task_name(value):
    """Extract one task label from scalar or common nested task records."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("task", "task_name", "name", "description", "prompt"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key].strip()
        if len(value) == 1:
            key, nested = next(iter(value.items()))
            if isinstance(nested, str) and nested.strip():
                return key if key not in {"0", "1"} else nested.strip()
            return _task_name(nested)
    return None


def _task_values(value):
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_task_values(item))
        return result
    name = _task_name(value)
    return [name] if name else []


def normalize_task_records(records):
    """Normalize direct task rows and rows containing lists of task dictionaries."""
    normalized = []
    next_index = 0
    for original in records:
        row = to_jsonable(original)
        if not isinstance(row, dict):
            row = {"task": row}
        nested = row.get("tasks")
        if nested is None:
            nested = row.get("task")
        children = nested if isinstance(nested, list) else [nested]
        children = children if children != [None] else [row]
        for child in children:
            record = {key: value for key, value in row.items() if key not in {"task", "tasks"}}
            if isinstance(child, dict):
                record.update(child)
            elif child is not None:
                record["task"] = child
            label = _task_name(record.get("task"))
            if not label:
                label = _task_name(record)
            if not label:
                continue
            record["task"] = label
            try:
                task_index = int(record.get("task_index"))
            except (TypeError, ValueError):
                task_index = next_index
            # A parent row's index cannot be shared by multiple child task records.
            if isinstance(nested, list) and len(children) > 1 and "task_index" not in (child if isinstance(child, dict) else {}):
                task_index = next_index
            record["task_index"] = task_index
            next_index = max(next_index, task_index + 1)
            normalized.append(record)
    # Deduplicate identical task/index pairs while retaining prompts/other metadata.
    unique = {}
    for record in normalized:
        key = (record["task_index"], record["task"])
        unique.setdefault(key, record)
    return sorted(unique.values(), key=lambda item: (item["task_index"], item["task"]))


def _records_from_json(path: Path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if any(key in value for key in ("task", "tasks", "task_index")):
            return [value]
        return [{"task": key, "prompt": item} for key, item in value.items()]
    return []


def _read_task_parquet(root: Path):
    source = root / "meta" / "tasks.parquet"
    if not source.is_file():
        return []
    return pd.read_parquet(source).to_dict(orient="records")


def ensure_task_sidecars(root: Path):
    """Verify tasks.json and produce normalized JSON/JSONL task sidecars from parquet."""
    meta = root / "meta"
    json_path = meta / "tasks.json"
    jsonl_path = meta / "tasks.jsonl"
    records = normalize_task_records(_records_from_json(json_path)) if json_path.is_file() else []
    if not records:
        records = normalize_task_records(_read_task_parquet(root))
    if not records and jsonl_path.is_file():
        try:
            jsonl_records = [
                json.loads(line)
                for line in jsonl_path.read_text().splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError:
            jsonl_records = []
        records = normalize_task_records(jsonl_records)
    if not records:
        raise RuntimeError(
            f"No valid task records found in {json_path} or meta/tasks.parquet."
        )
    _write_json_atomically(json_path, records)
    _write_jsonl_atomically(jsonl_path, records)
    print(f"[tasks] verified {len(records)} task records; wrote tasks.json and tasks.jsonl")
    return records


def _episode_tasks(record):
    value = record.get("tasks")
    if value is None:
        value = record.get("task")
    return _task_values(value)


def _normalize_episode_records(records):
    normalized = []
    for original in records:
        row = to_jsonable(original)
        if not isinstance(row, dict) or row.get("episode_index") is None:
            continue
        try:
            row["episode_index"] = int(row["episode_index"])
        except (TypeError, ValueError):
            continue
        tasks = _episode_tasks(row)
        row["tasks"] = tasks
        task_index = row.get("task_index")
        if isinstance(task_index, list):
            task_index = next((item for item in task_index if item is not None), None)
        if isinstance(task_index, dict):
            task_index = task_index.get("task_index", task_index.get("index"))
        try:
            row["task_index"] = int(task_index) if task_index is not None else None
        except (TypeError, ValueError):
            row["task_index"] = None
        normalized.append(row)
    by_index = {row["episode_index"]: row for row in normalized}
    return [by_index[index] for index in sorted(by_index)]


def _read_episode_parquet(root: Path):
    parquet_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not parquet_files:
        parquet_files = sorted((root / "meta" / "episodes").glob("*.parquet"))
    records = []
    for path in parquet_files:
        records.extend(pd.read_parquet(path).to_dict(orient="records"))
    return _normalize_episode_records(records)


def ensure_episode_sidecar(root: Path):
    """Verify episodes.jsonl and rebuild it from meta/episodes parquet when needed."""
    meta = root / "meta"
    sidecar = meta / "episodes.jsonl"
    records = []
    if sidecar.is_file():
        try:
            records = _normalize_episode_records(
                [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
            )
        except json.JSONDecodeError:
            records = []

    data_indices = {
        episode_index_from_path(path)
        for path in glob_sorted_by_episode(root, "data/*/episode_*.parquet")
    }
    records_by_index = {row["episode_index"]: row for row in records}
    expected_indices = data_indices or set(records_by_index)
    if not records_by_index or not expected_indices.issubset(records_by_index):
        parquet_records = _read_episode_parquet(root)
        parquet_by_index = {row["episode_index"]: row for row in parquet_records}
        missing = expected_indices - set(records_by_index)
        if missing and not missing.issubset(parquet_by_index):
            missing_parquet = sorted(missing - set(parquet_by_index))[:10]
            raise RuntimeError(
                f"episodes.jsonl is missing metadata for episode(s) {sorted(missing)[:10]} "
                f"and meta/episodes parquet does not contain {missing_parquet}."
            )
        records_by_index.update({index: parquet_by_index[index] for index in missing})
        if not records_by_index:
            records_by_index = parquet_by_index
    records = [records_by_index[index] for index in sorted(records_by_index)]
    if not records:
        raise RuntimeError(
            f"No valid episode metadata found in {sidecar} or meta/episodes parquet."
        )
    _write_jsonl_atomically(sidecar, records)
    print(f"[episodes-meta] verified {len(records)} episode records; wrote episodes.jsonl")
    return records


def ensure_source_metadata(root: Path):
    """Materialize the sidecars expected by this converter and downstream consumers."""
    episodes = ensure_episode_sidecar(root)
    tasks = ensure_task_sidecars(root)
    return episodes, tasks


# --------------------------------------------------------------------------- #
# FFprobe helpers
# --------------------------------------------------------------------------- #

def get_video_info(path: Path):
    """
    Return:

        frame_count
        fps
        width
        height
        duration

    The frame count is obtained using ffprobe -count_frames.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,r_frame_rate,avg_frame_rate,width,height",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]

    result = run_command(command)

    data = json.loads(result.stdout)

    if not data.get("streams"):
        raise RuntimeError(
            f"No video stream found in: {path}"
        )

    stream = data["streams"][0]

    frame_count = stream.get("nb_read_frames")

    if frame_count in (None, "N/A"):
        raise RuntimeError(
            f"Could not determine frame count for: {path}"
        )

    frame_count = int(frame_count)

    def parse_fraction(value):
        if not value or value == "N/A":
            return None

        if "/" in value:
            a, b = value.split("/", 1)

            a = float(a)
            b = float(b)

            if b == 0:
                return None

            return a / b

        return float(value)

    fps = parse_fraction(
        stream.get("avg_frame_rate")
    )

    if fps is None:
        fps = parse_fraction(
            stream.get("r_frame_rate")
        )

    duration = None

    if data.get("format"):
        raw_duration = data["format"].get("duration")

        if raw_duration not in (None, "N/A"):
            duration = float(raw_duration)

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "duration": duration,
    }


def get_video_pts(path: Path):
    """
    Return all decoded video frame PTS values.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(path),
    ]

    result = run_command(command)

    pts = []

    for line in result.stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            pts.append(float(line))
        except ValueError:
            continue

    return pts


# --------------------------------------------------------------------------- #
# Episode discovery
# --------------------------------------------------------------------------- #

def build_renumber_map(root: Path):
    """
    Find surviving source episodes from data/*.parquet.

    Example:

        original:
            0
            1
            3
            7
            10

        output:
            0 -> 0
            1 -> 1
            3 -> 2
            7 -> 3
            10 -> 4
    """

    ep_paths = glob_sorted_by_episode(
        root,
        "data/*/*.parquet",
    )

    if not ep_paths:
        raise RuntimeError(
            f"No episode parquet files found under: {root / 'data'}"
        )

    original_indices = [
        episode_index_from_path(path)
        for path in ep_paths
    ]

    if len(set(original_indices)) != len(original_indices):
        raise RuntimeError(
            "Duplicate episode indices found."
        )

    return {
        original: new
        for new, original in enumerate(original_indices)
    }


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def numeric_feature_keys(info):
    keys = []

    for name, feature in info.get(
        "features",
        {},
    ).items():

        if name in ADMIN_COLUMNS:
            continue

        dtype = feature.get("dtype")

        if dtype in (
            "video",
            "image",
            "string",
        ):
            continue

        keys.append(name)

    return keys


def compute_episode_stats(
    df: pd.DataFrame,
    feature_keys,
):
    stats = {}

    for key in feature_keys:

        if key not in df.columns:
            continue

        sample = df[key].iloc[0]

        if isinstance(
            sample,
            (list, tuple, np.ndarray),
        ):
            arr = np.stack(
                df[key].to_numpy()
            ).astype(np.float64)
        else:
            arr = (
                df[key]
                .to_numpy()
                .astype(np.float64)
                [:, None]
            )

        stats[key] = {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "min": arr.min(axis=0),
            "max": arr.max(axis=0),
            "count": np.array(len(arr)),
            "q01": np.quantile(
                arr,
                0.01,
                axis=0,
            ),
            "q99": np.quantile(
                arr,
                0.99,
                axis=0,
            ),
        }

    return stats


def aggregate_feature_stats(stats_list):
    means = np.stack(
        [x["mean"] for x in stats_list]
    )

    variances = np.stack(
        [x["std"] ** 2 for x in stats_list]
    )

    counts = np.stack(
        [x["count"] for x in stats_list]
    ).astype(np.float64)

    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(
            counts,
            axis=-1,
        )

    total_mean = (
        (means * counts).sum(axis=0)
        / total_count
    )

    delta_means = (
        means - total_mean
    )

    total_variance = (
        (
            variances
            + delta_means ** 2
        )
        * counts
    ).sum(axis=0) / total_count

    result = {
        "min": np.min(
            np.stack(
                [x["min"] for x in stats_list]
            ),
            axis=0,
        ),
        "max": np.max(
            np.stack(
                [x["max"] for x in stats_list]
            ),
            axis=0,
        ),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }

    if all(
        "q01" in x
        for x in stats_list
    ):
        result["q01"] = np.min(
            np.stack(
                [x["q01"] for x in stats_list]
            ),
            axis=0,
        )

    if all(
        "q99" in x
        for x in stats_list
    ):
        result["q99"] = np.max(
            np.stack(
                [x["q99"] for x in stats_list]
            ),
            axis=0,
        )

    return result


def aggregate_stats(
    episode_stats_list,
):
    keys = {
        key
        for stats in episode_stats_list
        for key in stats
    }

    return {
        key: aggregate_feature_stats(
            [
                stats[key]
                for stats in episode_stats_list
                if key in stats
            ]
        )
        for key in keys
    }


def flatten_stats(stats):
    flat = {}

    for feature, values in stats.items():
        for stat_name, value in values.items():
            flat[
                f"stats/{feature}/{stat_name}"
            ] = to_jsonable(value)

    return flat


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

def convert_tasks(
    root: Path,
    new_root: Path,
):
    tasks = ensure_task_sidecars(root)

    tasks = sorted(
        tasks,
        key=lambda x: x["task_index"],
    )

    df = pd.DataFrame(tasks)

    if "task" in df.columns:
        df = df.set_index("task")
        df.index.name = "task"

    output = (
        new_root
        / DEFAULT_TASKS_PATH
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(output)

    print(
        f"[tasks] wrote {len(df)} tasks"
    )


# --------------------------------------------------------------------------- #
# Data conversion
# --------------------------------------------------------------------------- #

def convert_data(
    root: Path,
    new_root: Path,
    data_file_size_in_mb: float,
    chunk_size: int,
    feature_keys,
    keep_columns,
    renumber_map,
    fps: int,
):
    ep_paths = glob_sorted_by_episode(
        root,
        "data/*/*.parquet",
    )

    chunk_idx = 0
    file_idx = 0

    size_in_mb = 0.0
    total_frames = 0

    dfs_to_write = []

    episodes_meta = []
    episodes_stats = []

    dropped_columns = set()

    for ep_path in ep_paths:

        original_episode = (
            episode_index_from_path(ep_path)
        )

        new_episode = (
            renumber_map[original_episode]
        )

        df = pd.read_parquet(ep_path)

        if len(df) == 0:
            raise RuntimeError(
                f"Episode contains zero frames: {ep_path}"
            )

        frame_count = len(df)

        print(
            f"[data] episode "
            f"{original_episode} -> "
            f"{new_episode} "
            f"frames={frame_count}"
        )

        # --------------------------------------------------------------- #
        # Normalize administrative columns.
        # --------------------------------------------------------------- #

        df = df.copy()

        df["episode_index"] = (
            new_episode
        )

        df["frame_index"] = np.arange(
            frame_count,
            dtype=np.int64,
        )

        df["timestamp"] = (
            np.arange(
                frame_count,
                dtype=np.float64,
            )
            / float(fps)
        )

        df["index"] = np.arange(
            total_frames,
            total_frames + frame_count,
            dtype=np.int64,
        )

        # --------------------------------------------------------------- #
        # Preserve only columns declared in info.json.
        # --------------------------------------------------------------- #

        extra_columns = [
            column
            for column in df.columns
            if column not in keep_columns
        ]

        dropped_columns.update(
            extra_columns
        )

        extra_values = {}

        for column in extra_columns:
            if len(df):
                extra_values[column] = (
                    df[column].iloc[0]
                )

        # --------------------------------------------------------------- #
        # File rollover.
        # --------------------------------------------------------------- #

        estimated_size = (
            get_file_size_in_mb(ep_path)
        )

        if (
            size_in_mb + estimated_size
            >= data_file_size_in_mb
            and dfs_to_write
        ):
            write_data_file(
                dfs_to_write,
                new_root,
                chunk_idx,
                file_idx,
                keep_columns,
            )

            chunk_idx, file_idx = (
                update_chunk_file_indices(
                    chunk_idx,
                    file_idx,
                    chunk_size,
                )
            )

            dfs_to_write = []
            size_in_mb = 0.0

        # --------------------------------------------------------------- #
        # Episode metadata.
        # --------------------------------------------------------------- #

        episode_meta = {
            "episode_index": new_episode,
            "_orig_episode_index": original_episode,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": total_frames,
            "dataset_to_index": (
                total_frames + frame_count
            ),
            "length": frame_count,
        }

        for column, value in extra_values.items():
            episode_meta[
                f"extra/{column}"
            ] = to_jsonable(value)

        episodes_meta.append(
            episode_meta
        )

        episodes_stats.append(
            compute_episode_stats(
                df,
                feature_keys,
            )
        )

        # --------------------------------------------------------------- #
        # Prepare output dataframe.
        # --------------------------------------------------------------- #

        available_columns = [
            column
            for column in df.columns
            if column in keep_columns
        ]

        df_out = df[
            available_columns
        ].copy()

        dfs_to_write.append(
            df_out
        )

        size_in_mb += estimated_size
        total_frames += frame_count

    if dfs_to_write:
        write_data_file(
            dfs_to_write,
            new_root,
            chunk_idx,
            file_idx,
            keep_columns,
        )

    if dropped_columns:
        print(
            "[data] undeclared columns "
            "removed from frame parquet:"
        )

        for column in sorted(
            dropped_columns
        ):
            print(
                f"    {column}"
            )

    print(
        f"[data] converted "
        f"{len(ep_paths)} episodes"
    )

    print(
        f"[data] total frames: "
        f"{total_frames}"
    )

    return (
        episodes_meta,
        episodes_stats,
        total_frames,
    )


def write_data_file(
    dataframes,
    new_root,
    chunk_idx,
    file_idx,
    keep_columns,
):
    output = (
        new_root
        / DEFAULT_DATA_PATH.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    ordered_columns = [
        column
        for column in keep_columns
        if column in df.columns
    ]

    df = df[
        ordered_columns
    ]

    df.to_parquet(
        output,
        index=False,
    )

    print(
        f"[data] wrote {output} "
        f"rows={len(df)}"
    )


# --------------------------------------------------------------------------- #
# Video normalization
# --------------------------------------------------------------------------- #

def validate_source_videos(
    root,
    video_keys,
    renumber_map,
    fps,
):
    """
    Validate that every surviving episode has every camera video.
    """

    expected_original = set(
        renumber_map.keys()
    )

    camera_info = {}

    for video_key in sorted(
        video_keys
    ):
        paths = glob_sorted_by_episode(
            root,
            f"videos/{video_key}/*/*.mp4",
        )

        found = {
            episode_index_from_path(path)
            for path in paths
        }

        missing = (
            expected_original - found
        )

        extra = (
            found - expected_original
        )

        if missing:
            raise RuntimeError(
                f"Camera '{video_key}' is missing "
                f"episodes: {sorted(missing)}"
            )

        if extra:
            print(
                f"[video:{video_key}] ignoring "
                f"non-surviving episodes: "
                f"{sorted(extra)}"
            )

        info_by_episode = {}

        for path in paths:
            original = (
                episode_index_from_path(path)
            )

            if original not in expected_original:
                continue

            info = get_video_info(path)

            if abs(
                info["fps"] - fps
            ) > 0.01:
                raise RuntimeError(
                    f"Unexpected FPS for {path}: "
                    f"{info['fps']} "
                    f"(expected {fps})"
                )

            info_by_episode[original] = {
                "path": path,
                "info": info,
            }

        camera_info[video_key] = (
            info_by_episode
        )

        print(
            f"[validate:{video_key}] "
            f"{len(info_by_episode)} videos OK"
        )

    return camera_info


def create_concat_file(
    paths,
):
    """
    Create an ffmpeg concat demuxer file.
    """

    file_handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
    )

    for path in paths:
        escaped = (
            str(path.resolve())
            .replace("'", "'\\''")
        )

        file_handle.write(
            f"file '{escaped}'\n"
        )

    file_handle.close()

    return Path(
        file_handle.name
    )

def reencode_video_group(
    input_paths,
    output_path,
    fps,
    crf,
    preset,
):
    """
    Concatenate source clips and re-encode them while preserving
    exactly the original number of frames.

    IMPORTANT:
        Source videos are already CFR 30 FPS.

        We therefore DO NOT use:
            -vf fps=30

        because the fps filter can duplicate/drop frames when the
        concatenated input contains tiny timestamp discontinuities.

        Instead, we use:
            setpts=N/(FPS*TB)

        to rebuild timestamps directly from decoded frame number.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    concat_file = create_concat_file(
        input_paths
    )

    temporary_output = (
        output_path.with_suffix(
            ".tmp.mp4"
        )
    )

    try:
        command = [
            "ffmpeg",
            "-y",

            "-f",
            "concat",

            "-safe",
            "0",

            "-i",
            str(concat_file),

            "-map",
            "0:v:0",

            # DO NOT use fps=30 here.
            # The source videos are already 30 FPS.
            "-vf",
            f"setpts=N/({fps}*TB)",

            # Force the encoder/output stream to 30 FPS
            # without using the fps filter.
            "-r",
            str(fps),

            "-c:v",
            "libx264",

            "-preset",
            preset,

            "-crf",
            str(crf),

            "-pix_fmt",
            "yuv420p",

            # Avoid B-frame timestamp complications.
            "-bf",
            "0",

            "-an",

            "-movflags",
            "+faststart",

            # 90 kHz gives an exact integer representation
            # for 1/30 second:
            #
            # 90000 / 30 = 3000 ticks
            "-video_track_timescale",
            "90000",

            str(temporary_output),
        ]

        run_command(
            command,
            description=(
                f"re-encoding {len(input_paths)} "
                f"episode videos -> "
                f"{output_path}"
            ),
        )

        if not temporary_output.exists():
            raise RuntimeError(
                f"FFmpeg completed but output "
                f"was not created: "
                f"{temporary_output}"
            )

        temporary_output.replace(
            output_path
        )

    finally:
        concat_file.unlink(
            missing_ok=True
        )

        temporary_output.unlink(
            missing_ok=True
        )


# --------------------------------------------------------------------------- #
# Video conversion
# --------------------------------------------------------------------------- #

def convert_videos(
    root,
    new_root,
    video_keys,
    video_file_size_in_mb,
    chunk_size,
    renumber_map,
    fps,
    crf,
    preset,
):
    if not video_keys:
        return []

    camera_infos = validate_source_videos(
        root,
        video_keys,
        renumber_map,
        fps,
    )

    # --------------------------------------------------------------- #
    # All cameras must have the same frame count for each episode.
    # --------------------------------------------------------------- #

    original_episode_order = sorted(
        renumber_map.keys()
    )

    for original_episode in original_episode_order:

        frame_counts = []

        for video_key in sorted(
            video_keys
        ):
            info = camera_infos[
                video_key
            ][original_episode]["info"]

            frame_counts.append(
                info["frame_count"]
            )

        if len(set(frame_counts)) != 1:
            raise RuntimeError(
                f"Camera frame-count mismatch "
                f"for source episode "
                f"{original_episode}: "
                f"{frame_counts}"
            )

        if frame_counts[0] == 0:
            raise RuntimeError(
                f"Episode "
                f"{original_episode} "
                f"contains zero video frames."
            )

    print(
        "[video] all cameras have "
        "matching episode frame counts"
    )

    # --------------------------------------------------------------- #
    # Convert camera-by-camera.
    # --------------------------------------------------------------- #

    per_camera_metadata = []

    for video_key in sorted(
        video_keys
    ):
        metadata = convert_one_camera(
            video_key=video_key,
            new_root=new_root,
            camera_info=camera_infos[
                video_key
            ],
            renumber_map=renumber_map,
            video_file_size_in_mb=video_file_size_in_mb,
            chunk_size=chunk_size,
            fps=fps,
            crf=crf,
            preset=preset,
        )

        per_camera_metadata.append(
            metadata
        )

    # --------------------------------------------------------------- #
    # Merge metadata for all cameras.
    # --------------------------------------------------------------- #

    number_of_episodes = len(
        per_camera_metadata[0]
    )

    for metadata in per_camera_metadata:
        if len(metadata) != number_of_episodes:
            raise RuntimeError(
                "Camera episode counts do not match."
            )

    combined = []

    for i in range(
        number_of_episodes
    ):
        merged = {}

        episode_index = (
            per_camera_metadata[0][i]
            ["episode_index"]
        )

        merged["episode_index"] = (
            episode_index
        )

        for metadata in per_camera_metadata:
            row = metadata[i]

            if row["episode_index"] != episode_index:
                raise RuntimeError(
                    "Camera episode ordering mismatch."
                )

            for key, value in row.items():

                if key == "episode_index":
                    continue

                merged[key] = value

        combined.append(
            merged
        )

    return combined


def convert_one_camera(
    video_key,
    new_root,
    camera_info,
    renumber_map,
    video_file_size_in_mb,
    chunk_size,
    fps,
    crf,
    preset,
):
    original_episode_order = sorted(
        renumber_map.keys()
    )

    # --------------------------------------------------------------- #
    # Build output groups.
    #
    # We estimate output size using source sizes.
    # The exact output size is not important for correctness.
    # --------------------------------------------------------------- #

    groups = []

    current_paths = []
    current_source_size = 0.0

    for original_episode in (
        original_episode_order
    ):

        path = camera_info[
            original_episode
        ]["path"]

        source_size = (
            get_file_size_in_mb(path)
        )

        if (
            current_paths
            and
            current_source_size + source_size
            >= video_file_size_in_mb
        ):
            groups.append(
                current_paths
            )

            current_paths = []
            current_source_size = 0.0

        current_paths.append(
            path
        )

        current_source_size += source_size

    if current_paths:
        groups.append(
            current_paths
        )

    # --------------------------------------------------------------- #
    # Convert groups.
    # --------------------------------------------------------------- #

    metadata_by_original = {}

    chunk_idx = 0
    file_idx = 0

    episode_cursor = 0

    for group_number, paths in enumerate(
        groups
    ):

        output_path = (
            new_root
            / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
        )

        print()
        print(
            f"[video:{video_key}] "
            f"group {group_number + 1}/"
            f"{len(groups)}"
        )

        print(
            f"[video:{video_key}] "
            f"episodes: "
            f"{len(paths)}"
        )

        # ----------------------------------------------------------- #
        # Re-encode.
        # ----------------------------------------------------------- #

        reencode_video_group(
            paths,
            output_path,
            fps,
            crf,
            preset,
        )

        # ----------------------------------------------------------- #
        # Validate generated video.
        # ----------------------------------------------------------- #

        output_info = get_video_info(
            output_path
        )

        expected_frames = sum(
            camera_info[
                episode_index_from_path(path)
            ]["info"]["frame_count"]
            for path in paths
        )

        if output_info[
            "frame_count"
        ] != expected_frames:

            raise RuntimeError(
                f"Frame count mismatch in "
                f"generated video:\n"
                f"    output: {output_path}\n"
                f"    expected: {expected_frames}\n"
                f"    actual: {output_info['frame_count']}"
            )

        if abs(
            output_info["fps"] - fps
        ) > 0.01:

            raise RuntimeError(
                f"Generated video FPS mismatch:\n"
                f"    output: {output_path}\n"
                f"    expected: {fps}\n"
                f"    actual: {output_info['fps']}"
            )

        print(
            f"[video:{video_key}] "
            f"validated frames="
            f"{output_info['frame_count']} "
            f"fps={output_info['fps']:.6f}"
        )

        # ----------------------------------------------------------- #
        # Metadata timestamps.
        #
        # IMPORTANT:
        #
        # For N frames at FPS:
        #
        #     first frame = 0
        #     last frame  = (N-1)/FPS
        #
        # but the segment duration is:
        #
        #     N/FPS
        #
        # The next episode starts at N/FPS.
        # ----------------------------------------------------------- #

        offset = 0.0

        for path in paths:

            original_episode = (
                episode_index_from_path(path)
            )

            new_episode = (
                renumber_map[
                    original_episode
                ]
            )

            frame_count = camera_info[
                original_episode
            ]["info"]["frame_count"]

            segment_duration = (
                frame_count / float(fps)
            )

            metadata_by_original[
                original_episode
            ] = {
                "episode_index": new_episode,

                f"videos/{video_key}/chunk_index":
                    chunk_idx,

                f"videos/{video_key}/file_index":
                    file_idx,

                f"videos/{video_key}/from_timestamp":
                    offset,

                f"videos/{video_key}/to_timestamp":
                    offset + segment_duration,
            }

            offset += segment_duration

            episode_cursor += 1

        chunk_idx, file_idx = (
            update_chunk_file_indices(
                chunk_idx,
                file_idx,
                chunk_size,
            )
        )

    # --------------------------------------------------------------- #
    # Return in NEW episode order.
    # --------------------------------------------------------------- #

    result = []

    for original_episode in (
        original_episode_order
    ):
        result.append(
            metadata_by_original[
                original_episode
            ]
        )

    print(
        f"[video:{video_key}] "
        f"converted {len(result)} episodes"
    )

    return result


# --------------------------------------------------------------------------- #
# Episode metadata
# --------------------------------------------------------------------------- #

def convert_episodes_metadata(
    root,
    new_root,
    data_meta,
    episodes_stats,
    video_meta,
    episodes_per_meta_file,
    chunk_size,
):
    legacy_episodes = {
        row["episode_index"]: row
        for row in ensure_episode_sidecar(root)
    }

    if not (
        len(data_meta)
        == len(episodes_stats)
    ):
        raise RuntimeError(
            "Data metadata and episode stats "
            "have different lengths."
        )

    if video_meta and not (
        len(video_meta)
        == len(data_meta)
    ):
        raise RuntimeError(
            "Video metadata and data metadata "
            "have different lengths."
        )

    rows = []

    for i in range(
        len(data_meta)
    ):
        data_row = data_meta[i]

        original_episode = (
            data_row[
                "_orig_episode_index"
            ]
        )

        new_episode = (
            data_row[
                "episode_index"
            ]
        )

        legacy = legacy_episodes.get(
            original_episode
        )

        if legacy is None:
            raise RuntimeError(
                f"No episodes.jsonl entry "
                f"for source episode "
                f"{original_episode}"
            )

        row = {
            key: value
            for key, value in data_row.items()
            if key != "_orig_episode_index"
        }

        if video_meta:
            video_row = video_meta[i]

            if (
                video_row["episode_index"]
                != new_episode
            ):
                raise RuntimeError(
                    "Episode ordering mismatch "
                    "between data and video metadata."
                )

            row.update(
                {
                    key: value
                    for key, value
                    in video_row.items()
                    if key != "episode_index"
                }
            )

        episode_tasks = legacy.get("tasks")
        if episode_tasks is None:
            episode_task = legacy.get("task")
            episode_tasks = [] if episode_task is None else [episode_task]
        elif isinstance(episode_tasks, str):
            episode_tasks = [episode_tasks]
        row["tasks"] = [str(task) for task in episode_tasks if task is not None]

        if "prompt" in legacy:
            row["prompt"] = legacy[
                "prompt"
            ]

        row["task_index"] = legacy.get(
            "task_index"
        )

        row.update(
            flatten_stats(
                episodes_stats[i]
            )
        )

        rows.append(row)

    # Keep the JSONL sidecar in sync with renumbered v3 episode metadata.
    _write_jsonl_atomically(
        new_root / "meta" / "episodes.jsonl",
        rows,
    )
    ep_df = pd.DataFrame(rows)

    # --------------------------------------------------------------- #
    # meta/episodes chunk/file.
    # --------------------------------------------------------------- #

    meta_chunk = []
    meta_file = []

    current_chunk = 0
    current_file = 0
    count = 0

    for _ in range(
        len(ep_df)
    ):

        if (
            count
            == episodes_per_meta_file
        ):
            current_chunk, current_file = (
                update_chunk_file_indices(
                    current_chunk,
                    current_file,
                    chunk_size,
                )
            )

            count = 0

        meta_chunk.append(
            current_chunk
        )

        meta_file.append(
            current_file
        )

        count += 1

    ep_df[
        "meta/episodes/chunk_index"
    ] = meta_chunk

    ep_df[
        "meta/episodes/file_index"
    ] = meta_file

    for (
        mc,
        mf
    ), group in ep_df.groupby(
        [
            "meta/episodes/chunk_index",
            "meta/episodes/file_index",
        ]
    ):
        output = (
            new_root
            / DEFAULT_EPISODES_PATH.format(
                chunk_index=mc,
                file_index=mf,
            )
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        group.to_parquet(
            output,
            index=False,
        )

    print(
        f"[episodes-meta] wrote "
        f"{len(ep_df)} episodes"
    )

    return aggregate_stats(
        episodes_stats
    )


# --------------------------------------------------------------------------- #
# Info and stats
# --------------------------------------------------------------------------- #

def write_stats(
    new_root,
    dataset_stats,
    video_keys,
):
    serializable = {
        feature: {
            key: to_jsonable(value)
            for key, value in stats.items()
        }
        for feature, stats in dataset_stats.items()
    }

    # LeRobot expects camera keys to have stats
    # dictionaries when ImageNet stats are enabled.
    for camera in video_keys:

        if (
            camera not in serializable
            or not serializable[camera]
        ):
            serializable[camera] = {
                "mean": [
                    [[0.5]],
                    [[0.5]],
                    [[0.5]],
                ],
                "std": [
                    [[0.5]],
                    [[0.5]],
                    [[0.5]],
                ],
            }

    output = (
        new_root
        / "meta"
        / "stats.json"
    )

    output.write_text(
        json.dumps(
            serializable,
            indent=4,
        )
    )


def convert_info(
    root,
    new_root,
    data_file_size_in_mb,
    video_file_size_in_mb,
    chunk_size,
    total_episodes,
    total_frames,
    total_tasks,
    fps,
):
    info = json.loads(
        (
            root
            / "meta"
            / "info.json"
        ).read_text()
    )

    info["codebase_version"] = "v3.0"

    info.pop(
        "total_chunks",
        None,
    )

    info.pop(
        "total_videos",
        None,
    )

    info["chunks_size"] = (
        chunk_size
    )

    info["data_files_size_in_mb"] = (
        data_file_size_in_mb
    )

    info["video_files_size_in_mb"] = (
        video_file_size_in_mb
    )

    info["data_path"] = (
        DEFAULT_DATA_PATH
    )

    info["video_path"] = (
        DEFAULT_VIDEO_PATH
    )

    info["fps"] = int(fps)

    for key, feature in info.get(
        "features",
        {},
    ).items():

        if feature.get(
            "dtype"
        ) != "video":
            feature["fps"] = int(fps)

    info["total_episodes"] = (
        total_episodes
    )

    info["total_frames"] = (
        total_frames
    )

    info["total_tasks"] = (
        total_tasks
    )

    info["splits"] = {
        "train":
            f"0:{total_episodes}"
    }

    output = (
        new_root
        / "meta"
        / "info.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            info,
            indent=2,
        )
    )


# --------------------------------------------------------------------------- #
# Extra metadata
# --------------------------------------------------------------------------- #

def copy_extra_meta(
    root,
    new_root,
):
    for name in (
        "modality.json",
        "tasks.json",
        "tasks.jsonl",
    ):
        source = (
            root
            / "meta"
            / name
        )

        if source.exists():
            destination = new_root / "meta" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    source = (
        root
        / "meta"
        / "relative_stats.json"
    )

    if source.exists():
        shutil.copy2(
            source,
            new_root
            / "meta"
            / "relative_stats.json",
        )

    # ROS camera and robot topics are stored at the dataset root (not under
    # meta/). Preserve them so converted datasets keep their inference wiring.
    source = root / "ros2_topics.json"
    if source.is_file():
        shutil.copy2(source, new_root / "ros2_topics.json")


# --------------------------------------------------------------------------- #
# Final validation
# --------------------------------------------------------------------------- #

def validate_final_dataset(
    output,
    video_keys,
    fps,
    expected_total_frames,
    expected_total_episodes,
):
    print()
    print("=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------------- #
    # Info.
    # --------------------------------------------------------------- #

    info_path = (
        output
        / "meta"
        / "info.json"
    )

    if not info_path.exists():
        raise RuntimeError(
            "meta/info.json was not generated."
        )

    info = json.loads(
        info_path.read_text()
    )

    if info.get(
        "codebase_version"
    ) != "v3.0":
        raise RuntimeError(
            "Dataset is not marked v3.0."
        )

    if int(
        info.get("fps", -1)
    ) != fps:
        raise RuntimeError(
            "Dataset FPS is incorrect."
        )

    if int(
        info.get(
            "total_episodes",
            -1,
        )
    ) != expected_total_episodes:
        raise RuntimeError(
            "Dataset episode count is incorrect."
        )

    if int(
        info.get(
            "total_frames",
            -1,
        )
    ) != expected_total_frames:
        raise RuntimeError(
            "Dataset frame count is incorrect."
        )

    # --------------------------------------------------------------- #
    # Data timestamps.
    # --------------------------------------------------------------- #

    data_files = sorted(
        output.glob(
            "data/*/*.parquet"
        )
    )

    if not data_files:
        raise RuntimeError(
            "No output data parquet files."
        )

    all_data = []

    for path in data_files:
        df = pd.read_parquet(path)
        all_data.append(df)

    data = pd.concat(
        all_data,
        ignore_index=True,
    )

    if len(data) != expected_total_frames:
        raise RuntimeError(
            f"Final data frame count mismatch: "
            f"{len(data)} != "
            f"{expected_total_frames}"
        )

    timestamp_error = np.max(
        np.abs(
            data["timestamp"].to_numpy(
                dtype=np.float64
            )
            -
            (
                data["frame_index"].to_numpy(
                    dtype=np.float64
                )
                / float(fps)
            )
        )
    )

    print(
        f"[validate] maximum timestamp "
        f"error inside episodes: "
        f"{timestamp_error:.12f} s"
    )

    if timestamp_error > 1e-9:
        raise RuntimeError(
            "Parquet timestamps are not exact "
            "frame_index/FPS timestamps."
        )

    # --------------------------------------------------------------- #
    # Episode count.
    # --------------------------------------------------------------- #

    unique_episodes = sorted(
        data["episode_index"]
        .unique()
        .tolist()
    )

    expected_episodes = list(
        range(expected_total_episodes)
    )

    if unique_episodes != expected_episodes:
        raise RuntimeError(
            "Episode indices are not contiguous."
        )

    # --------------------------------------------------------------- #
    # Video validation.
    # --------------------------------------------------------------- #

    for camera in sorted(
        video_keys
    ):
        video_files = sorted(
            (
                output
                / "videos"
                / camera
            ).glob(
                "*/*.mp4"
            )
        )

        if not video_files:
            raise RuntimeError(
                f"No videos found for camera "
                f"{camera}"
            )

        for video_path in video_files:

            video_info = get_video_info(
                video_path
            )

            if abs(
                video_info["fps"] - fps
            ) > 0.01:
                raise RuntimeError(
                    f"Incorrect FPS in "
                    f"{video_path}: "
                    f"{video_info['fps']}"
                )

            pts = get_video_pts(
                video_path
            )

            if len(pts) != (
                video_info["frame_count"]
            ):
                raise RuntimeError(
                    f"PTS/frame count mismatch "
                    f"in {video_path}"
                )

            if not pts:
                raise RuntimeError(
                    f"No PTS found in "
                    f"{video_path}"
                )

            # ------------------------------------------------------- #
            # Check CFR PTS.
            # ------------------------------------------------------- #

            expected = (
                np.arange(
                    len(pts),
                    dtype=np.float64,
                )
                / float(fps)
            )

            actual = np.asarray(
                pts,
                dtype=np.float64,
            )

            max_error = np.max(
                np.abs(
                    actual - expected
                )
            )

            max_step_error = 0.0

            if len(actual) > 1:
                steps = np.diff(actual)

                max_step_error = np.max(
                    np.abs(
                        steps
                        -
                        (1.0 / float(fps))
                    )
                )

            print(
                f"[validate] {video_path}"
            )

            print(
                f"    frames       : "
                f"{len(actual)}"
            )

            print(
                f"    first PTS    : "
                f"{actual[0]:.9f}"
            )

            print(
                f"    last PTS     : "
                f"{actual[-1]:.9f}"
            )

            print(
                f"    max PTS err  : "
                f"{max_error:.12f} s"
            )

            print(
                f"    max step err : "
                f"{max_step_error:.12f} s"
            )

            # We use 0.1 ms as the target because
            # LeRobot's failing tolerance was 0.0001.
            if max_error > 0.00005:
                raise RuntimeError(
                    f"Generated video PTS are not "
                    f"accurate enough: "
                    f"{max_error} s"
                )

            if max_step_error > 0.00005:
                raise RuntimeError(
                    f"Generated video is not "
                    f"properly CFR: "
                    f"{max_step_error} s"
                )

    print()
    print(
        "FINAL VALIDATION: PASSED"
    )
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Main conversion
# --------------------------------------------------------------------------- #

def convert_dataset(
    root,
    output,
    data_file_size_in_mb,
    video_file_size_in_mb,
    chunk_size,
    episodes_per_meta_file,
    fps,
    crf,
    preset,
):
    root = root.resolve()
    output = output.resolve()

    if not root.exists():
        raise RuntimeError(
            f"Source dataset does not exist: "
            f"{root}"
        )

    if output.exists() and any(
        output.iterdir()
    ):
        raise RuntimeError(
            f"Output directory already exists "
            f"and is not empty:\n"
            f"{output}\n\n"
            f"Choose a new output directory."
        )

    # --------------------------------------------------------------- #
    # Read source info.
    # --------------------------------------------------------------- #

    info_path = (
        root
        / "meta"
        / "info.json"
    )

    if not info_path.exists():
        raise RuntimeError(
            f"Missing source info.json: "
            f"{info_path}"
        )

    # Some legacy datasets only store these metadata tables as parquet.
    # Materialize verified JSON sidecars before any conversion stage reads them.
    ensure_source_metadata(root)

    info = json.loads(
        info_path.read_text()
    )

    source_fps = int(
        info.get(
            "fps",
            fps,
        )
    )

    if source_fps != fps:
        raise RuntimeError(
            f"Source dataset FPS is "
            f"{source_fps}, but converter "
            f"was configured for {fps}."
        )

    video_keys = info.get(
        "video_keys"
    )

    if not video_keys:
        video_keys = info.get(
            "camera_keys",
            [],
        )

    if not video_keys:
        video_keys = [
            name
            for name, feature
            in info.get(
                "features",
                {},
            ).items()
            if feature.get(
                "dtype"
            ) == "video"
        ]

    feature_keys = (
        numeric_feature_keys(
            info
        )
    )

    keep_columns = {
        name
        for name, feature
        in info.get(
            "features",
            {},
        ).items()
        if feature.get(
            "dtype"
        ) not in (
            "video",
            "image",
        )
    }

    # --------------------------------------------------------------- #
    # Episode map.
    # --------------------------------------------------------------- #

    renumber_map = (
        build_renumber_map(
            root
        )
    )

    print()
    print("=" * 70)
    print("LE ROBOT V3.0 CONVERSION")
    print("=" * 70)

    print(
        f"Source          : {root}"
    )

    print(
        f"Output          : {output}"
    )

    print(
        f"Episodes        : "
        f"{len(renumber_map)}"
    )

    print(
        f"FPS             : {fps}"
    )

    print(
        f"Cameras         : "
        f"{video_keys}"
    )

    print(
        f"Video CRF       : {crf}"
    )

    print(
        f"Video preset    : {preset}"
    )

    print(
        "Video conversion: RE-ENCODE"
    )

    print(
        "Stream copy     : DISABLED"
    )

    print("=" * 70)
    print()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------- #
    # Tasks.
    # --------------------------------------------------------------- #

    convert_tasks(
        root,
        output,
    )

    # --------------------------------------------------------------- #
    # Data.
    # --------------------------------------------------------------- #

    (
        data_meta,
        episode_stats,
        total_frames,
    ) = convert_data(
        root=root,
        new_root=output,
        data_file_size_in_mb=(
            data_file_size_in_mb
        ),
        chunk_size=chunk_size,
        feature_keys=feature_keys,
        keep_columns=keep_columns,
        renumber_map=renumber_map,
        fps=fps,
    )

    # --------------------------------------------------------------- #
    # Videos.
    # --------------------------------------------------------------- #

    video_meta = convert_videos(
        root=root,
        new_root=output,
        video_keys=video_keys,
        video_file_size_in_mb=(
            video_file_size_in_mb
        ),
        chunk_size=chunk_size,
        renumber_map=renumber_map,
        fps=fps,
        crf=crf,
        preset=preset,
    )

    # --------------------------------------------------------------- #
    # Episode metadata.
    # --------------------------------------------------------------- #

    dataset_stats = (
        convert_episodes_metadata(
            root=root,
            new_root=output,
            data_meta=data_meta,
            episodes_stats=episode_stats,
            video_meta=video_meta,
            episodes_per_meta_file=(
                episodes_per_meta_file
            ),
            chunk_size=chunk_size,
        )
    )

    # --------------------------------------------------------------- #
    # Stats.
    # --------------------------------------------------------------- #

    write_stats(
        output,
        dataset_stats,
        video_keys,
    )

    # --------------------------------------------------------------- #
    # Info.
    # --------------------------------------------------------------- #

    total_tasks = len(ensure_task_sidecars(root))

    convert_info(
        root=root,
        new_root=output,
        data_file_size_in_mb=(
            data_file_size_in_mb
        ),
        video_file_size_in_mb=(
            video_file_size_in_mb
        ),
        chunk_size=chunk_size,
        total_episodes=len(
            data_meta
        ),
        total_frames=total_frames,
        total_tasks=total_tasks,
        fps=fps,
    )

    # --------------------------------------------------------------- #
    # Extra metadata.
    # --------------------------------------------------------------- #

    copy_extra_meta(
        root,
        output,
    )

    # --------------------------------------------------------------- #
    # Final validation.
    # --------------------------------------------------------------- #

    validate_final_dataset(
        output=output,
        video_keys=video_keys,
        fps=fps,
        expected_total_frames=(
            total_frames
        ),
        expected_total_episodes=len(
            data_meta
        ),
    )

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(
        f"Dataset: {output}"
    )

    print(
        f"Episodes: {len(data_meta)}"
    )

    print(
        f"Frames: {total_frames}"
    )

    print(
        "The source dataset was not modified."
    )

    print("=" * 70)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
    )

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Original v2.1 dataset root.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New v3.0 dataset root.",
    )

    parser.add_argument(
        "--data-file-size-in-mb",
        type=float,
        default=100.0,
        help="Approximate data parquet size.",
    )

    parser.add_argument(
        "--video-file-size-in-mb",
        type=float,
        default=200.0,
        help="Approximate video output size.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum files per chunk directory.",
    )

    parser.add_argument(
        "--episodes-per-meta-file",
        type=int,
        default=1_000_000,
        help="Episodes per meta/episodes parquet.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Dataset/video FPS.",
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality. Lower = higher quality/larger file.",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="x264 encoding preset.",
    )

    args = parser.parse_args()

    convert_dataset(
        root=args.root,
        output=args.output,
        data_file_size_in_mb=(
            args.data_file_size_in_mb
        ),
        video_file_size_in_mb=(
            args.video_file_size_in_mb
        ),
        chunk_size=args.chunk_size,
        episodes_per_meta_file=(
            args.episodes_per_meta_file
        ),
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()

