"""Convert the project's raw/legacy dataset directly into LeRobot v3.0.

This file intentionally combines the two previously separate conversion stages:

Stage 1 (metadata preparation)
    - normalize tasks.jsonl / episodes.jsonl
    - update info.json totals
    - optionally copy modality.json
    - generate stats.json
    - optionally generate relative_stats.json

Stage 2 (LeRobot v3.0 conversion)
    - rewrite episode parquet data into v3.0 chunked files
    - concatenate videos into v3.0 chunked files
    - generate v3.0 episodes metadata
    - write tasks/stats/info

The public entry point is:

    convert_dataset(root=..., ...)

The CLI is:

    python scripts/convert_dataset_to_v30.py --root /path/to/dataset

The script preserves the original dataset at <root>_old and replaces the
original root with the converted LeRobot v3.0 dataset, matching the behavior
of the original Stage-2 converter.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Iterable

import jsonlines
import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import Dataset, Features, Image
from lerobot.datasets import aggregate_stats
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.io_utils import (
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    load_info,
    load_json,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    DatasetInfo,
    update_chunk_file_indices,
)
from lerobot.datasets.video_utils import (
    concatenate_video_files,
    get_video_duration_in_s,
)
from lerobot.utils.utils import flatten_dict, init_logging


V30 = "v3.0"
STATS_FEATURE_KEYS = [
    "action",
    "observation.state",
    "observation.leader_state",
    "timestamp",
]

DEFAULT_CHUNKS_SIZE = 1024
DEFAULT_ACTION_HORIZON = 16
DEFAULT_RELATIVE_KEYS = ("left_arm", "right_arm")
STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")
STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"

logger = logging.getLogger(__name__)


# ============================================================================
# Stage 1: metadata preparation
# ============================================================================


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    """Resolve a dataset root containing meta/info.json."""
    root = Path(dataset_root).expanduser().resolve()

    if (root / "meta" / "info.json").exists():
        return root

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    nested_roots = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "meta" / "info.json").exists()
    )

    if len(nested_roots) == 1:
        return nested_roots[0]

    if len(nested_roots) > 1:
        raise FileNotFoundError(
            "Multiple LeRobot-style dataset roots found under "
            f"{root}: {', '.join(str(p) for p in nested_roots)}"
        )

    raise FileNotFoundError(
        f"Could not find meta/info.json under {root}. "
        "Pass the exact dataset root that contains the dataset."
    )


def episode_parquet_paths(dataset_root: Path) -> list[Path]:
    """Find source episode parquet files."""
    paths = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))

    if not paths:
        # Be slightly more tolerant of legacy/source layouts.  The Stage-2
        # conversion itself accepts any one-level chunk directory.
        paths = sorted(dataset_root.glob("data/*/*.parquet"))

    if not paths:
        raise FileNotFoundError(
            f"No episode parquet files found under {dataset_root / 'data'}."
        )

    return paths


def parse_episode_index(path: Path) -> int:
    stem = path.stem
    if stem.startswith("episode_"):
        return int(stem.split("_", 1)[1])
    raise ValueError(f"Could not parse episode index from filename: {path.name}")


def read_tasks(meta_dir: Path) -> list[dict[str, object]]:
    """Read tasks from either task.json or tasks.parquet."""
    task_json = meta_dir / "task.json"
    tasks_parquet = meta_dir / "tasks.parquet"
    tasks_jsonl = meta_dir / "tasks.jsonl"

    if task_json.exists():
        raw_tasks = json.loads(task_json.read_text())
    elif tasks_parquet.exists():
        raw_tasks = pd.read_parquet(tasks_parquet).to_dict(orient="records")
    elif tasks_jsonl.exists():
        with jsonlines.open(tasks_jsonl, "r") as reader:
            raw_tasks = list(reader)
    else:
        raise FileNotFoundError(
            f"Could not find task.json, tasks.parquet, or tasks.jsonl in {meta_dir}."
        )

    tasks: list[dict[str, object]] = []
    for row in raw_tasks:
        if "task_index" not in row or "task" not in row:
            raise ValueError(
                "Each task record must contain at least 'task_index' and 'task'. "
                f"Got keys: {list(row.keys())}"
            )
        task_index = int(row["task_index"])
        task_text = str(row["task"])
        prompt = str(row.get("prompt", task_text))
        tasks.append(
            {
                "task_index": task_index,
                "task": task_text,
                "prompt": prompt,
            }
        )

    tasks.sort(key=lambda item: int(item["task_index"]))
    return tasks


def write_tasks_jsonl(meta_dir: Path, tasks: list[dict[str, object]]) -> Path:
    path = meta_dir / "tasks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    return path


def build_episodes(episode_paths: list[Path]) -> list[dict[str, object]]:
    """Build episode metadata from source parquet files."""
    episodes: list[dict[str, object]] = []

    for parquet_path in episode_paths:
        df = pd.read_parquet(parquet_path)
        if len(df) == 0:
            raise ValueError(f"Empty episode parquet: {parquet_path}")

        file_episode_index = parse_episode_index(parquet_path)
        episode_index = (
            int(df["episode_index"].iloc[0])
            if "episode_index" in df.columns
            else file_episode_index
        )

        if episode_index != file_episode_index:
            raise ValueError(
                f"Episode index mismatch for {parquet_path.name}: "
                f"file says {file_episode_index}, parquet says {episode_index}"
            )

        episode: dict[str, object] = {
            "episode_index": episode_index,
            "length": int(len(df)),
        }

        for key in ("task_index", "task", "prompt"):
            if key in df.columns:
                value = df[key].iloc[0]
                episode[key] = int(value) if key == "task_index" else str(value)

        if "timestamp" in df.columns:
            episode["timestamp_start"] = float(df["timestamp"].iloc[0])
            episode["timestamp_end"] = float(df["timestamp"].iloc[-1])

        episodes.append(episode)

    episodes.sort(key=lambda item: int(item["episode_index"]))
    return episodes


def write_episodes_jsonl(meta_dir: Path, episodes: list[dict[str, object]]) -> Path:
    path = meta_dir / "episodes.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
    return path


def update_info_json(
    meta_dir: Path,
    chunks_size: int,
    tasks: list[dict[str, object]],
    episodes: list[dict[str, object]],
) -> tuple[Path, dict]:
    """Update dataset-wide counts in info.json."""
    path = meta_dir / "info.json"
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)

    info["chunks_size"] = int(info.get("chunks_size", chunks_size))
    info["total_episodes"] = len(episodes)
    info["total_frames"] = int(
        sum(int(ep["length"]) for ep in episodes)
    )
    info["total_tasks"] = len(tasks)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    return path, info


def copy_modality_template(
    meta_dir: Path,
    template_path: Path | None,
) -> Path | None:
    """Copy a GR00T-style modality.json if supplied."""
    if template_path is None:
        return None
    if not template_path.exists():
        raise FileNotFoundError(
            f"Modality template does not exist: {template_path}"
        )

    target_path = meta_dir / "modality.json"
    if template_path.resolve() != target_path.resolve():
        shutil.copy2(template_path, target_path)
    return target_path


def float_feature_names(info: dict) -> list[str]:
    """Every feature whose dtype contains 'float'."""
    features = info.get("features", {})
    return [
        name
        for name, spec in features.items()
        if "float" in str(spec.get("dtype", ""))
    ]


def column_to_matrix(column: pd.Series) -> np.ndarray:
    """Stack a parquet column into (n_rows, dim) float32."""
    values = column.to_numpy()
    if values.dtype == object:
        return np.vstack(
            [np.asarray(v, dtype=np.float32) for v in values]
        )
    return values.astype(np.float32).reshape(len(values), -1)


def compute_stats(
    episode_paths: list[Path],
    features: list[str],
) -> dict[str, dict[str, list]]:
    per_feature: dict[str, list[np.ndarray]] = {
        name: [] for name in features
    }

    for i, parquet_path in enumerate(episode_paths, start=1):
        df = pd.read_parquet(
            parquet_path,
            columns=features,
        )
        for name in features:
            per_feature[name].append(
                column_to_matrix(df[name])
            )

        if i % 100 == 0 or i == len(episode_paths):
            print(
                f"  stats: read {i}/{len(episode_paths)} episodes",
                flush=True,
            )

    stats: dict[str, dict[str, list]] = {}

    for name in features:
        data = np.concatenate(
            per_feature[name],
            axis=0,
        )
        per_feature[name] = []

        stats[name] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }

        print(
            f"  stats: computed {name} (dim={data.shape[1]})",
            flush=True,
        )

    return stats


def stats_are_valid(
    stats_path: Path,
    features: list[str],
) -> bool:
    if not stats_path.exists():
        return False

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except json.JSONDecodeError:
        return False

    for feature in features:
        entry = stats.get(feature)
        if not isinstance(entry, dict):
            return False
        if any(stat not in entry for stat in STAT_NAMES):
            return False

    return True


def generate_stats(
    dataset_root: Path,
    info: dict,
    episode_paths: list[Path],
    force: bool,
) -> Path:
    stats_path = dataset_root / "meta" / "stats.json"
    features = float_feature_names(info)

    if not features:
        raise ValueError(
            "No float features found in meta/info.json; cannot compute stats."
        )

    if not force and stats_are_valid(stats_path, features):
        print(
            f"  stats.json already valid for {len(features)} features, skipping."
        )
        return stats_path

    stats = compute_stats(
        episode_paths,
        features,
    )

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    return stats_path


def load_modality_ranges(
    modality_path: Path,
) -> dict[str, dict[str, tuple[int, int]]]:
    with open(modality_path, "r", encoding="utf-8") as f:
        modality = json.load(f)

    ranges: dict[str, dict[str, tuple[int, int]]] = {}

    for group in ("state", "action"):
        ranges[group] = {
            key: (
                int(spec["start"]),
                int(spec["end"]),
            )
            for key, spec in modality.get(group, {}).items()
        }

    return ranges


def relative_chunks_for_key(
    episode_paths: list[Path],
    state_slice: tuple[int, int],
    action_slice: tuple[int, int],
    horizon: int,
) -> np.ndarray:
    """Stack all relative action chunks as (n_chunks, horizon, dim)."""
    chunks: list[np.ndarray] = []
    total_skipped = 0

    for i, parquet_path in enumerate(episode_paths, start=1):
        df = pd.read_parquet(
            parquet_path,
            columns=[STATE_COLUMN, ACTION_COLUMN],
        )

        state = column_to_matrix(df[STATE_COLUMN])[
            :, state_slice[0] : state_slice[1]
        ]
        action = column_to_matrix(df[ACTION_COLUMN])[
            :, action_slice[0] : action_slice[1]
        ]

        usable = len(df) - (horizon - 1)
        if usable <= 0:
            total_skipped += 1
            continue

        windows = np.lib.stride_tricks.sliding_window_view(
            action,
            horizon,
            axis=0,
        )
        windows = np.transpose(
            windows[:usable],
            (0, 2, 1),
        )

        chunks.append(
            (windows - state[:usable, None, :]).astype(
                np.float32
            )
        )

        if i % 100 == 0 or i == len(episode_paths):
            print(
                f"  relative: read {i}/{len(episode_paths)} episodes",
                flush=True,
            )

    if not chunks:
        raise ValueError(
            f"No episode is long enough for an action horizon of {horizon} frames."
        )

    if total_skipped:
        print(
            f"  relative: skipped {total_skipped} episodes shorter than {horizon} frames"
        )

    return np.concatenate(
        chunks,
        axis=0,
    )


def generate_relative_stats(
    dataset_root: Path,
    episode_paths: list[Path],
    modality_path: Path,
    relative_keys: Iterable[str],
    horizon: int,
    force: bool,
) -> Path:
    stats_path = dataset_root / "meta" / "relative_stats.json"

    if stats_path.exists() and not force:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    else:
        stats = {}

    ranges = load_modality_ranges(
        modality_path
    )

    for key in sorted(relative_keys):
        if key in stats and not force:
            print(
                f"  relative_stats.json already has '{key}', skipping."
            )
            continue

        if key not in ranges["action"]:
            raise KeyError(
                f"Relative key '{key}' is not in the action section of "
                f"{modality_path}. Available: {sorted(ranges['action'])}"
            )

        state_slice = ranges["state"].get(
            key,
            ranges["action"][key],
        )
        action_slice = ranges["action"][key]

        print(
            f"  computing relative stats for '{key}' "
            f"(horizon={horizon})..."
        )

        data = relative_chunks_for_key(
            episode_paths,
            state_slice,
            action_slice,
            horizon,
        )

        stats[key] = {
            "max": np.max(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
        }

        print(
            f"  relative: '{key}' -> shape {data.shape[1:]} "
            f"from {len(data)} chunks"
        )

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    return stats_path


def prepare_metadata(
    dataset_root: Path,
    *,
    modality_template: str | Path | None,
    chunks_size: int,
    action_horizon: int,
    relative_keys: list[str],
    skip_stats: bool,
    skip_relative_stats: bool,
    force_stats: bool,
) -> None:
    """Run all Stage-1 preparation steps in one place."""
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_paths = episode_parquet_paths(
        dataset_root
    )

    tasks = read_tasks(meta_dir)
    episodes = build_episodes(episode_paths)

    info_path, info = update_info_json(
        meta_dir,
        chunks_size,
        tasks,
        episodes,
    )

    modality_path = copy_modality_template(
        meta_dir,
        Path(modality_template).expanduser().resolve()
        if modality_template
        else None,
    )

    tasks_path = write_tasks_jsonl(
        meta_dir,
        tasks,
    )
    episodes_path = write_episodes_jsonl(
        meta_dir,
        episodes,
    )

    print("Prepared dataset metadata:")
    print(f"  dataset_root: {dataset_root}")
    print(f"  info.json: {info_path}")
    print(
        "  modality.json: "
        f"{modality_path if modality_path else '(skipped)'}"
    )
    print(f"  tasks.jsonl: {tasks_path}")
    print(f"  episodes.jsonl: {episodes_path}")
    print(f"  total_episodes: {len(episodes)}")
    print(f"  total_frames: {info['total_frames']}")

    if not skip_stats:
        print("Generating stats.json...")
        generate_stats(
            dataset_root,
            info,
            episode_paths,
            force_stats,
        )

    if not skip_relative_stats:
        if not relative_keys:
            print("No relative keys requested; skipping relative_stats.json.")
        elif modality_path is None:
            print(
                "No modality.json supplied; skipping relative_stats.json."
            )
        else:
            print("Generating relative_stats.json...")
            generate_relative_stats(
                dataset_root,
                episode_paths,
                modality_path,
                relative_keys,
                action_horizon,
                force_stats,
            )


# ============================================================================
# Stage 2: LeRobot v3.0 conversion
# ============================================================================


def robo_load_episodes(root: Path) -> dict[int, dict]:
    """Read meta/episodes.jsonl and densify episode indices."""
    items = []
    with jsonlines.open(root / "meta" / "episodes.jsonl", "r") as reader:
        items = list(reader)

    episodes: dict[int, dict] = {}

    for ep_idx, item in enumerate(
        sorted(items, key=lambda x: int(x["episode_index"]))
    ):
        episodes[ep_idx] = {
            "episode_index": ep_idx,
            "tasks": [item["task"]],
            "length": item["length"],
        }

    return episodes


def get_image_keys(root: Path) -> list[str]:
    info = load_info(root)
    return [
        key
        for key, ft in info.features.items()
        if ft["dtype"] == "image"
    ]


def get_video_keys(root: Path) -> list[str]:
    info = load_info(root)
    return [
        key
        for key, ft in info.features.items()
        if ft["dtype"] == "video"
    ]


def load_and_patch_episode_df(
    ep_path: Path,
    ep_idx: int,
    index_start: int,
    declared_columns: list[str],
    vector_keys: list[str],
    fps: float,
) -> pd.DataFrame:
    """Read and patch one source episode parquet for v3.0."""
    df = pd.read_parquet(
        ep_path,
        columns=declared_columns,
    )

    df["episode_index"] = ep_idx
    df["index"] = index_start + np.arange(len(df))

    for key in vector_keys:
        df[key] = df[key].apply(
            lambda arr: np.asarray(
                arr,
                dtype=np.float32,
            )
        )

    if "timestamp" in df.columns:
        if "frame_index" not in df.columns:
            raise ValueError(
                f"Dataset has timestamp but no frame_index: {ep_path}"
            )
        df["timestamp"] = (
            df["frame_index"].astype(np.float32) / fps
        )

    return df


def concat_data_files(
    dataframes: list[pd.DataFrame],
    new_root: Path,
    chunk_idx: int,
    file_idx: int,
    image_keys: list[str],
) -> None:
    concatenated_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    path = new_root / DEFAULT_DATA_PATH.format(
        chunk_index=chunk_idx,
        file_index=file_idx,
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if image_keys:
        schema = pa.Schema.from_pandas(concatenated_df)
        features = Features.from_arrow_schema(schema)
        for key in image_keys:
            features[key] = Image()
        schema = features.arrow_schema
    else:
        schema = None

    concatenated_df.to_parquet(
        path,
        index=False,
        schema=schema,
    )


def convert_data(
    root: Path,
    new_root: Path,
    data_file_size_in_mb: int,
) -> list[dict]:
    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))

    if not ep_paths:
        raise FileNotFoundError(
            f"No source episode parquet files under {data_dir}."
        )

    image_keys = get_image_keys(root)
    info = load_info(root)

    declared_columns = [
        key
        for key, ft in info.features.items()
        if ft["dtype"] != "video"
    ]
    vector_keys = [
        key
        for key, ft in info.features.items()
        if ft["dtype"] not in ("video", "string")
        and len(ft["shape"]) >= 1
        and ft["shape"][0] > 1
    ]
    fps = info.fps

    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    num_frames = 0
    dfs_to_cat: list[pd.DataFrame] = []
    episodes_metadata: list[dict] = []

    logging.info(
        f"Converting data files from {len(ep_paths)} episodes"
    )

    for ep_idx, ep_path in enumerate(
        __import__("tqdm").tqdm(
            ep_paths,
            desc="convert data files",
        )
    ):
        ep_size_in_mb = get_parquet_file_size_in_mb(ep_path)
        ep_df = load_and_patch_episode_df(
            ep_path,
            ep_idx,
            num_frames,
            declared_columns,
            vector_keys,
            fps,
        )
        ep_num_frames = len(ep_df)

        if (
            size_in_mb + ep_size_in_mb >= data_file_size_in_mb
            and dfs_to_cat
        ):
            concat_data_files(
                dfs_to_cat,
                new_root,
                chunk_idx,
                file_idx,
                image_keys,
            )
            chunk_idx, file_idx = update_chunk_file_indices(
                chunk_idx,
                file_idx,
                DEFAULT_CHUNK_SIZE,
            )
            size_in_mb = 0
            dfs_to_cat = []

        ep_metadata = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": num_frames,
            "dataset_to_index": num_frames + ep_num_frames,
        }

        size_in_mb += ep_size_in_mb
        num_frames += ep_num_frames
        episodes_metadata.append(ep_metadata)
        dfs_to_cat.append(ep_df)

    if dfs_to_cat:
        concat_data_files(
            dfs_to_cat,
            new_root,
            chunk_idx,
            file_idx,
            image_keys,
        )

    return episodes_metadata


def convert_videos_of_camera(
    root: Path,
    new_root: Path,
    video_key: str,
    video_file_size_in_mb: int,
) -> list[dict]:
    videos_dir = root / "videos"
    ep_paths = sorted(
        videos_dir.glob(f"{video_key}/*/*.mp4")
    )

    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    duration_in_s = 0.0
    paths_to_cat: list[Path] = []
    episodes_metadata: list[dict] = []

    import tqdm

    for ep_path in tqdm.tqdm(
        ep_paths,
        desc=f"convert videos of {video_key}",
    ):
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        if (
            size_in_mb + ep_size_in_mb >= video_file_size_in_mb
            and paths_to_cat
        ):
            concatenate_video_files(
                paths_to_cat,
                new_root
                / DEFAULT_VIDEO_PATH.format(
                    video_key=video_key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                ),
            )

            for i, _ in enumerate(paths_to_cat):
                past_ep_idx = ep_idx - len(paths_to_cat) + i
                episodes_metadata[past_ep_idx][
                    f"videos/{video_key}/chunk_index"
                ] = chunk_idx
                episodes_metadata[past_ep_idx][
                    f"videos/{video_key}/file_index"
                ] = file_idx

            chunk_idx, file_idx = update_chunk_file_indices(
                chunk_idx,
                file_idx,
                DEFAULT_CHUNK_SIZE,
            )
            size_in_mb = 0
            duration_in_s = 0.0
            paths_to_cat = []

        ep_metadata = {
            "episode_index": ep_idx,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": duration_in_s,
            f"videos/{video_key}/to_timestamp": (
                duration_in_s + ep_duration_in_s
            ),
        }
        episodes_metadata.append(ep_metadata)

        paths_to_cat.append(ep_path)
        size_in_mb += ep_size_in_mb
        duration_in_s += ep_duration_in_s
        ep_idx += 1

    if paths_to_cat:
        concatenate_video_files(
            paths_to_cat,
            new_root
            / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            ),
        )

        for i, _ in enumerate(paths_to_cat):
            past_ep_idx = ep_idx - len(paths_to_cat) + i
            episodes_metadata[past_ep_idx][
                f"videos/{video_key}/chunk_index"
            ] = chunk_idx
            episodes_metadata[past_ep_idx][
                f"videos/{video_key}/file_index"
            ] = file_idx

    return episodes_metadata


def convert_videos(
    root: Path,
    new_root: Path,
    video_file_size_in_mb: int,
) -> list[dict] | None:
    video_keys = sorted(get_video_keys(root))
    if not video_keys:
        return None

    eps_metadata_per_cam = [
        convert_videos_of_camera(
            root,
            new_root,
            camera,
            video_file_size_in_mb,
        )
        for camera in video_keys
    ]

    num_eps_per_cam = [
        len(eps_cam_map)
        for eps_cam_map in eps_metadata_per_cam
    ]

    if len(set(num_eps_per_cam)) != 1:
        raise ValueError(
            "All cameras must have the same number of episodes: "
            f"{num_eps_per_cam}"
        )

    episodes_metadata: list[dict] = []
    num_cameras = len(video_keys)
    num_episodes = num_eps_per_cam[0]

    for ep_idx in range(num_episodes):
        ep_ids = [
            eps_metadata_per_cam[cam_idx][ep_idx]["episode_index"]
            for cam_idx in range(num_cameras)
        ]
        ep_ids.append(ep_idx)

        if len(set(ep_ids)) != 1:
            raise ValueError(
                f"All episode indices need to match: {ep_ids}"
            )

        ep_dict: dict = {}
        for cam_idx in range(num_cameras):
            ep_dict.update(
                eps_metadata_per_cam[cam_idx][ep_idx]
            )
        episodes_metadata.append(ep_dict)

    return episodes_metadata


def compute_robo_episodes_stats(root: Path) -> dict:
    """Compute per-episode stats directly from source data."""
    info = load_info(root)
    features_subset = {
        key: info.features[key]
        for key in STATS_FEATURE_KEYS
    }

    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))
    fps = info.fps

    episodes_stats: dict = {}

    import tqdm

    for ep_idx, ep_path in enumerate(
        tqdm.tqdm(
            ep_paths,
            desc="compute per-episode stats",
        )
    ):
        required_columns = list(
            set(STATS_FEATURE_KEYS + ["frame_index"])
        )
        df = pd.read_parquet(
            ep_path,
            columns=required_columns,
        )

        episode_data: dict = {}

        for key in STATS_FEATURE_KEYS:
            if key == "timestamp":
                episode_data[key] = (
                    df["frame_index"]
                    .to_numpy()
                    / fps
                ).astype(np.float32)
                continue

            shape = features_subset[key]["shape"]
            if len(shape) == 1 and shape[0] > 1:
                episode_data[key] = np.stack(
                    df[key].to_numpy()
                ).astype(np.float32)
            else:
                episode_data[key] = (
                    df[key]
                    .to_numpy()
                    .astype(np.float32)
                )

        episodes_stats[ep_idx] = compute_episode_stats(
            episode_data,
            features_subset,
        )

    return dict(
        sorted(
            episodes_stats.items()
        )
    )


def generate_episode_metadata_dict(
    episodes_legacy_metadata: dict,
    episodes_metadata: list[dict],
    episodes_stats: dict,
    episodes_videos: list[dict] | None = None,
):
    num_episodes = len(episodes_metadata)
    episodes_legacy_metadata_vals = list(
        episodes_legacy_metadata.values()
    )
    episodes_stats_vals = list(
        episodes_stats.values()
    )
    episodes_stats_keys = list(
        episodes_stats.keys()
    )

    for i in range(num_episodes):
        ep_legacy_metadata = (
            episodes_legacy_metadata_vals[i]
        )
        ep_metadata = episodes_metadata[i]
        ep_stats = episodes_stats_vals[i]

        ep_ids_set = {
            ep_legacy_metadata["episode_index"],
            ep_metadata["episode_index"],
            episodes_stats_keys[i],
        }

        if episodes_videos is None:
            ep_video = {}
        else:
            ep_video = episodes_videos[i]
            ep_ids_set.add(
                ep_video["episode_index"]
            )

        if len(ep_ids_set) != 1:
            raise ValueError(
                f"Number of episode indices is not the same: {ep_ids_set}"
            )

        ep_dict = {
            **ep_metadata,
            **ep_video,
            **ep_legacy_metadata,
            **flatten_dict({"stats": ep_stats}),
        }
        ep_dict["meta/episodes/chunk_index"] = 0
        ep_dict["meta/episodes/file_index"] = 0
        yield ep_dict


def convert_episodes_metadata(
    root: Path,
    new_root: Path,
    episodes_metadata: list[dict],
    episodes_video_metadata: list[dict] | None = None,
) -> None:
    logging.info(
        f"Converting episodes metadata from {root} to {new_root}"
    )

    episodes_legacy_metadata = robo_load_episodes(root)
    episodes_stats = compute_robo_episodes_stats(root)

    num_eps_set = {
        len(episodes_legacy_metadata),
        len(episodes_metadata),
        len(episodes_stats),
    }

    if episodes_video_metadata is not None:
        num_eps_set.add(
            len(episodes_video_metadata)
        )

    if len(num_eps_set) != 1:
        raise ValueError(
            f"Number of episodes is not the same: {num_eps_set}"
        )

    ds_episodes = Dataset.from_generator(
        lambda: generate_episode_metadata_dict(
            episodes_legacy_metadata,
            episodes_metadata,
            episodes_stats,
            episodes_video_metadata,
        )
    )

    write_episodes(
        ds_episodes,
        new_root,
    )

    stats = aggregate_stats(
        list(episodes_stats.values())
    )
    write_stats(
        stats,
        new_root,
    )


def convert_tasks(
    root: Path,
    new_root: Path,
) -> None:
    logging.info(
        f"Converting tasks from {root} to {new_root}"
    )

    tasks = []
    with jsonlines.open(root / "meta" / "tasks.jsonl", "r") as reader:
        tasks = list(reader)

    tasks = {
        int(item["task_index"]): str(item["task"])
        for item in sorted(
            tasks,
            key=lambda x: int(x["task_index"]),
        )
    }

    df_tasks = pd.DataFrame(
        {
            "task_index": tasks.keys(),
            "task": tasks.values(),
        }
    )
    write_tasks(
        df_tasks,
        new_root,
    )


def convert_info(
    root: Path,
    new_root: Path,
    data_file_size_in_mb: int,
    video_file_size_in_mb: int,
) -> None:
    info = load_json(
        root / "meta" / "info.json"
    )

    info["codebase_version"] = V30

    for legacy_key in (
        "total_chunks",
        "total_videos",
    ):
        info.pop(legacy_key, None)

    info["data_files_size_in_mb"] = (
        data_file_size_in_mb
    )
    info["video_files_size_in_mb"] = (
        video_file_size_in_mb
    )
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = (
        DEFAULT_VIDEO_PATH
        if info.get("video_path")
        else None
    )
    info["fps"] = int(info["fps"])

    for key in info["features"]:
        if info["features"][key]["dtype"] == "video":
            continue
        info["features"][key]["fps"] = info["fps"]

    dataset_info = DatasetInfo.from_dict(info)
    write_info(
        dataset_info,
        new_root,
    )


def convert_dataset(
    root: str | Path,
    *,
    data_file_size_in_mb: int | None = None,
    video_file_size_in_mb: int | None = None,
    modality_template: str | Path | None = None,
    chunks_size: int = DEFAULT_CHUNKS_SIZE,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    relative_keys: list[str] | None = None,
    skip_stats: bool = False,
    skip_relative_stats: bool = True,
    force_stats: bool = False,
) -> Path:
    """Run the complete raw -> LeRobot v3.0 conversion."""
    if data_file_size_in_mb is None:
        data_file_size_in_mb = (
            DEFAULT_DATA_FILE_SIZE_IN_MB
        )
    if video_file_size_in_mb is None:
        video_file_size_in_mb = (
            DEFAULT_VIDEO_FILE_SIZE_IN_MB
        )
    if relative_keys is None:
        relative_keys = list(
            DEFAULT_RELATIVE_KEYS
        )

    root = resolve_dataset_root(root)
    old_root = root.parent / f"{root.name}_old"
    new_root = root.parent / f"{root.name}_v30"

    if old_root.is_dir():
        raise FileExistsError(
            f"Backup dir already exists at {old_root}. "
            "Remove or move it aside before re-running."
        )

    if new_root.is_dir():
        shutil.rmtree(new_root)

    print("=" * 72)
    print("VLA DATASET CONVERSION: ")
    print("=" * 72)
    print(f"Input: {root}")

    # ------------------------------------------------------------------
    # Stage 1: metadata/stats preparation
    # ------------------------------------------------------------------
    print("\n[Stage 1/2] Preparing metadata and statistics...")
    prepare_metadata(
        root,
        modality_template=modality_template,
        chunks_size=chunks_size,
        action_horizon=action_horizon,
        relative_keys=relative_keys,
        skip_stats=skip_stats,
        skip_relative_stats=skip_relative_stats,
        force_stats=force_stats,
    )

    # ------------------------------------------------------------------
    # Stage 2: actual v3.0 conversion
    # ------------------------------------------------------------------
    print("\n[Stage 2/2] Converting data/video/metadata..")

    convert_info(
        root,
        new_root,
        data_file_size_in_mb,
        video_file_size_in_mb,
    )

    convert_tasks(
        root,
        new_root,
    )

    episodes_metadata = convert_data(
        root,
        new_root,
        data_file_size_in_mb,
    )

    episodes_videos_metadata = convert_videos(
        root,
        new_root,
        video_file_size_in_mb,
    )

    convert_episodes_metadata(
        root,
        new_root,
        episodes_metadata,
        episodes_videos_metadata,
    )

    # Preserve source and promote v3.0 output to the original path.
    shutil.move(
        str(root),
        str(old_root),
    )
    shutil.move(
        str(new_root),
        str(root),
    )

    print("\n" + "=" * 72)
    print("CONVERSION COMPLETE")
    print("=" * 72)
    print(f"dataset: {root}")
    print(f"Original backup:       {old_root}")

    return root


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the project's raw/legacy dataset directly to "
            "LeRobot v3.0."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root containing data/ and meta/.",
    )
    parser.add_argument(
        "--modality-template",
        default=None,
        help="Optional GR00T-style modality.json template.",
    )
    parser.add_argument(
        "--chunks-size",
        type=int,
        default=DEFAULT_CHUNKS_SIZE,
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
    )
    parser.add_argument(
        "--relative-keys",
        nargs="*",
        default=list(DEFAULT_RELATIVE_KEYS),
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
    )
    parser.add_argument(
        "--skip-relative-stats",
        action="store_true",
    )
    parser.add_argument(
        "--force-stats",
        action="store_true",
    )
    parser.add_argument(
        "--data-file-size-in-mb",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--video-file-size-in-mb",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> int:
    init_logging()
    args = parse_args()

    convert_dataset(
        root=args.root,
        data_file_size_in_mb=args.data_file_size_in_mb,
        video_file_size_in_mb=args.video_file_size_in_mb,
        modality_template=args.modality_template,
        chunks_size=args.chunks_size,
        action_horizon=args.action_horizon,
        relative_keys=args.relative_keys,
        skip_stats=args.skip_stats,
        skip_relative_stats=args.skip_relative_stats,
        force_stats=args.force_stats,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
