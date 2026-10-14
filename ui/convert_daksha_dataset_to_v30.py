#!/usr/bin/env python3
"""Convert a Daksha-recorded dataset (v2.1-ish layout, GR00T-style exporter) to real LeRobot v3.0 format.

The Daksha recorder writes a layout that looks like LeRobot v2.1 (per-episode parquet/mp4 files,
meta/episodes.jsonl, meta/tasks.jsonl, meta/stats.json) but deviates from it in ways the stock
`lerobot.scripts.convert_dataset_v21_to_v30` doesn't handle:
  - meta/info.json has no `total_chunks` / `total_videos` keys (that script does `del info[...]`)
  - meta/info.json's `codebase_version` is the pip package version ("0.6.2"), not "v2.1"
  - there is no meta/episodes_stats.jsonl (per-episode stats are recomputed here instead)
  - meta/episodes.jsonl entries use singular `task`/`task_index`, not the v2.1 `tasks: [...]` list
  - videos are laid out as videos/<camera>/chunk-XXX/episode_XXXXXX.mp4, not videos/chunk-XXX/<camera>/...

Converts in place: the original dataset is preserved untouched at `<root>_old`, and the new v3.0
dataset is written to `<root>`.

Usage:
    python examples/daksha/convert_daksha_dataset_to_v30.py --root=/home/s1/Desktop/datasets/pick_place_brown_300
"""

import argparse
import logging
import shutil
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import pyarrow as pa
import tqdm
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
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
from lerobot.utils.utils import flatten_dict, init_logging

V30 = "v3.0"
STATS_FEATURE_KEYS = ["action", "observation.state", "observation.leader_state", "timestamp"]

logger = logging.getLogger(__name__)


def load_jsonlines(fpath: Path) -> list:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def daksha_load_episodes(root: Path) -> dict:
    """Read meta/episodes.jsonl, normalizing the singular `task` field into a `tasks` list.

    Daksha's episode_index values have gaps (discarded takes during recording were removed but
    not renumbered), while v3.0 expects a dense 0..N-1 range matching the position of each
    episode's file in sorted order (the same order convert_data() and
    compute_daksha_episodes_stats() enumerate in). So episodes are renumbered positionally here.
    """
    if (root / "meta/episodes.jsonl").exists():
        items = load_jsonlines(root / "meta/episodes.jsonl")
    else:
        paths = sorted((root / "meta/episodes").glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No episode metadata found in {root}")
        items = pd.concat([pd.read_parquet(path) for path in paths]).to_dict("records")
    episodes = {}
    for ep_idx, item in enumerate(sorted(items, key=lambda x: x["episode_index"])):
        episodes[ep_idx] = {
            "episode_index": ep_idx,
            "tasks": [item["task"]],
            "length": item["length"],
        }
    return episodes


def get_image_keys(root: Path) -> list[str]:
    info = load_info(root)
    return [key for key, ft in info.features.items() if ft["dtype"] == "image"]


def get_video_keys(root: Path) -> list[str]:
    info = load_info(root)
    return [key for key, ft in info.features.items() if ft["dtype"] == "video"]


def load_and_patch_episode_df(ep_path, ep_idx, index_start, declared_columns, vector_keys, fps):
    """Read one Daksha episode parquet and patch it to match v3.0 expectations.

    - Projects down to only the columns declared in info.json's features (Daksha's parquets carry
      extra un-declared columns like prompt/task_label/joint names that the v3.0 reader can't cast).
    - Renumbers `episode_index` to the dense positional index (Daksha's original episode_index has
      gaps from discarded takes) and `index` to a dataset-global contiguous counter.
    - Rewrites `timestamp` as episode-relative seconds (Daksha stores raw wall-clock epoch time).
    - Casts vector/scalar float columns to float32 to match the declared feature dtype.
    """
    df = pd.read_parquet(ep_path, columns=declared_columns)
    df["episode_index"] = ep_idx
    df["index"] = index_start + np.arange(len(df))
    for key in vector_keys:
        df[key] = df[key].apply(lambda arr: np.asarray(arr, dtype=np.float32))
    if "timestamp" in df.columns:
        df["timestamp"] = df["frame_index"].astype(np.float32) / fps
    return df


def concat_data_files(dataframes, new_root, chunk_idx, file_idx, image_keys):
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(image_keys) > 0:
        schema = pa.Schema.from_pandas(concatenated_df)
        features = Features.from_arrow_schema(schema)
        for key in image_keys:
            features[key] = Image()
        schema = features.arrow_schema
    else:
        schema = None

    concatenated_df.to_parquet(path, index=False, schema=schema)


def convert_data(root: Path, new_root: Path, data_file_size_in_mb: int):
    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))
    image_keys = get_image_keys(root)

    info = load_info(root)
    declared_columns = [key for key, ft in info.features.items() if ft["dtype"] != "video"]
    vector_keys = [key for key, ft in info.features.items() if ft["dtype"] not in ("video", "string") and len(ft["shape"]) >= 1 and ft["shape"][0] > 1]
    fps = info.fps

    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    num_frames = 0
    dfs_to_cat = []
    episodes_metadata = []

    logging.info(f"Converting data files from {len(ep_paths)} episodes")

    for ep_idx, ep_path in enumerate(tqdm.tqdm(ep_paths, desc="convert data files")):
        ep_size_in_mb = get_parquet_file_size_in_mb(ep_path)
        ep_df = load_and_patch_episode_df(ep_path, ep_idx, num_frames, declared_columns, vector_keys, fps)
        ep_num_frames = len(ep_df)

        if size_in_mb + ep_size_in_mb >= data_file_size_in_mb and len(dfs_to_cat) > 0:
            concat_data_files(dfs_to_cat, new_root, chunk_idx, file_idx, image_keys)
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
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
        concat_data_files(dfs_to_cat, new_root, chunk_idx, file_idx, image_keys)

    return episodes_metadata


def convert_videos_of_camera(root: Path, new_root: Path, video_key: str, video_file_size_in_mb: int):
    # Daksha layout: videos/<video_key>/chunk-XXX/episode_XXXXXX.mp4 (camera dir before chunk dir).
    videos_dir = root / "videos"
    ep_paths = sorted(videos_dir.glob(f"{video_key}/*/*.mp4"))

    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    duration_in_s = 0.0
    paths_to_cat = []
    episodes_metadata = []

    for ep_path in tqdm.tqdm(ep_paths, desc=f"convert videos of {video_key}"):
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        if size_in_mb + ep_size_in_mb >= video_file_size_in_mb and len(paths_to_cat) > 0:
            concatenate_video_files(
                paths_to_cat,
                new_root
                / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx),
            )
            for i, _ in enumerate(paths_to_cat):
                past_ep_idx = ep_idx - len(paths_to_cat) + i
                episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
                episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0
            duration_in_s = 0.0
            paths_to_cat = []

        ep_metadata = {
            "episode_index": ep_idx,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": duration_in_s,
            f"videos/{video_key}/to_timestamp": duration_in_s + ep_duration_in_s,
        }
        episodes_metadata.append(ep_metadata)

        paths_to_cat.append(ep_path)
        size_in_mb += ep_size_in_mb
        duration_in_s += ep_duration_in_s
        ep_idx += 1

    if paths_to_cat:
        concatenate_video_files(
            paths_to_cat,
            new_root / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx),
        )
        for i, _ in enumerate(paths_to_cat):
            past_ep_idx = ep_idx - len(paths_to_cat) + i
            episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
            episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

    return episodes_metadata


def convert_videos(root: Path, new_root: Path, video_file_size_in_mb: int):
    video_keys = sorted(get_video_keys(root))
    if len(video_keys) == 0:
        return None

    eps_metadata_per_cam = [
        convert_videos_of_camera(root, new_root, camera, video_file_size_in_mb) for camera in video_keys
    ]

    num_eps_per_cam = [len(eps_cam_map) for eps_cam_map in eps_metadata_per_cam]
    if len(set(num_eps_per_cam)) != 1:
        raise ValueError(f"All cams dont have same number of episodes ({num_eps_per_cam}).")

    episodes_metadata = []
    num_cameras = len(video_keys)
    num_episodes = num_eps_per_cam[0]
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="convert videos"):
        ep_ids = [eps_metadata_per_cam[cam_idx][ep_idx]["episode_index"] for cam_idx in range(num_cameras)]
        ep_ids += [ep_idx]
        if len(set(ep_ids)) != 1:
            raise ValueError(f"All episode indices need to match ({ep_ids}).")

        ep_dict = {}
        for cam_idx in range(num_cameras):
            ep_dict.update(eps_metadata_per_cam[cam_idx][ep_idx])
        episodes_metadata.append(ep_dict)

    return episodes_metadata


def compute_daksha_episodes_stats(root: Path) -> dict:
    """Compute per-episode stats directly from the raw data (no episodes_stats.jsonl available)."""
    info = load_info(root)
    features_subset = {key: info.features[key] for key in STATS_FEATURE_KEYS}

    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))

    fps = info.fps
    episodes_stats = {}
    for ep_idx, ep_path in enumerate(tqdm.tqdm(ep_paths, desc="compute per-episode stats")):
        df = pd.read_parquet(ep_path, columns=STATS_FEATURE_KEYS + ["frame_index"])

        episode_data = {}
        for key in STATS_FEATURE_KEYS:
            if key == "timestamp":
                # Daksha's "timestamp" is raw wall-clock epoch time; use episode-relative time
                # to match what convert_data() writes to the actual data files.
                episode_data[key] = (df["frame_index"].to_numpy() / fps).astype(np.float32)
                continue
            shape = features_subset[key]["shape"]
            if len(shape) == 1 and shape[0] > 1:
                episode_data[key] = np.stack(df[key].to_numpy()).astype(np.float32)
            else:
                episode_data[key] = df[key].to_numpy().astype(np.float32)

        episodes_stats[ep_idx] = compute_episode_stats(episode_data, features_subset)

    return dict(sorted(episodes_stats.items()))


def generate_episode_metadata_dict(episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_videos=None):
    num_episodes = len(episodes_metadata)
    episodes_legacy_metadata_vals = list(episodes_legacy_metadata.values())
    episodes_stats_vals = list(episodes_stats.values())
    episodes_stats_keys = list(episodes_stats.keys())

    for i in range(num_episodes):
        ep_legacy_metadata = episodes_legacy_metadata_vals[i]
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
            ep_ids_set.add(ep_video["episode_index"])

        if len(ep_ids_set) != 1:
            raise ValueError(f"Number of episodes is not the same ({ep_ids_set}).")

        ep_dict = {**ep_metadata, **ep_video, **ep_legacy_metadata, **flatten_dict({"stats": ep_stats})}
        ep_dict["meta/episodes/chunk_index"] = 0
        ep_dict["meta/episodes/file_index"] = 0
        yield ep_dict


def convert_episodes_metadata(root, new_root, episodes_metadata, episodes_video_metadata=None):
    logging.info(f"Converting episodes metadata from {root} to {new_root}")

    episodes_legacy_metadata = daksha_load_episodes(root)
    episodes_stats = compute_daksha_episodes_stats(root)

    num_eps_set = {len(episodes_legacy_metadata), len(episodes_metadata), len(episodes_stats)}
    if episodes_video_metadata is not None:
        num_eps_set.add(len(episodes_video_metadata))

    if len(num_eps_set) != 1:
        raise ValueError(f"Number of episodes is not the same ({num_eps_set}).")

    ds_episodes = Dataset.from_generator(
        lambda: generate_episode_metadata_dict(
            episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_video_metadata
        )
    )
    write_episodes(ds_episodes, new_root)

    stats = aggregate_stats(list(episodes_stats.values()))
    write_stats(stats, new_root)


def convert_tasks(root, new_root):
    logging.info(f"Converting tasks from {root} to {new_root}")
    if (root / "meta/tasks.jsonl").exists():
        tasks = load_jsonlines(root / "meta/tasks.jsonl")
    else:
        tasks = pd.read_parquet(root / "meta/tasks.parquet").to_dict("records")
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    df_tasks = pd.DataFrame({"task_index": tasks.keys()}, index=pd.Index(tasks.values(), name="task"))
    write_tasks(df_tasks, new_root)


def convert_info(root, new_root, data_file_size_in_mb, video_file_size_in_mb):
    info = load_json(root / "meta/info.json")
    info["codebase_version"] = V30
    # total_chunks/total_videos are v2.1-only bookkeeping fields with no v3.0 equivalent.
    # camera_keys/video_keys/use_videos/repo_id aren't part of DatasetInfo (lerobot warns and
    # ignores them), but other Daksha tooling (examples/daksha/smolvla/check_dataset_smolvla.py,
    # ui/inference_client.py) reads them directly off info.json, so they're kept for compatibility.
    for legacy_key in ("total_chunks", "total_videos"):
        info.pop(legacy_key, None)
    info["data_files_size_in_mb"] = data_file_size_in_mb
    info["video_files_size_in_mb"] = video_file_size_in_mb
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH if info.get("video_path") else None
    info["fps"] = int(info["fps"])
    for key in info["features"]:
        if info["features"][key]["dtype"] == "video":
            continue
        info["features"][key]["fps"] = info["fps"]

    dataset_info = DatasetInfo.from_dict(info)
    write_info(dataset_info, new_root)


def convert_dataset(root: str | Path, data_file_size_in_mb: int | None = None, video_file_size_in_mb: int | None = None):
    if data_file_size_in_mb is None:
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_file_size_in_mb is None:
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    old_root = root.parent / f"{root.name}_old"
    new_root = root.parent / f"{root.name}_v30"

    if old_root.is_dir():
        raise FileExistsError(
            f"Backup dir already exists at {old_root}. Remove or move it aside before re-running."
        )
    if new_root.exists():
        raise FileExistsError(f"Conversion directory already exists at {new_root}")

    convert_info(root, new_root, data_file_size_in_mb, video_file_size_in_mb)
    convert_tasks(root, new_root)
    episodes_metadata = convert_data(root, new_root, data_file_size_in_mb)
    episodes_videos_metadata = convert_videos(root, new_root, video_file_size_in_mb)
    convert_episodes_metadata(root, new_root, episodes_metadata, episodes_videos_metadata)

    shutil.move(str(root), str(old_root))
    shutil.move(str(new_root), str(root))
    logging.info(f"Done. Original dataset preserved at {old_root}. Converted v3.0 dataset at {root}.")


if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Local dataset directory (contains meta/, data/, videos/).")
    parser.add_argument("--data-file-size-in-mb", type=int, default=None)
    parser.add_argument("--video-file-size-in-mb", type=int, default=None)
    args = parser.parse_args()
    convert_dataset(root=args.root, data_file_size_in_mb=args.data_file_size_in_mb, video_file_size_in_mb=args.video_file_size_in_mb)
