"""
LeRobot v3.0 dataset adapter for the VLA evaluation framework.

Supports the final v3.0 layout produced by this project's converter:

    root/
      meta/info.json
      meta/tasks.parquet
      meta/episodes/chunk-XXX/file-XXX.parquet
      meta/stats.json
      data/chunk-XXX/file-XXX.parquet
      videos/<camera_key>/chunk-XXX/file-XXX.mp4

The v3.0 format consolidates multiple episodes into shared parquet/video
files. Episode metadata tells us which data shard + row range belongs to
an episode, and which video shard + timestamp range belongs to each camera.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vla_eval.core.base_dataset import BaseDataset
from vla_eval.core.types import (
    EpisodeInfo,
    GroundTruth,
    Observation,
    Sample,
)

_CHUNK_DIR_RE = re.compile(r"chunk-(\d+)$")
_FILE_RE = re.compile(r"file-(\d+)\.parquet$")


class _LRUCache:
    """Small bounded cache for parquet DataFrames / video captures."""

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("Cache size must be >= 1")
        self.max_size = max_size
        self._store: "OrderedDict[object, object]" = OrderedDict()

    def get(self, key):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key, value):
        if key in self._store:
            old = self._store.pop(key)
            close = getattr(old, "release", None)
            if callable(close):
                close()
        self._store[key] = value
        while len(self._store) > self.max_size:
            _, evicted = self._store.popitem(last=False)
            close = getattr(evicted, "release", None)
            if callable(close):
                close()

    def clear(self):
        for value in self._store.values():
            close = getattr(value, "release", None)
            if callable(close):
                close()
        self._store.clear()


@dataclass(frozen=True)
class _EpisodeRecord:
    episode_index: int
    length: int
    task: str
    data_path: Path
    data_start: int
    data_end: int
    chunk_index: int
    video_info: Dict[str, Dict[str, object]]


class LeRobotCustomDataset(BaseDataset):
    """Read the final LeRobot v3.0 dataset produced by the converter."""

    def __init__(
        self,
        root: str,
        cache_data_files: int = 2,
        cache_video_captures: int = 8,
    ):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.info = self._load_info()
        self.fps = float(self.info.get("fps", 30))
        if self.fps <= 0:
            raise ValueError(f"Invalid dataset fps: {self.fps}")

        self.camera_keys = self._discover_camera_keys()

        self._task_lookup = self._load_task_lookup()

        self._data_path_template = self.info.get(
            "data_path",
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        )
        self._video_path_template = self.info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )

        self._episodes: Dict[int, _EpisodeRecord] = (
            self._load_episode_records()
        )
        if not self._episodes:
            raise FileNotFoundError(
                f"No v3.0 episode metadata found under "
                f"{self.root / 'meta' / 'episodes'}"
            )

        self._episode_infos: List[EpisodeInfo] = [
            EpisodeInfo(
                episode_index=ep.episode_index,
                length=ep.length,
                task=ep.task,
                chunk_index=ep.chunk_index,
            )
            for ep in self._episodes.values()
        ]

        self._cum_lengths: List[int] = []
        offset = 0
        for ep in self._episode_infos:
            self._cum_lengths.append(offset)
            offset += ep.length
        self._total_frames = offset

        declared_total_frames = self.info.get("total_frames")
        if declared_total_frames is not None and int(declared_total_frames) != self._total_frames:
            raise ValueError(
                "Episode metadata total does not match info.json total_frames: "
                f"metadata={self._total_frames}, info={declared_total_frames}"
            )

        declared_total_episodes = self.info.get("total_episodes")
        if declared_total_episodes is not None and int(declared_total_episodes) != len(self._episode_infos):
            raise ValueError(
                "Episode metadata count does not match info.json total_episodes: "
                f"metadata={len(self._episode_infos)}, info={declared_total_episodes}"
            )

        self._episode_df_cache = _LRUCache(cache_data_files)
        self._video_cap_cache = _LRUCache(cache_video_captures)
        self._video_next_rows: Dict[Tuple[int, str], int] = {}

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _load_info(self) -> dict:
        path = self.root / "meta" / "info.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing info.json: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _discover_camera_keys(self) -> List[str]:
        """Discover video features from info.json, not legacy camera_keys."""
        features = self.info.get("features", {})
        cameras = [
            key
            for key, spec in features.items()
            if isinstance(spec, dict) and spec.get("dtype") in {"video", "image"}
        ]

        if not cameras:
            raise ValueError(
                "No camera/video features found in meta/info.json"
            )

        return sorted(cameras)

    def _load_task_lookup(self) -> Dict[int, str]:
        path = self.root / "meta" / "tasks.parquet"
        if not path.exists():
            return {}

        df = pd.read_parquet(path)
        if df.empty:
            return {}

        idx_col = next(
            (c for c in df.columns if str(c) == "task_index"),
            None,
        )
        task_col = next(
            (c for c in df.columns if str(c) == "task"),
            None,
        )

        if idx_col is None or task_col is None:
            return {}

        return {
            int(row[idx_col]): str(row[task_col])
            for _, row in df.iterrows()
        }

    def _task_from_row(self, row, default: str = "") -> str:
        if "task" in row.index and pd.notna(row["task"]):
            return str(row["task"])

        if "task_index" in row.index and pd.notna(row["task_index"]):
            task = self._task_lookup.get(int(row["task_index"]))
            if task:
                return task

        if "tasks" in row.index and row["tasks"] is not None:
            tasks = row["tasks"]
            if isinstance(tasks, (list, tuple, np.ndarray)):
                if len(tasks) > 0 and pd.notna(tasks[0]):
                    return str(tasks[0])
            elif pd.notna(tasks):
                return str(tasks)

        return default

    def _load_episode_records(self) -> Dict[int, _EpisodeRecord]:
        metadata_files = sorted(
            (self.root / "meta" / "episodes").glob(
                "chunk-*/file-*.parquet"
            )
        )

        if not metadata_files:
            raise FileNotFoundError(
                "No v3.0 episode metadata files found. Expected "
                "meta/episodes/chunk-*/file-*.parquet"
            )

        records: Dict[int, _EpisodeRecord] = {}

        for metadata_path in metadata_files:
            df = pd.read_parquet(metadata_path)
            for _, row in df.iterrows():
                episode_index = int(row["episode_index"])
                if episode_index in records:
                    raise ValueError(
                        f"Duplicate episode_index in metadata: {episode_index}"
                    )

                length = int(row["length"])
                data_start = int(row["dataset_from_index"])
                data_end = int(row["dataset_to_index"])

                if data_end - data_start != length:
                    raise ValueError(
                        f"Episode {episode_index} has inconsistent range: "
                        f"from={data_start}, to={data_end}, length={length}"
                    )

                data_chunk = int(row["data/chunk_index"])
                data_file = int(row["data/file_index"])
                data_path = self.root / self._data_path_template.format(
                    chunk_index=data_chunk,
                    file_index=data_file,
                    episode_chunk=data_chunk,
                    episode_index=episode_index,
                )

                if not data_path.exists():
                    raise FileNotFoundError(
                        f"Data shard for episode {episode_index} does not exist: "
                        f"{data_path}"
                    )

                task = self._task_from_row(row)

                video_info: Dict[str, Dict[str, object]] = {}

                for camera_key in self.camera_keys:
                    prefix = f"videos/{camera_key}/"
                    chunk_col = prefix + "chunk_index"
                    file_col = prefix + "file_index"
                    from_col = prefix + "from_timestamp"
                    to_col = prefix + "to_timestamp"

                    if chunk_col not in row.index or file_col not in row.index:
                        raise ValueError(
                            f"Episode {episode_index} is missing video metadata "
                            f"for camera '{camera_key}'"
                        )

                    video_chunk = int(row[chunk_col])
                    video_file = int(row[file_col])
                    from_timestamp = float(row[from_col]) if from_col in row.index else 0.0
                    to_timestamp = float(row[to_col]) if to_col in row.index else None

                    video_path = self.root / self._video_path_template.format(
                        video_key=camera_key,
                        chunk_index=video_chunk,
                        file_index=video_file,
                        episode_chunk=video_chunk,
                        episode_index=episode_index,
                    )

                    if not video_path.exists():
                        raise FileNotFoundError(
                            f"Video shard for episode {episode_index}, "
                            f"camera '{camera_key}' does not exist: {video_path}"
                        )

                    video_info[camera_key] = {
                        "path": video_path,
                        "from_timestamp": from_timestamp,
                        "to_timestamp": to_timestamp,
                    }

                records[episode_index] = _EpisodeRecord(
                    episode_index=episode_index,
                    length=length,
                    task=task,
                    data_path=data_path,
                    data_start=data_start,
                    data_end=data_end,
                    chunk_index=data_chunk,
                    video_info=video_info,
                )

        # Validate the global data ranges are continuous in episode order.
        ordered = sorted(records.values(), key=lambda e: e.episode_index)
        expected_start = 0
        for ep in ordered:
            if ep.data_start != expected_start:
                raise ValueError(
                    "Non-contiguous dataset ranges in episode metadata at "
                    f"episode {ep.episode_index}: expected start {expected_start}, "
                    f"found {ep.data_start}"
                )
            expected_start = ep.data_end

        return dict(sorted(records.items()))

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_frames

    @property
    def episodes(self) -> List[EpisodeInfo]:
        return self._episode_infos

    def get_episode_frames(self, episode_index: int) -> List[int]:
        if episode_index not in self._episodes:
            raise KeyError(f"Unknown episode_index: {episode_index}")

        target_pos = next(
            i
            for i, ep in enumerate(self._episode_infos)
            if ep.episode_index == episode_index
        )
        start = self._cum_lengths[target_pos]
        length = self._episode_infos[target_pos].length
        return list(range(start, start + length))

    def __getitem__(self, index: int) -> Sample:
        if index < 0:
            index += self._total_frames
        if index < 0 or index >= self._total_frames:
            raise IndexError(
                f"Index {index} out of range (0, {self._total_frames - 1})"
            )

        episode_index, local_row = self._locate(index)
        episode = self._episodes[episode_index]

        row = self._get_data_row(episode, local_row)

        images = {
            camera_key: self._read_frame(
                episode,
                camera_key,
                local_row,
            )
            for camera_key in self.camera_keys
        }

        state = np.asarray(
            row["observation.state"],
            dtype=np.float32,
        )

        leader_state = None
        if "observation.leader_state" in row.index:
            leader_state = np.asarray(
                row["observation.leader_state"],
                dtype=np.float32,
            )

        task = self._task_from_row(row, episode.task)

        observation = Observation(
            images=images,
            state=state,
            instruction=task,
            timestamp=float(row["timestamp"]),
            frame_index=int(row["frame_index"]),
            episode_index=int(row["episode_index"]),
            extra=(
                {"leader_state": leader_state}
                if leader_state is not None
                else {}
            ),
        )

        ground_truth = GroundTruth(
            action=np.asarray(
                row["action"],
                dtype=np.float32,
            ),
            task=task,
            episode_index=int(row["episode_index"]),
            frame_index=int(row["frame_index"]),
            index=int(row["index"]) if "index" in row.index else index,
        )

        return Sample(
            observation=observation,
            ground_truth=ground_truth,
        )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _locate(
        self,
        global_index: int,
    ) -> Tuple[int, int]:
        """
        Convert a dataset-global frame index into:

            episode_index
            local frame index
        """

        if (
            global_index < 0
            or global_index >= self._total_frames
        ):
            raise IndexError(
                f"Global index {global_index} is out of range "
                f"[0, {self._total_frames})"
            )

        episode_position = (
            bisect_right(
                self._cum_lengths,
                global_index,
            )
            - 1
        )

        episode = self._episode_infos[
            episode_position
        ]

        episode_start = (
            self._cum_lengths[
                episode_position
            ]
        )

        local_row = (
            global_index
            - episode_start
        )

        if local_row >= episode.length:
            raise RuntimeError(
                "Global-to-episode mapping is inconsistent: "
                f"episode={episode.episode_index}, "
                f"local_row={local_row}, "
                f"length={episode.length}"
            )

        return (
            episode.episode_index,
            local_row,
        )
    def _get_episode_df(
        self,
        episode: _EpisodeRecord,
    ) -> pd.DataFrame:
        """
        Load the data rows belonging to exactly one episode.

        For LeRobot v3.0 a data shard can contain many episodes, so
        episode_index is the authoritative selector. The global index
        range from episode metadata is then used as an additional
        consistency check.
        """

        key = str(episode.data_path)

        cached = self._episode_df_cache.get(key)

        # IMPORTANT:
        # The cache key must include the episode because multiple
        # episodes can live in the same v3.0 parquet shard.
        cache_key = (
            str(episode.data_path),
            episode.episode_index,
        )

        cached = self._episode_df_cache.get(cache_key)

        if cached is not None:
            return cached

        df = pd.read_parquet(
            episode.data_path
        )

        if "episode_index" not in df.columns:
            raise ValueError(
                "v3.0 data file is missing 'episode_index': "
                f"{episode.data_path}"
            )

        # ------------------------------------------------------------------
        # Authoritative selection:
        #
        # Multiple episodes can share the same parquet file.
        # Select rows by episode_index.
        # ------------------------------------------------------------------

        episode_mask = (
            df["episode_index"].astype(int)
            == episode.episode_index
        )

        episode_df = (
            df.loc[episode_mask]
            .copy()
            .reset_index(drop=True)
        )

        # ------------------------------------------------------------------
        # Validate frame count.
        # ------------------------------------------------------------------

        if len(episode_df) != episode.length:
            raise ValueError(
                f"Episode {episode.episode_index} expected "
                f"{episode.length} rows but found "
                f"{len(episode_df)} in {episode.data_path}"
            )

        # ------------------------------------------------------------------
        # Validate global index range when available.
        # ------------------------------------------------------------------

        if "index" in episode_df.columns:

            actual_indices = (
                episode_df["index"]
                .astype(int)
                .to_numpy()
            )

            expected_indices = np.arange(
                episode.data_start,
                episode.data_end,
            )

            if not np.array_equal(
                actual_indices,
                expected_indices,
            ):
                raise ValueError(
                    "Global index range does not match episode metadata "
                    f"for episode {episode.episode_index}.\n"
                    f"Expected: {episode.data_start}.."
                    f"{episode.data_end - 1}\n"
                    f"Actual:   {actual_indices[0]}.."
                    f"{actual_indices[-1]}"
                )

        # ------------------------------------------------------------------
        # Validate episode_index after filtering.
        # ------------------------------------------------------------------

        actual_episode_indices = (
            episode_df["episode_index"]
            .astype(int)
            .unique()
            .tolist()
        )

        if actual_episode_indices != [
            episode.episode_index
        ]:
            raise ValueError(
                f"Episode filtering failed for "
                f"episode {episode.episode_index}: "
                f"found episode indices "
                f"{actual_episode_indices}"
            )

        self._episode_df_cache.put(
            cache_key,
            episode_df,
        )

        return episode_df
    def _get_data_row(
        self,
        episode: _EpisodeRecord,
        local_row: int,
    ) -> pd.Series:
        if local_row < 0 or local_row >= episode.length:
            raise IndexError(
                f"Local row {local_row} outside episode {episode.episode_index}"
            )

        df = self._get_episode_df(episode)
        return df.iloc[local_row]

    # ------------------------------------------------------------------
    # Video access
    # ------------------------------------------------------------------

    def _read_frame(
        self,
        episode: _EpisodeRecord,
        camera_key: str,
        local_row: int,
    ) -> np.ndarray:

        video = episode.video_info[camera_key]

        cap = self._get_video_capture(
            episode,
            camera_key,
        )

        from_timestamp = float(
            video["from_timestamp"]
        )

        target_time = (
            from_timestamp
            + (local_row / self.fps)
        )

        position_key = (episode.episode_index, camera_key)
        next_row = self._video_next_rows.get(position_key)
        if next_row != local_row:
            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                target_time * 1000.0,
            )

        ok, frame_bgr = cap.read()

        if not ok:
            raise RuntimeError(
                "Failed to read v3.0 video frame: "
                f"episode={episode.episode_index}, "
                f"camera={camera_key}, "
                f"local_row={local_row}, "
                f"target_time={target_time:.6f}, "
                f"video={video['path']}"
            )

        self._video_next_rows[position_key] = local_row + 1

        return cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

    def _get_video_capture(
        self,
        episode: _EpisodeRecord,
        camera_key: str,
    ) -> cv2.VideoCapture:
        cache_key = (
            str(episode.video_info[camera_key]["path"]),
        )

        cached = self._video_cap_cache.get(cache_key)
        if cached is not None:
            return cached

        path = Path(
            episode.video_info[camera_key]["path"]
        )

        cap = cv2.VideoCapture(
            str(path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video shard: {path}"
            )

        self._video_cap_cache.put(
            cache_key,
            cap,
        )

        return cap

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release cached files and video handles."""
        self._episode_df_cache.clear()
        self._video_cap_cache.clear()
        self._video_next_rows.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
