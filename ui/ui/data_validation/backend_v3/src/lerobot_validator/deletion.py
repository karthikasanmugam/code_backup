from __future__ import annotations

"""Safe in-place deletion for LeRobot v3 datasets used by the validation UI."""

import json
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_DELETE_LOCK = threading.Lock()
_VALIDATION_REFERENCE_STATS = "validation_reference_action_percentiles.json"


def _compat_episode_stats_loader(original_loader):
    """Normalize scalar `stats/*/count` values for LeRobot's stats aggregator."""
    import numpy as np

    def load(dataset, episode_index):
        stats = original_loader(dataset, episode_index)
        for key, value in list(stats.items()):
            if key.startswith("stats/") and key.endswith("/count") and np.isscalar(value):
                stats[key] = np.asarray([value])
        return stats

    return load


def _cached_episode_stats_loader(dataset_tools, original_loader):
    """Read each episode-metadata parquet shard once during a delete rebuild."""
    import pandas as pd

    shards: dict[Path, Any] = {}

    def load(dataset, episode_index):
        episode = dataset.meta.episodes[episode_index]
        chunk_index = episode["meta/episodes/chunk_index"]
        file_index = episode["meta/episodes/file_index"]
        path = dataset.root / dataset_tools.DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_index, file_index=file_index
        )
        frame = shards.get(path)
        if frame is None:
            frame = pd.read_parquet(path).set_index("episode_index", drop=False)
            shards[path] = frame
        row = frame.loc[int(episode_index)]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"Episode metadata contains duplicate index {episode_index}")
        return row.to_dict()

    return _compat_episode_stats_loader(load)


def _copy_videos_without_reencoding(src_dataset, dst_meta, episode_mapping):
    """Reuse source video shards and original segment timestamps during deletion."""
    video_metadata: dict[int, dict[str, Any]] = {new_idx: {} for new_idx in episode_mapping.values()}
    copied: set[Path] = set()
    for old_idx, new_idx in episode_mapping.items():
        episode = src_dataset.meta.episodes[old_idx]
        for video_key in src_dataset.meta.video_keys:
            prefix = f"videos/{video_key}/"
            for name in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                key = prefix + name
                if key not in episode:
                    raise ValueError(f"Episode {old_idx} is missing video metadata '{key}'")
                video_metadata[new_idx][key] = episode[key]

            source_relative = Path(src_dataset.meta.get_video_file_path(old_idx, video_key))
            source_path = src_dataset.root / source_relative
            destination_path = dst_meta.root / source_relative
            if source_relative not in copied:
                if not source_path.is_file():
                    raise FileNotFoundError(f"Video shard does not exist: {source_path}")
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_path, destination_path)
                except OSError:
                    shutil.copy2(source_path, destination_path)
                copied.add(source_relative)
    return video_metadata


def _restore_custom_metadata(
    source_info: dict[str, Any],
    source_meta: Path,
    output_root: Path,
) -> None:
    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)

    # These are Daksha/GROOT metadata extensions, not emitted by LeRobot's
    # standard dataset editor. Keep modality ranges and other non-derived info.
    modality = source_meta / "modality.json"
    if modality.is_file():
        shutil.copy2(modality, output_meta / "modality.json")

    output_info_path = output_meta / "info.json"
    output_info = json.loads(output_info_path.read_text(encoding="utf-8"))
    for key, value in source_info.items():
        if key not in output_info and key not in {"total_episodes", "total_frames", "total_tasks", "splits"}:
            output_info[key] = value
    output_info_path.write_text(json.dumps(output_info, indent=2) + "\n", encoding="utf-8")


def _preserve_validation_reference_stats(source_meta: Path, output_root: Path) -> None:
    """Keep the validator's original action-range bounds while pruning episodes.

    LeRobot's aggregate stats.json percentiles are not equivalent to the
    validator's percentiles over frame-level action values. Freeze bounds from
    the validator's own calculation so those limits stay consistent across
    repeated deletions.
    """
    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    reference_path = output_meta / _VALIDATION_REFERENCE_STATS
    source_reference = source_meta / _VALIDATION_REFERENCE_STATS

    if source_reference.is_file():
        try:
            existing = json.loads(source_reference.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {}
        if existing.get("version") == 1:
            shutil.copy2(source_reference, reference_path)
            return
        # Unversioned files were produced by the old implementation, which
        # copied LeRobot aggregate percentiles and caused false rejects.
        logger.warning("Refreshing legacy action percentile reference in %s", source_reference)

    from .dataset import compute_dataset_stats

    data_root = source_meta.parent / "data"
    parquet_files = sorted(data_root.rglob("*.parquet")) if data_root.is_dir() else []
    if not parquet_files:
        logger.warning("No data parquet files found to establish action reference under %s", data_root)
        return

    try:
        stats = compute_dataset_stats(
            parquet_files=parquet_files,
            action_col="action",
            percentiles=(1.0, 99.0),
        )
        low = stats.action_percentiles.get("low")
        high = stats.action_percentiles.get("high")
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Could not compute validator action reference from %s: %s", data_root, exc)
        return

    if low is None or high is None or low.size == 0 or low.shape != high.shape:
        logger.warning("Could not compute valid action percentile bounds from %s", data_root)
        return

    reference_path.write_text(
        json.dumps(
            {"version": 1, "low": low.tolist(), "high": high.tolist()},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _recompute_relative_stats(source_meta: Path, output_root: Path) -> None:
    """Refresh Daksha's horizon-relative action stats from retained v3 rows."""
    import numpy as np
    import pandas as pd

    old_stats_path = source_meta / "relative_stats.json"
    modality_path = source_meta / "modality.json"
    if not old_stats_path.is_file() or not modality_path.is_file():
        return

    old_stats = json.loads(old_stats_path.read_text(encoding="utf-8"))
    modality = json.loads(modality_path.read_text(encoding="utf-8"))
    action_ranges = modality.get("action", {})
    state_ranges = modality.get("state", {})
    if not old_stats or not action_ranges:
        return

    horizon = 16
    for entry in old_stats.values():
        mean = np.asarray(entry.get("mean", [])) if isinstance(entry, dict) else np.asarray([])
        if mean.ndim >= 2 and mean.shape[0] > 0:
            horizon = int(mean.shape[0])
            break

    parquet_files = sorted((output_root / "data").rglob("*.parquet"))
    relative_chunks: dict[str, list[Any]] = {key: [] for key in old_stats if key in action_ranges}
    for parquet_path in parquet_files:
        frame = pd.read_parquet(parquet_path, columns=["episode_index", "observation.state", "action"])
        for _, episode in frame.groupby("episode_index", sort=True):
            states = np.stack([np.asarray(value, dtype=np.float64) for value in episode["observation.state"]])
            actions = np.stack([np.asarray(value, dtype=np.float64) for value in episode["action"]])
            usable = len(episode) - horizon + 1
            if usable <= 0:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(actions, horizon, axis=0)
            windows = np.transpose(windows[:usable], (0, 2, 1))
            for key, action_spec in action_ranges.items():
                if key not in relative_chunks:
                    continue
                action_start, action_end = int(action_spec["start"]), int(action_spec["end"])
                state_spec = state_ranges.get(key, action_spec)
                state_start, state_end = int(state_spec["start"]), int(state_spec["end"])
                action_window = windows[:, :, action_start:action_end]
                state_window = states[:usable, state_start:state_end]
                if action_window.shape[2] != state_window.shape[1]:
                    raise ValueError(f"Relative state/action dimensions differ for modality '{key}'")
                relative_chunks[key].append(action_window - state_window[:, None, :])

    refreshed: dict[str, Any] = {}
    for key, chunks in relative_chunks.items():
        if not chunks:
            # No retained episode is long enough; never leave stale stats for deleted data.
            continue
        values = np.concatenate(chunks, axis=0)
        refreshed[key] = {
            "max": np.max(values, axis=0).tolist(),
            "min": np.min(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
        }
    (output_root / "meta" / "relative_stats.json").write_text(
        json.dumps(refreshed, indent=2) + "\n", encoding="utf-8"
    )


def delete_episodes_in_place(dataset_root: str | Path, episode_indices: list[int]) -> dict[str, Any]:
    """Delete selected v3 episodes and rebuild the dataset atomically.

    Parquet, episode metadata, video segments and standard stats are regenerated
    in a sibling staging directory. The original is restored if the replacement
    or cleanup fails. The dataset root itself is never used as a staging target.
    """
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not root.is_dir() or not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root: {root}")

    source_info = json.loads(info_path.read_text(encoding="utf-8"))
    if str(source_info.get("codebase_version", "")) != "v3.0":
        raise ValueError("Episode deletion from the validation UI currently requires a LeRobot v3.0 dataset")

    requested = sorted({int(index) for index in episode_indices})
    if not requested:
        raise ValueError("Select at least one episode to delete")
    if requested[0] < 0:
        raise ValueError("Episode indices must be non-negative")

    try:
        from lerobot.datasets import dataset_tools
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot dataset editing tools are unavailable in this Python environment") from exc

    with _DELETE_LOCK:
        repo_id = f"local/{root.name}"
        dataset = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)
        total_episodes = int(dataset.meta.total_episodes)
        invalid = sorted(set(requested) - set(range(total_episodes)))
        if invalid:
            raise ValueError(f"Episode index/indices not found: {invalid}")
        if len(requested) >= total_episodes:
            raise ValueError("Cannot delete every episode; keep at least one episode in the dataset")

        stage_dir = Path(tempfile.mkdtemp(prefix=f".{root.name}.delete-", dir=root.parent))
        output_root = stage_dir / "rebuilt"
        backup_root = stage_dir / "original"
        try:
            original_loader = dataset_tools._load_episode_with_stats
            original_video_copy = dataset_tools._copy_and_reindex_videos
            dataset_tools._load_episode_with_stats = _cached_episode_stats_loader(
                dataset_tools, original_loader
            )
            dataset_tools._copy_and_reindex_videos = _copy_videos_without_reencoding
            try:
                logger.info(
                    "Deleting episodes %s from v3 dataset %s (preserving video shards)",
                    requested,
                    root,
                )
                rebuilt = dataset_tools.delete_episodes(
                    dataset,
                    episode_indices=requested,
                    output_dir=output_root,
                    repo_id=repo_id,
                )
            finally:
                dataset_tools._load_episode_with_stats = original_loader
                dataset_tools._copy_and_reindex_videos = original_video_copy

            if int(rebuilt.meta.total_episodes) != total_episodes - len(requested):
                shutil.rmtree(stage_dir, ignore_errors=True)
                raise RuntimeError("LeRobot rebuild returned an unexpected episode count")

            _restore_custom_metadata(source_info, root / "meta", output_root)
            _preserve_validation_reference_stats(root / "meta", output_root)
            _recompute_relative_stats(root / "meta", output_root)

            # Validate the rebuilt dataset before touching the original directory.
            checked = LeRobotDataset(repo_id=repo_id, root=output_root, download_videos=False)
            if int(checked.meta.total_episodes) != total_episodes - len(requested):
                shutil.rmtree(stage_dir, ignore_errors=True)
                raise RuntimeError("Rebuilt dataset metadata failed validation")

        except BaseException:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise

        try:
            os.replace(root, backup_root)
            try:
                os.replace(output_root, root)
            except Exception:
                os.replace(backup_root, root)
                raise
            try:
                shutil.rmtree(backup_root)
            except Exception as exc:
                failed_output = stage_dir / "failed-rebuild"
                os.replace(root, failed_output)
                os.replace(backup_root, root)
                raise RuntimeError(f"Could not securely remove the old dataset contents; deletion was rolled back: {exc}") from exc
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    logger.info("Deleted episodes %s; dataset now has %s episodes", requested, total_episodes - len(requested))
    return {
        "deleted": requested,
        "dataset_root": str(root),
        "remaining_episodes": total_episodes - len(requested),
    }
