#!/usr/bin/env python3
from pathlib import Path
import json
import argparse
import sys
import pandas as pd
import shutil
import tempfile
import numpy as np


def delete_episode(dataset_path, episode_id):
    root = Path(dataset_path)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    ep = int(episode_id)
    episodes_path = (
        root
        / "meta"
        / "episodes"
        / "chunk-000"
        / "episodes.parquet"
    )

    if not episodes_path.exists():
        raise FileNotFoundError(
            f"Missing episodes metadata file: {episodes_path}"
        )

    episodes = pd.read_parquet(episodes_path).sort_values("episode_index").reset_index(drop=True)
    old_indices = [int(value) for value in episodes["episode_index"]]
    if ep not in old_indices:
        raise ValueError(f"Episode {ep} not found in metadata")

    print(f"\nDeleting episode_{ep:06d} and rebuilding retained episodes...")

    data_dir = root / "data" / "chunk-000"
    video_dirs = [path for path in (root / "videos").glob("*") if path.is_dir()]
    retained = [old for old in old_indices if old != ep]
    mapping = {old: new for new, old in enumerate(retained)}

    # Stage all retained files first. This prevents episode_000001 from
    # overwriting episode_000002 while the list is being resequenced.
    with tempfile.TemporaryDirectory(prefix="dataset-rebuild-", dir=root) as staging:
        staging_root = Path(staging)
        staged_data = staging_root / "data" / "chunk-000"
        staged_data.mkdir(parents=True)
        staged_videos = staging_root / "videos"

        global_index = 0
        rebuilt_rows = []
        for old_index in retained:
            old_data = data_dir / f"episode_{old_index:06d}.parquet"
            if not old_data.exists():
                raise FileNotFoundError(f"Missing data parquet for episode {old_index}: {old_data}")

            new_index = mapping[old_index]
            frame_df = pd.read_parquet(old_data)
            frame_count = len(frame_df)
            if frame_count == 0:
                raise ValueError(f"Episode {old_index} has no parquet rows")

            if "episode_index" in frame_df.columns:
                frame_df["episode_index"] = new_index
            if "frame_index" in frame_df.columns:
                frame_df["frame_index"] = range(frame_count)
            if "index" in frame_df.columns:
                frame_df["index"] = range(global_index, global_index + frame_count)
            global_index += frame_count
            frame_df.to_parquet(staged_data / f"episode_{new_index:06d}.parquet", index=False)

            for video_dir in video_dirs:
                old_video = video_dir / "chunk-000" / f"episode_{old_index:06d}.mp4"
                if not old_video.exists():
                    raise FileNotFoundError(f"Missing video for episode {old_index}: {old_video}")
                destination = staged_videos / video_dir.name / "chunk-000"
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_video, destination / f"episode_{new_index:06d}.mp4")

            row = episodes[episodes["episode_index"] == old_index].iloc[0].copy()
            row["episode_index"] = new_index
            row["dataset_from_index"] = global_index - frame_count
            row["dataset_to_index"] = global_index
            row["length"] = frame_count
            if "data_path" in row.index:
                row["data_path"] = f"data/chunk-000/episode_{new_index:06d}.parquet"
            rebuilt_rows.append(row)

        rebuilt = pd.DataFrame(rebuilt_rows, columns=episodes.columns)
        rebuilt.to_parquet(staging_root / "episodes.parquet", index=False)

        for path in data_dir.glob("episode_*.parquet"):
            path.unlink()
        for path in staged_data.glob("episode_*.parquet"):
            shutil.move(str(path), data_dir / path.name)
        for video_dir in video_dirs:
            target_dir = video_dir / "chunk-000"
            for path in target_dir.glob("episode_*.mp4"):
                path.unlink()
            staged_dir = staged_videos / video_dir.name / "chunk-000"
            for path in staged_dir.glob("episode_*.mp4"):
                shutil.move(str(path), target_dir / path.name)
        shutil.move(str(staging_root / "episodes.parquet"), episodes_path)

    # Keep the optional JSONL metadata source in lockstep with episodes.parquet.
    # jsonl can already be out of sync with episodes.parquet (e.g. left stale by an
    # earlier interrupted run of this same script, or never updated by the recorder
    # for episodes added later) -- rows whose episode_index has no entry in `mapping`
    # are dropped rather than crashing here, which would abort *after* data/videos/
    # episodes.parquet were already renumbered and leave the dataset half-migrated
    # (the exact state that made validate_dataset.py flag the whole dataset as broken).
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        rows = [json.loads(line) for line in episodes_jsonl.read_text().splitlines() if line.strip()]
        kept_rows = []
        orphaned = []
        for row in rows:
            old_index = int(row.get("episode_index", -1))
            if old_index == ep:
                continue
            if old_index not in mapping:
                orphaned.append(old_index)
                continue
            row["episode_index"] = mapping[old_index]
            kept_rows.append(row)
        if orphaned:
            print(f"  (dropping {len(orphaned)} episodes.jsonl row(s) with no matching "
                  f"episodes.parquet entry -- metadata sources were already out of sync: "
                  f"{sorted(orphaned)})")
        episodes_jsonl.write_text("".join(json.dumps(row) + "\n" for row in kept_rows), encoding="utf-8")

    # ---------------------------------------------------
    # Rebuild info.json and stats.json
    # ---------------------------------------------------
    info_path = root / "meta" / "info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing info file: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))

    print("\nRebuilding dataset metadata...")

    # The recorder finalizer is intentionally not used here: it can discover
    # stale files during a middle-episode rename. Update authoritative totals
    # directly after the staged rebuild.
    info["total_episodes"] = len(retained)
    info["total_frames"] = int(global_index)
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        stats = {}
        retained_frames = [
            pd.read_parquet(data_dir / f"episode_{index:06d}.parquet")
            for index in range(len(retained))
        ]
        columns = retained_frames[0].columns if retained_frames else []
        for column in columns:
            values = []
            for frame_df in retained_frames:
                if column not in frame_df.columns:
                    continue
                values.extend(frame_df[column].tolist())
            if not values:
                continue
            try:
                array = np.asarray(values, dtype=np.float64)
                if array.ndim == 1:
                    array = array[:, None]
                if array.ndim != 2 or not np.isfinite(array).all():
                    continue
                stats[column] = {
                    "mean": np.mean(array, axis=0).tolist(),
                    "std": np.std(array, axis=0).tolist(),
                    "min": np.min(array, axis=0).tolist(),
                    "max": np.max(array, axis=0).tolist(),
                    "q01": np.percentile(array, 1, axis=0).tolist(),
                    "q99": np.percentile(array, 99, axis=0).tolist(),
                }
            except (TypeError, ValueError):
                continue
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    new_info = info

    print("\nDone")
    print("Total episodes:", new_info["total_episodes"])
    print("Total frames:", new_info["total_frames"])


def main():
    parser = argparse.ArgumentParser(description="Delete a specific episode from the dataset.")

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset root path"
    )

    parser.add_argument(
        "--episode",
        required=True,
        type=int,
        help="Episode number to delete"
    )

    args = parser.parse_args()

    try:
        delete_episode(args.dataset, args.episode)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
