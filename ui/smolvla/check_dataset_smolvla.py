#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REQUIRED_CAMERAS = [
    "observation.images.world",
    "observation.images.wrist_right",
    "observation.images.wrist_left"
]
REQUIRED_META_FILES = [
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_tasks(dataset_root: Path) -> list[str]:
    tasks_df = pd.read_parquet(dataset_root / "meta/tasks.parquet")
    return [str(task) for task in tasks_df.index]


def repair_missing_quantiles(dataset_root: Path, info: dict, stats: dict) -> dict:
    keys = [key for key in ("action", "observation.state")
            if any(q not in stats.get(key, {}) for q in ("q01", "q99"))]
    if not keys:
        return stats
    paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not paths:
        raise ValueError("No parquet data found for quantile computation")
    for key in keys:
        values = np.concatenate([
            np.asarray(pq.ParquetFile(path).read(columns=[key])[key].to_pylist(), dtype=np.float64)
            for path in paths
        ])
        if values.shape != (info["total_frames"], 16) or not np.isfinite(values).all():
            raise ValueError(f"Cannot compute {key} quantiles: invalid shape or nonfinite values")
        for name, fraction in (("q01", 0.01), ("q99", 0.99)):
            stats.setdefault(key, {}).setdefault(name, np.quantile(values, fraction, axis=0).tolist())
    stats_path = dataset_root / "meta/stats.json"
    with tempfile.NamedTemporaryFile(dir=stats_path.parent, prefix="stats.before-quantiles-", suffix=".json", delete=False) as backup:
        backup_path = Path(backup.name)
    shutil.copy2(stats_path, backup_path)
    with tempfile.NamedTemporaryFile(mode="w", dir=stats_path.parent, suffix=".tmp", delete=False) as output:
        json.dump(stats, output, indent=2, allow_nan=False)
        output.write("\n")
    Path(output.name).replace(stats_path)
    print(f"Added missing quantiles for {', '.join(keys)}. Backup: {backup_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Check smolvla dataset compatibility.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-repo-id", default="local/task1_full")
    parser.add_argument("--repair-missing-quantiles", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    problems: list[str] = []

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    for rel_path in REQUIRED_META_FILES:
        if not (dataset_root / rel_path).exists():
            problems.append(f"Missing required file: {rel_path}")
    if not list((dataset_root / "meta/episodes").glob("*/*.parquet")):
        problems.append("Missing required file: meta/episodes/chunk-*/file-*.parquet")

    info = load_json(dataset_root / "meta/info.json")
    stats = load_json(dataset_root / "meta/stats.json")
    if info.get("codebase_version") != "v3.0":
        problems.append("Dataset must be converted to LeRobot v3.0 before training")
    if args.repair_missing_quantiles and not problems:
        stats = repair_missing_quantiles(dataset_root, info, stats)
    task_lines = load_tasks(dataset_root)

    features = info.get("features", {})
    # A real LeRobot v3.0 info.json has no "camera_keys" field -- DatasetInfo doesn't
    # define one, so lerobot.scripts.convert_dataset_v21_to_v30 (and this repo's own
    # convert_daksha_dataset_to_v30.py) silently drops it on conversion. Derive the
    # camera list from `features` (dtype == "video") the same way
    # LeRobotDatasetMetadata.video_keys does, falling back to an explicit "camera_keys"
    # list for datasets that still carry the older, non-standard field.
    camera_keys = info.get("camera_keys") or [
        key for key, ft in features.items() if ft.get("dtype") == "video"
    ]

    for camera_key in REQUIRED_CAMERAS:
        if camera_key not in features:
            problems.append(f"Missing camera feature in info.json: {camera_key}")
        if camera_key not in camera_keys:
            problems.append(f"Missing camera key in info.json camera_keys: {camera_key}")

    action_feature = features.get("action", {})
    state_feature = features.get("observation.state", {})
    if action_feature.get("shape") != [16]:
        problems.append(f"Expected action shape [16], got {action_feature.get('shape')}")
    if state_feature.get("shape") != [16]:
        problems.append(
            f"Expected observation.state shape [16], got {state_feature.get('shape')}"
        )

    for stats_key in ["action", "observation.state"]:
        stats_entry = stats.get(stats_key, {})
        for quantile_key in ["q01", "q99"]:
            if quantile_key not in stats_entry:
                problems.append(f"Missing {stats_key}.{quantile_key} in meta/stats.json")

    if not task_lines:
        problems.append("meta/tasks.parquet is empty")

    print("=== smolvla Dataset Check ===")
    print(f"dataset_root: {dataset_root}")
    print(f"dataset_repo_id: {args.dataset_repo_id}")
    print(f"robot_type: {info.get('robot_type')}")
    print(f"fps: {info.get('fps')}")
    print(f"total_episodes: {info.get('total_episodes')}")
    print(f"total_frames: {info.get('total_frames')}")
    print(f"camera_keys: {camera_keys}")
    print(f"task_examples: {task_lines[:3]}")
    print(f"action_shape: {action_feature.get('shape')}")
    print(f"state_shape: {state_feature.get('shape')}")
    print("Quantile statistics available; normalization is selected by the policy configuration.")

    if problems:
        print("")
        print("Dataset check failed:")
        for item in problems:
            print(f"- {item}")
        sys.exit(1)

    print("")
    print("Dataset check passed.")


if __name__ == "__main__":
    main()
