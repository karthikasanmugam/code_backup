#!/usr/bin/env python3
"""
Stage 1 of the Daksha dataset conversion pipeline: raw-metadata normalizer.

This is a self-contained (no gr00t import required) pre-processor for raw Daksha dumps that
don't yet have `meta/tasks.jsonl` / `meta/episodes.jsonl` (e.g. a bare per-episode parquet dump
with only `meta/info.json` and a `task.json`/`tasks.parquet` task table). It produces the
v2.1-ish layout that `convert_daksha_dataset_to_v30.py` (stage 2) expects as input:

  - meta/modality.json       (copied from a template, if one is configured -- optional)
  - meta/tasks.jsonl
  - meta/episodes.jsonl
  - meta/info.json           (chunks_size / totals refreshed)
  - meta/stats.json          (per-feature mean/std/min/max/q01/q99; skippable)
  - meta/relative_stats.json (per-horizon-step stats for relative action keys; skippable,
                               requires a modality.json template)

Datasets recorded directly by this repo's Daksha recorder already ship with
`meta/tasks.jsonl` / `meta/episodes.jsonl`, so for those this stage is a no-op and the
pipeline goes straight to stage 2. See `examples/daksha/ui/app.py::ensure_dataset_v30` for
the orchestration that chains this script with `convert_daksha_dataset_to_v30.py`.

The stats this script computes (when not skipped) match what gr00t.data.stats would emit:

  * stats.json covers every feature in info.json whose dtype contains "float", stacked
    across all episodes and cast to float32.
  * relative_stats.json covers the action keys declared RELATIVE (left_arm, right_arm by
    default). For each frame i, the relative chunk is action[i : i + horizon] - state[i].
    Each entry has shape (horizon, joint_dim).

Example:
  python examples/daksha/convert_daksha_dataset_metadata.py \
    --dataset-root /home/s1/Desktop/datasets/some_raw_dataset
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
# Optional: point this at a gr00t-style modality.json if you have one. Not required for
# LeRobot ACT/Diffusion training (which only needs stage 2's v3.0 output); only needed if you
# also want relative_stats.json / modality.json for external GR00T-style consumption.
DEFAULT_MODALITY_TEMPLATE = REPO_ROOT / "examples" / "daksha" / "modality.json"
DEFAULT_CHUNKS_SIZE = 1024

DEFAULT_ACTION_HORIZON = 16
DEFAULT_RELATIVE_KEYS = ("left_arm", "right_arm")

STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")
STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"


# --------------------------------------------------------------------------------------
# Dataset layout
# --------------------------------------------------------------------------------------
def resolve_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if (root / "meta" / "info.json").exists():
        return root

    nested_roots = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "meta" / "info.json").exists()
    )
    if len(nested_roots) == 1:
        return nested_roots[0]
    if len(nested_roots) > 1:
        raise FileNotFoundError(
            f"Multiple LeRobot dataset roots found under {root}: "
            f"{', '.join(str(p) for p in nested_roots)}"
        )
    raise FileNotFoundError(
        f"Could not find meta/info.json under {root}. "
        "Pass the exact dataset root that contains the dataset."
    )


def episode_parquet_paths(dataset_root: Path) -> list[Path]:
    paths = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No episode parquet files found under {dataset_root / 'data'}. "
            "Expected files like data/chunk-000/episode_000000.parquet."
        )
    return paths


def parse_episode_index(path: Path) -> int:
    stem = path.stem
    if stem.startswith("episode_"):
        return int(stem.split("_", 1)[1])
    raise ValueError(f"Could not parse episode index from filename: {path.name}")


# --------------------------------------------------------------------------------------
# tasks.jsonl / episodes.jsonl / info.json / modality.json
# --------------------------------------------------------------------------------------
def read_tasks(meta_dir: Path) -> list[dict[str, object]]:
    task_json = meta_dir / "task.json"
    tasks_parquet = meta_dir / "tasks.parquet"

    if task_json.exists():
        raw_tasks = json.loads(task_json.read_text())
    elif tasks_parquet.exists():
        raw_tasks = pd.read_parquet(tasks_parquet).to_dict(orient="records")
    else:
        raise FileNotFoundError(
            f"Could not find task.json or tasks.parquet in {meta_dir}. "
            "One of those files is needed to generate tasks.jsonl."
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
        tasks.append({"task_index": task_index, "task": task_text, "prompt": prompt})

    tasks.sort(key=lambda item: int(item["task_index"]))
    return tasks


def write_tasks_jsonl(meta_dir: Path, tasks: list[dict[str, object]]) -> Path:
    path = meta_dir / "tasks.jsonl"
    with open(path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    return path


def build_episodes(episode_paths: list[Path]) -> list[dict[str, object]]:
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
        if "task_index" in df.columns:
            episode["task_index"] = int(df["task_index"].iloc[0])
        if "task" in df.columns:
            episode["task"] = str(df["task"].iloc[0])
        if "prompt" in df.columns:
            episode["prompt"] = str(df["prompt"].iloc[0])
        if "timestamp" in df.columns:
            episode["timestamp_start"] = float(df["timestamp"].iloc[0])
            episode["timestamp_end"] = float(df["timestamp"].iloc[-1])

        episodes.append(episode)

    episodes.sort(key=lambda item: int(item["episode_index"]))
    return episodes


def write_episodes_jsonl(meta_dir: Path, episodes: list[dict[str, object]]) -> Path:
    path = meta_dir / "episodes.jsonl"
    with open(path, "w") as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
    return path


def update_info_json(
    meta_dir: Path,
    chunks_size: int,
    tasks: list[dict[str, object]],
    episodes: list[dict[str, object]],
) -> tuple[Path, dict]:
    path = meta_dir / "info.json"
    with open(path, "r") as f:
        info = json.load(f)

    info["chunks_size"] = int(info.get("chunks_size", chunks_size))
    info["total_episodes"] = len(episodes)
    info["total_frames"] = int(sum(int(ep["length"]) for ep in episodes))
    info["total_tasks"] = len(tasks)

    with open(path, "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return path, info


def copy_modality_template(meta_dir: Path, template_path: Path) -> Path | None:
    """Copy the modality.json template if one is configured. Best-effort: this file is
    only needed for external GR00T-style consumption, not for LeRobot ACT/Diffusion
    training, so a missing template is skipped rather than treated as an error.
    """
    if not template_path.exists():
        return None
    target_path = meta_dir / "modality.json"
    if template_path.resolve() != target_path.resolve():
        shutil.copy2(template_path, target_path)
    return target_path


# --------------------------------------------------------------------------------------
# stats.json  (replaces gr00t.data.stats.generate_stats)
# --------------------------------------------------------------------------------------
def float_feature_names(info: dict) -> list[str]:
    """Every feature whose dtype contains 'float', matching gr00t's selection rule."""
    features = info.get("features", {})
    return [name for name, spec in features.items() if "float" in str(spec.get("dtype", ""))]


def column_to_matrix(column: pd.Series) -> np.ndarray:
    """Stack a parquet column into a (n_rows, dim) float32 matrix.

    Vector columns arrive as object arrays of per-row numpy arrays; scalar columns such as
    `timestamp` arrive as a plain numeric dtype and become (n_rows, 1).
    """
    values = column.to_numpy()
    if values.dtype == object:
        return np.vstack([np.asarray(v, dtype=np.float32) for v in values])
    return values.astype(np.float32).reshape(len(values), -1)


def compute_stats(episode_paths: list[Path], features: list[str]) -> dict[str, dict[str, list]]:
    per_feature: dict[str, list[np.ndarray]] = {name: [] for name in features}

    for i, parquet_path in enumerate(episode_paths, start=1):
        df = pd.read_parquet(parquet_path, columns=features)
        for name in features:
            per_feature[name].append(column_to_matrix(df[name]))
        if i % 100 == 0 or i == len(episode_paths):
            print(f"  stats: read {i}/{len(episode_paths)} episodes", flush=True)

    stats: dict[str, dict[str, list]] = {}
    for name in features:
        data = np.concatenate(per_feature[name], axis=0)
        per_feature[name] = []  # free as we go
        stats[name] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
        print(f"  stats: computed {name} (dim={data.shape[1]})", flush=True)
    return stats


def stats_are_valid(stats_path: Path, features: list[str]) -> bool:
    if not stats_path.exists():
        return False
    try:
        with open(stats_path, "r") as f:
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
    dataset_root: Path, info: dict, episode_paths: list[Path], force: bool
) -> Path:
    stats_path = dataset_root / "meta" / "stats.json"
    features = float_feature_names(info)
    if not features:
        raise ValueError("No float features found in meta/info.json; cannot compute stats.")

    if not force and stats_are_valid(stats_path, features):
        print(f"  stats.json already valid for {len(features)} features, skipping.")
        return stats_path

    stats = compute_stats(episode_paths, features)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    return stats_path


# --------------------------------------------------------------------------------------
# relative_stats.json  (replaces gr00t.data.stats.generate_rel_stats)
# --------------------------------------------------------------------------------------
def load_modality_ranges(modality_path: Path) -> dict[str, dict[str, tuple[int, int]]]:
    with open(modality_path, "r") as f:
        modality = json.load(f)
    ranges: dict[str, dict[str, tuple[int, int]]] = {}
    for group in ("state", "action"):
        ranges[group] = {
            key: (int(spec["start"]), int(spec["end"]))
            for key, spec in modality.get(group, {}).items()
        }
    return ranges


def relative_chunks_for_key(
    episode_paths: list[Path],
    state_slice: tuple[int, int],
    action_slice: tuple[int, int],
    horizon: int,
) -> np.ndarray:
    """Stack every relative action chunk in the dataset into (n_chunks, horizon, dim).

    For frame i the chunk is action[i : i+horizon] - state[i], i.e. the joint-space
    relative chunking against the last observed state.
    """
    chunks: list[np.ndarray] = []
    total_skipped = 0

    for i, parquet_path in enumerate(episode_paths, start=1):
        df = pd.read_parquet(parquet_path, columns=[STATE_COLUMN, ACTION_COLUMN])
        state = column_to_matrix(df[STATE_COLUMN])[:, state_slice[0] : state_slice[1]]
        action = column_to_matrix(df[ACTION_COLUMN])[:, action_slice[0] : action_slice[1]]

        usable = len(df) - (horizon - 1)
        if usable <= 0:
            total_skipped += 1
            continue

        # Sliding window of shape (usable, horizon, dim) without copying per frame.
        windows = np.lib.stride_tricks.sliding_window_view(action, horizon, axis=0)
        windows = np.transpose(windows[:usable], (0, 2, 1))
        chunks.append((windows - state[:usable, None, :]).astype(np.float32))

        if i % 100 == 0 or i == len(episode_paths):
            print(f"  relative: read {i}/{len(episode_paths)} episodes", flush=True)

    if not chunks:
        raise ValueError(
            f"No episode is long enough for an action horizon of {horizon} frames."
        )
    if total_skipped:
        print(f"  relative: skipped {total_skipped} episodes shorter than {horizon} frames")
    return np.concatenate(chunks, axis=0)


def generate_relative_stats(
    dataset_root: Path,
    episode_paths: list[Path],
    modality_path: Path,
    relative_keys: list[str],
    horizon: int,
    force: bool,
) -> Path:
    stats_path = dataset_root / "meta" / "relative_stats.json"
    if stats_path.exists() and not force:
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}

    ranges = load_modality_ranges(modality_path)
    for key in sorted(relative_keys):
        if key in stats and not force:
            print(f"  relative_stats.json already has '{key}', skipping.")
            continue
        if key not in ranges["action"]:
            raise KeyError(
                f"Relative key '{key}' is not in the action section of {modality_path}. "
                f"Available: {sorted(ranges['action'])}"
            )
        state_slice = ranges["state"].get(key, ranges["action"][key])
        action_slice = ranges["action"][key]

        print(f"  computing relative stats for '{key}' (horizon={horizon})...")
        data = relative_chunks_for_key(episode_paths, state_slice, action_slice, horizon)
        stats[key] = {
            "max": np.max(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
        }
        print(f"  relative: '{key}' -> shape {data.shape[1:]} from {len(data)} chunks")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    return stats_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 of the Daksha conversion pipeline: normalize a raw dataset dump's "
            "metadata (tasks.jsonl, episodes.jsonl, info.json totals) so that "
            "convert_daksha_dataset_to_v30.py can turn it into real LeRobot v3.0."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the dataset root that contains data/ and meta/.",
    )
    parser.add_argument(
        "--modality-template",
        default=str(DEFAULT_MODALITY_TEMPLATE),
        help="Path to a gr00t-style modality.json template (optional).",
    )
    parser.add_argument(
        "--chunks-size",
        type=int,
        default=DEFAULT_CHUNKS_SIZE,
        help="Value to write into meta/info.json if chunks_size is missing.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help="Action chunk length used for relative stats.",
    )
    parser.add_argument(
        "--relative-keys",
        nargs="*",
        default=list(DEFAULT_RELATIVE_KEYS),
        help="Action keys stored as RELATIVE. Default: left_arm right_arm.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip stats.json generation.",
    )
    parser.add_argument(
        "--skip-relative-stats",
        action="store_true",
        help="Skip relative_stats.json generation.",
    )
    parser.add_argument(
        "--force-stats",
        action="store_true",
        help="Recompute stats even if existing files already look complete.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = episode_parquet_paths(dataset_root)
    tasks = read_tasks(meta_dir)
    episodes = build_episodes(episode_paths)

    info_path, info = update_info_json(meta_dir, args.chunks_size, tasks, episodes)
    modality_path = copy_modality_template(meta_dir, Path(args.modality_template))
    tasks_path = write_tasks_jsonl(meta_dir, tasks)
    episodes_path = write_episodes_jsonl(meta_dir, episodes)

    print("Converted dataset metadata:")
    print(f"  dataset_root: {dataset_root}")
    print(f"  info.json: {info_path}")
    print(f"  modality.json: {modality_path if modality_path else '(skipped, no template configured)'}")
    print(f"  tasks.jsonl: {tasks_path}")
    print(f"  episodes.jsonl: {episodes_path}")
    print(f"  total_episodes: {len(episodes)}")
    print(f"  total_frames: {info['total_frames']}")
    print(f"  chunks_size: {info['chunks_size']}")

    if not args.skip_stats:
        print("Generating stats.json (standalone)...")
        stats_path = generate_stats(dataset_root, info, episode_paths, args.force_stats)
        print(f"  stats.json: {stats_path}")

    if not args.skip_relative_stats:
        if not args.relative_keys:
            print("No relative keys requested, skipping relative_stats.json.")
        elif modality_path is None:
            print("No modality.json template configured, skipping relative_stats.json.")
        else:
            print("Generating relative_stats.json (standalone)...")
            rel_path = generate_relative_stats(
                dataset_root,
                episode_paths,
                modality_path,
                args.relative_keys,
                args.action_horizon,
                args.force_stats,
            )
            print(f"  relative_stats.json: {rel_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

