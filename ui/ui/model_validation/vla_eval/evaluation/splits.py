"""Reconstruct training's ordered, per-task split from saved configuration."""

import math
from pathlib import Path


def checkpoint_episode_split(dataset_path: str, dataset_config: dict, episodes_meta: list[dict]) -> dict:
    saved_fraction = float(dataset_config.get("eval_split", 0))
    if not 0 <= saved_fraction < 1:
        raise ValueError("Saved eval_split must be in [0, 1)")
    fraction = saved_fraction or 0.1
    by_index = {episode["episode_index"]: episode for episode in episodes_meta}
    # Training preserves an explicit allowlist's order before splitting by task.
    candidates = dataset_config.get("episodes")
    if candidates is None:
        candidates = sorted(by_index)
    excluded = set(dataset_config.get("exclude_episodes") or [])
    groups = {}
    for index in candidates:
        if index in by_index and index not in excluded:
            groups.setdefault(by_index[index]["task"], []).append(index)
    evaluation = []
    for indices in groups.values():
        evaluation.extend(indices[-math.ceil(len(indices) * fraction):])
    saved_root = dataset_config.get("root")
    same_path = bool(saved_root) and Path(saved_root).expanduser().resolve() == Path(dataset_path).resolve()
    return {
        "eval_split": fraction,
        "held_out_during_training": bool(same_path and saved_fraction > 0),
        "split_source": (
            "saved_training_config" if same_path and saved_fraction > 0
            else "saved_config_dataset_unverified" if saved_fraction > 0
            else "retrospective"
        ),
        "test_episodes": sorted(set(evaluation)),
    }
