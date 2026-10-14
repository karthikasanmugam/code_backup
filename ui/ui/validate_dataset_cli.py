#!/usr/bin/env python3
"""Standalone CLI wrapper around a lightweight LeRobot dataset integrity check.

app.py's own process runs under the lightweight UI venv (Flask/psutil only -- see
requirements.txt), which does not have lerobot's real dependencies (torch, pandas,
huggingface_hub, ...) installed. This script is meant to be invoked via
resolve_repo_python() (the repo's own .venv, or `uv run`) so the check actually has what
it needs. Prints a single JSON line to stdout on success.

There is no `validate_episode_integrity` helper in this lerobot version (an earlier
version of this file imported one that doesn't exist here), so this implements its own
check directly against LeRobotDatasetMetadata: for every episode, confirm the parquet
data file and video file(s) it's mapped to actually exist and are structurally readable.
This is a fast metadata/file-integrity pass, not a frame-by-frame decode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow.parquet as pq  # noqa: E402

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402


def check_dataset_integrity(dataset_path: str, repo_id: str) -> dict:
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_path)
    total_episodes = meta.total_episodes

    data_file_status: dict[Path, str | None] = {}
    video_file_status: dict[Path, str | None] = {}
    corrupted_episodes: list[int] = []
    corruption_reasons: dict[int, str] = {}

    for ep_idx in range(total_episodes):
        problems: list[str] = []

        try:
            data_path = meta.root / meta.get_data_file_path(ep_idx)
        except Exception as e:
            problems.append(f"no data file mapping: {e}")
            data_path = None

        if data_path is not None:
            if data_path not in data_file_status:
                if not data_path.exists():
                    data_file_status[data_path] = f"data file missing: {data_path.name}"
                else:
                    try:
                        pq.ParquetFile(data_path)
                        data_file_status[data_path] = None
                    except Exception as e:
                        data_file_status[data_path] = f"unreadable parquet ({data_path.name}): {e}"
            if data_file_status[data_path]:
                problems.append(data_file_status[data_path])

        for vid_key in meta.video_keys:
            try:
                vid_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
            except Exception as e:
                problems.append(f"no video mapping for {vid_key}: {e}")
                continue
            if vid_path not in video_file_status:
                if not vid_path.exists():
                    video_file_status[vid_path] = f"video file missing: {vid_path.name}"
                elif vid_path.stat().st_size == 0:
                    video_file_status[vid_path] = f"video file empty: {vid_path.name}"
                else:
                    video_file_status[vid_path] = None
            if video_file_status[vid_path]:
                problems.append(video_file_status[vid_path])

        if problems:
            corrupted_episodes.append(ep_idx)
            corruption_reasons[ep_idx] = "; ".join(problems)

    return {
        "total_episodes": total_episodes,
        "valid_episodes": total_episodes - len(corrupted_episodes),
        "corrupted_episodes": corrupted_episodes,
        "corruption_reasons": corruption_reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()

    results = check_dataset_integrity(args.dataset_path, args.repo_id)

    total_episodes = results["total_episodes"]
    valid_episodes = results["valid_episodes"]
    corrupted = results["corrupted_episodes"]
    valid_rate = round((valid_episodes / total_episodes * 100.0), 2) if total_episodes else 0.0
    corruption_rate = round((len(corrupted) / total_episodes * 100.0), 2) if total_episodes else 0.0

    issues = [
        {"episode": int(ep), "reason": results["corruption_reasons"].get(ep, "Unknown issue")}
        for ep in corrupted[:20]
    ]

    output = {
        "status": "success",
        "validation_status": "warning" if corrupted else "ok",
        "message": (
            f"Validation complete: {valid_episodes}/{total_episodes} episodes valid"
            if total_episodes
            else "Validation complete: no episodes found"
        ),
        "path": args.dataset_path,
        "repo_id": args.repo_id,
        "total_episodes": total_episodes,
        "valid_episodes": valid_episodes,
        "valid_percentage": valid_rate,
        "corrupted_episodes": [int(ep) for ep in corrupted],
        "corruption_rate": corruption_rate,
        "issues": issues,
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
