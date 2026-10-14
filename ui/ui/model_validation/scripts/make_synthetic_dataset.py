"""
Builds a miniature dataset that matches the real `pick_two_boxes`-style
schema exactly (same feature names/shapes, same folder layout) but with
few episodes/frames, so the dataset adapter can be verified quickly
without needing the real ~37k-frame dataset on disk.

Usage:
    python scripts/make_synthetic_dataset.py /tmp/synthetic_lerobot_ds
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ACTION_NAMES = [
    "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
    "left_joint_5", "left_joint_6", "left_joint_7", "left_gripper_left_joint",
    "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
    "right_joint_5", "right_joint_6", "right_joint_7", "right_gripper_right_joint",
]
CAMERA_KEYS = [
    "observation.images.zed_right",
    "observation.images.zed_left",
    "observation.images.wrist_right",
    "observation.images.wrist_left",
]
FPS = 30
NUM_EPISODES = 3
FRAMES_PER_EPISODE = [20, 25, 18]
IMG_H, IMG_W = 240, 320
TASK = "pick two boxes"


def build(root: Path):
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for cam in CAMERA_KEYS:
        (root / "videos" / cam / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "custom_v3_local",
        "repo_id": "local/pick_two_boxes_synthetic",
        "robot_type": "leader_follower_6dof_4camera",
        "fps": FPS,
        "total_episodes": NUM_EPISODES,
        "total_frames": sum(FRAMES_PER_EPISODE),
        "total_tasks": 1,
        "use_videos": True,
        "camera_keys": CAMERA_KEYS,
        "video_keys": CAMERA_KEYS,
        "features": {
            "action": {"dtype": "float32", "shape": [16], "names": ACTION_NAMES},
            "observation.state": {"dtype": "float32", "shape": [16], "names": ACTION_NAMES},
            "observation.leader_state": {"dtype": "float32", "shape": [16], "names": ACTION_NAMES},
            **{
                cam: {"dtype": "video", "shape": [IMG_H, IMG_W, 3], "names": ["height", "width", "channels"]}
                for cam in CAMERA_KEYS
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "task": {"dtype": "string", "shape": [1], "names": None},
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "splits": {"train": f"0:{NUM_EPISODES}"},
    }
    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    pd.DataFrame({"task_index": [0], "task": [TASK]}).to_parquet(root / "meta" / "tasks.parquet")

    ep_rows = []
    global_index = 0
    rng = np.random.default_rng(0)

    for ep_idx, n_frames in enumerate(FRAMES_PER_EPISODE):
        rows = []
        for f_idx in range(n_frames):
            rows.append({
                "action": rng.normal(size=16).astype(np.float32),
                "observation.state": rng.normal(size=16).astype(np.float32),
                "observation.leader_state": rng.normal(size=16).astype(np.float32),
                "timestamp": f_idx / FPS,
                "frame_index": f_idx,
                "episode_index": ep_idx,
                "index": global_index,
                "task_index": 0,
                "task": TASK,
            })
            global_index += 1
        df = pd.DataFrame(rows)
        df.to_parquet(root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
        ep_rows.append({"episode_index": ep_idx, "length": n_frames, "task": TASK, "chunk_index": 0})

        for cam in CAMERA_KEYS:
            video_path = root / "videos" / cam / "chunk-000" / f"episode_{ep_idx:06d}.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_W, IMG_H)
            )
            cam_color = (CAMERA_KEYS.index(cam) * 60) % 256  # stable across processes, unlike hash()
            for f_idx in range(n_frames):
                # encode episode/camera/frame into pixel values (written here
                # in BGR order, since that's what cv2.VideoWriter expects) so
                # the test can verify the loader reads back the *correct*
                # frame rather than just "some" frame.
                color = (
                    (ep_idx * 40) % 256,  # B
                    cam_color,             # G
                    (f_idx * 10) % 256,   # R
                )
                img = np.full((IMG_H, IMG_W, 3), color, dtype=np.uint8)
                cv2.putText(img, f"e{ep_idx}_f{f_idx}", (10, IMG_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                writer.write(img)
            writer.release()

    pd.DataFrame(ep_rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "episodes.parquet")

    stats = {"action": {"mean": [0.0] * 16, "std": [1.0] * 16}}
    with open(root / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Synthetic dataset written to {root}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_lerobot_ds")
    build(out_dir)
