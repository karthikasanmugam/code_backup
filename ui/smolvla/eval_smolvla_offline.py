#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.common.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_image_hwc_uint8(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.max(arr)) if arr.size else 0.0
            arr = np.clip(arr * 255.0, 0, 255) if max_val <= 1.0 + 1e-6 else np.clip(arr, 0, 255)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return arr


def _to_vector_float32(value: Any) -> np.ndarray:
    return np.asarray(_to_numpy(value), dtype=np.float32).squeeze()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline smolvla inference/evaluation on a local LeRobot dataset.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-repo-id", default="local/task1_full")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_safe_torch_device(args.device)

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
    )

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=policy_cfg)
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    available = len(dataset)
    if available == 0:
        raise ValueError("Dataset has zero frames.")

    sample_count = min(args.num_samples, available)
    rng = random.Random(args.seed)
    indices = rng.sample(range(available), sample_count)

    mae_sum = 0.0
    mse_sum = 0.0
    input_keys = [k for k in policy.config.input_features.keys() if k.startswith("observation.")]

    for idx in indices:
        frame = dataset[idx]
        observation: dict[str, np.ndarray] = {}

        for key in input_keys:
            if key not in frame:
                raise KeyError(f"Missing key '{key}' in dataset frame. Available keys: {list(frame.keys())}")
            if "image" in key:
                observation[key] = _to_image_hwc_uint8(frame[key])
            else:
                observation[key] = _to_vector_float32(frame[key])

        task = frame.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        pred = predict_action(
            observation=observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(getattr(policy.config, "use_amp", False)),
            task=str(task),
            robot_type=dataset.meta.robot_type,
        )

        pred_np = np.squeeze(pred.detach().cpu().numpy()).astype(np.float32)
        target_np = _to_vector_float32(frame["action"])
        if target_np.ndim > 1:
            target_np = target_np[0]

        mae_sum += float(np.mean(np.abs(pred_np - target_np)))
        mse_sum += float(np.mean((pred_np - target_np) ** 2))

    print("=== smolvla Offline Eval Summary ===")
    print(f"policy_path: {args.policy_path}")
    print(f"dataset: {args.dataset_repo_id}")
    print(f"dataset_root: {args.dataset_root}")
    print(f"samples: {sample_count}")
    print(f"avg_mae: {mae_sum / sample_count:.6f}")
    print(f"avg_mse: {mse_sum / sample_count:.6f}")


if __name__ == "__main__":
    main()
