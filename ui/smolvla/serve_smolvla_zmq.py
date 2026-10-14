#!/usr/bin/env python3
"""
Simple ZMQ smolvla policy server for the Piper ROS2 4-camera bridge.
"""

from __future__ import annotations

import argparse
import io
import traceback
from typing import Any

import numpy as np
import torch

try:
    import msgpack
    import zmq
except ImportError as exc:
    raise ImportError("Missing dependency for ZMQ server. Install with: pip install msgpack pyzmq") from exc

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.device_utils import get_safe_torch_device


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(
            data, object_hook=MsgSerializer.decode_custom, raw=False, strict_map_key=False
        )

    @staticmethod
    def encode_custom(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj

    @staticmethod
    def decode_custom(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


def _unwrap_leading_singletons(arr: Any) -> np.ndarray:
    x = np.asarray(arr)
    while x.ndim > 0 and x.shape[0] == 1:
        x = x[0]
    return x


def _normalize_image(arr: Any) -> np.ndarray:
    img = _unwrap_leading_singletons(arr)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            max_val = float(np.max(img)) if img.size else 0.0
            img = np.clip(img * 255.0, 0, 255) if max_val <= 1.0 + 1e-6 else np.clip(img, 0, 255)
        else:
            img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    return img


def _extract_task(obs: dict[str, Any], options: dict[str, Any] | None, task_default: str) -> str:
    if isinstance(options, dict):
        for key in ("task", "instruction", "language"):
            if key in options and isinstance(options[key], str) and options[key].strip():
                return options[key]

    lang = obs.get("language", {})
    if isinstance(lang, dict):
        for key in ("annotation.human.task_description", "task"):
            if key in lang:
                value = _unwrap_leading_singletons(lang[key])
                if isinstance(value, np.ndarray):
                    if value.shape == ():
                        return str(value.item())
                    return str(value.reshape(-1)[0])
                return str(value)

    return task_default


def _build_policy_observation(obs_in: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    video = obs_in.get("video", {})
    if isinstance(video, dict):
        for cam_key, img in video.items():
            normalized_key = (
                cam_key if cam_key.startswith("observation.images.") else f"observation.images.{cam_key}"
            )
            out[normalized_key] = _normalize_image(img)

    state = obs_in.get("state", {})
    if isinstance(state, dict):
        if "observation.state" in state:
            out["observation.state"] = _unwrap_leading_singletons(state["observation.state"]).astype(
                np.float32
            )
        elif "single_arm" in state and "gripper" in state:
            arm = _unwrap_leading_singletons(state["single_arm"]).astype(np.float32).reshape(-1)
            grip = _unwrap_leading_singletons(state["gripper"]).astype(np.float32).reshape(-1)
            out["observation.state"] = np.concatenate([arm, grip], axis=0).astype(np.float32)
        elif "right_arm" in state and "left_arm" in state:
            r_arm = _unwrap_leading_singletons(state["right_arm"]).astype(np.float32).reshape(-1)
            r_grip = _unwrap_leading_singletons(state["right_gripper"]).astype(np.float32).reshape(-1)
            l_arm = _unwrap_leading_singletons(state["left_arm"]).astype(np.float32).reshape(-1)
            l_grip = _unwrap_leading_singletons(state["left_gripper"]).astype(np.float32).reshape(-1)
            out["observation.state"] = np.concatenate([r_arm, r_grip, l_arm, l_grip], axis=0).astype(np.float32)

    if "observation.state" not in out:
        raise ValueError("Missing state. Expected single_arm + gripper, right_arm + left_arm, or observation.state")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ZMQ smolvla policy server for Piper ROS2 bridge")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--robot-type", default="bimanual_16dof_4camera")
    parser.add_argument("--task-default", default="task1")
    parser.add_argument("--action-horizon", type=int, default=0, help="0 means full policy chunk")
    args = parser.parse_args()

    device = get_safe_torch_device(args.device)

    print(f"[serve_smolvla_zmq] Loading checkpoint: {args.policy_path}", flush=True)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=policy_cfg)
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(
        f"[serve_smolvla_zmq] CHECKPOINT LOADED: path={args.policy_path} "
        f"type={policy.config.type} device={device}",
        flush=True,
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[serve_smolvla_zmq] listening on tcp://{args.host}:{args.port}", flush=True)
    req_count = 0

    while True:
        req_raw = sock.recv()
        try:
            req = MsgSerializer.from_bytes(req_raw)
            endpoint = req.get("endpoint")

            if endpoint == "ping":
                sock.send(MsgSerializer.to_bytes({"ok": True}))
                continue

            if endpoint == "reset":
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                sock.send(MsgSerializer.to_bytes({"ok": True}))
                continue

            if endpoint != "get_action":
                sock.send(MsgSerializer.to_bytes({"error": f"Unknown endpoint: {endpoint}"}))
                continue

            data = req.get("data", {}) or {}
            obs_in = data.get("observation", {}) or {}
            options = data.get("options", {}) or {}
            req_count += 1

            obs_np = _build_policy_observation(obs_in)
            task = _extract_task(obs_in, options, args.task_default)

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type)
                if device.type == "cuda" and getattr(policy.config, "use_amp", False)
                else torch.no_grad(),
            ):
                obs_t = prepare_observation_for_inference(obs_np, device, task, args.robot_type)
                obs_t = preprocessor(obs_t)
                chunk = policy.predict_action_chunk(obs_t)

                processed = []
                for t in range(chunk.shape[1]):
                    processed.append(postprocessor(chunk[:, t, :]))
                action = torch.stack(processed, dim=1).detach().cpu().numpy().astype(np.float32)

            if args.action_horizon > 0:
                action = action[:, : args.action_horizon]

            if req_count % 10 == 0:
                print(
                    f"[serve_smolvla_zmq] get_action requests={req_count} "
                    f"horizon={action.shape[1]} task='{task}' obs_keys={list(obs_np.keys())}"
                )

            response = [{"action": action}, {"horizon": int(action.shape[1]), "task": task}]
            sock.send(MsgSerializer.to_bytes(response))

        except Exception as exc:
            traceback.print_exc()
            sock.send(MsgSerializer.to_bytes({"error": str(exc)}))


if __name__ == "__main__":
    main()
