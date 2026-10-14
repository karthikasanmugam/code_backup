#!/usr/bin/env python3
"""
ROS2 Daksha B1 bridge for LeRobot SmolVLA policy checkpoints.

Dedicated inference bridge for policies trained under examples/daksha/smolvla. The
checkpoint itself is never loaded here -- it is loaded once by
examples/daksha/smolvla/serve_smolvla_zmq.py, a ZMQ server started by
ui/app.py's start_inference_server() for any checkpoint whose config.json has
"type": "smolvla" (it also passes that server a --robot-type flag read from the
checkpoint's training dataset, which the generic ACT/Diffusion path does not need).
Start the server first (or use run_inference.sh, which starts the server and this
bridge together and does not start the control loop until the checkpoint has
actually finished loading):

    python examples/daksha/smolvla/serve_smolvla_zmq.py \\
        --policy-path <run_dir_or_pretrained_model_dir> --device cuda

This file is forked from examples/daksha/inference/eval_daksha_act.py, which remains
the bridge for ACT checkpoints (ui/app.py's ACT_BRIDGE_SCRIPT vs. SMOLVLA_BRIDGE_SCRIPT,
chosen by the checkpoint's own policy type -- see start_bridge() there). The ZMQ wire
protocol, safety clamping and ROS2 topic handling below are currently identical to
that file since both talk to the same generic PolicyZmqClient protocol; keeping them as
separate files lets SmolVLA-specific bridge behavior (e.g. a different default
--action-horizon/--replan-at to match SmolVLA's action-chunk size, or extra
--robot-type-aware handling) diverge here later without touching the ACT/Diffusion
bridges.

Wire protocol: msgpack over a ZMQ REQ/REP socket (see PolicyZmqClient below, matching
serve_smolvla_zmq.py / examples/daksha/ui/inference_client.py). This is *not* the
GR00T PolicyClient wire format -- there is no external gr00t dependency here.

Policy state/action layout (16 DoF): the recorded dataset's own column order, i.e.
whatever --state-obs-keys-csv / --action-keys-csv are given as. Verified against every
Daksha B1 dataset under examples/daksha (meta/info.json -> features.action.names):
LEFT-first --
  left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
The bridge sends this as a single flat "observation.state" vector (in exactly that
order) rather than the server's separate right_arm/left_arm grouping -- that grouping
concatenates right-first internally, which would silently swap the arms for a
checkpoint trained (like every Daksha B1 dataset checked) on the left-first layout.
If you ever train on a dataset with a different column order, pass matching
--state-obs-keys-csv / --action-keys-csv; nothing else in this file assumes a fixed
side-to-slot mapping.

Video modality keys are read from the selected checkpoint by the UI and passed to
this bridge. The four physical camera slots can therefore feed checkpoints trained
with names such as world, zed_right, wrist_right, and wrist_left. The Daksha UI maps
the world slot to /camera/world/color/image_raw/compressed by default.

Action representation: examples/daksha/act/train_act_local.sh and
diffusion/train_diffusion_local.sh do not configure any relative/delta-action
wrapper, so the policy is trained on -- and predicts -- ABSOLUTE joint positions
directly in the units the dataset was recorded in. (Unlike a GR00T checkpoint using
use_relative_action=True, there is no relative->absolute conversion happening
server-side here.) If a future checkpoint does use delta actions, this assumption
needs revisiting before trusting the safety clamp below.

Safety clamp limits (--joint-min-csv / --joint-max-csv / --max-step-delta-csv) have no
built-in default: they must be derived from the checkpoint's own training dataset, not
copied from another checkpoint or robot. To derive them, in --state-obs-keys-csv /
--action-keys-csv order:

    python - <<'EOF'
    import json
    keys = "left_joint_1,left_joint_2,left_joint_3,left_joint_4,left_joint_5,left_joint_6,left_joint_7,left_gripper_left_joint,right_joint_1,right_joint_2,right_joint_3,right_joint_4,right_joint_5,right_joint_6,right_joint_7,right_gripper_right_joint".split(",")
    stats = json.load(open("<dataset_root>/meta/stats.json"))
    names = json.load(open("<dataset_root>/meta/info.json"))["features"]["action"]["names"]
    idx = [names.index(k) for k in keys]
    lo = [min(stats["action"]["min"][i], stats["observation.state"]["min"][i]) for i in idx]
    hi = [max(stats["action"]["max"][i], stats["observation.state"]["max"][i]) for i in idx]
    margin = [0.01 if "gripper" in k else max(0.1 * (h - l), 0.05) for k, l, h in zip(keys, lo, hi)]
    print("--joint-min-csv", ",".join(f"{l - m:.5f}" for l, m in zip(lo, margin)))
    print("--joint-max-csv", ",".join(f"{h + m:.5f}" for h, m in zip(hi, margin)))
    EOF

Commands are published as sensor_msgs/JointState on --cmd-topic (/joint_cmd) with
default (reliable, depth 10) QoS -- deliberately *not* qos_profile_sensor_data, which
is best-effort and can silently drop a command. Best-effort QoS is only appropriate
for the sensor_data *subscriptions* below (joints, cameras), where a dropped stale
frame is fine and a blocked publisher is not.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import math
import time
from collections import deque
from typing import Any

import cv2
import msgpack
import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, JointState

# ════════════════════════════════════════════════════════════════════════════════════
# ZMQ CLIENT -- matches serve_smolvla_zmq.py / examples/daksha/serve_policy_zmq.py's
# wire protocol exactly (same MsgSerializer as smolvla/eval_piper_ros2_remote_4cam.py
# and ui/inference_client.py). Duplicated inline rather than imported: this file has to
# run under the ROS2 system interpreter (no lerobot installed there), matching how
# every other daksha ROS2 bridge script in this repo is self-contained.
# ════════════════════════════════════════════════════════════════════════════════════


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


class PolicyZmqClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 15000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self._make_socket()

    def _make_socket(self):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        if hasattr(zmq, "REQ_RELAXED"):
            socket.setsockopt(zmq.REQ_RELAXED, 1)
        if hasattr(zmq, "REQ_CORRELATE"):
            socket.setsockopt(zmq.REQ_CORRELATE, 1)
        socket.connect(f"tcp://{self.host}:{self.port}")
        return socket

    def _reset_socket(self) -> None:
        try:
            self.socket.close(0)
        except Exception:
            pass
        self.socket = self._make_socket()

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        req: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data or {}
        try:
            self.socket.send(MsgSerializer.to_bytes(req))
            reply = self.socket.recv()
        except zmq.error.Again as exc:
            self._reset_socket()
            raise TimeoutError(
                f"Timed out waiting for policy server reply after {self.timeout_ms} ms"
            ) from exc
        except zmq.error.ZMQError as exc:
            self._reset_socket()
            raise RuntimeError(f"ZMQ request failed: {exc}") from exc
        response = MsgSerializer.from_bytes(reply)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call("ping", requires_input=False)
            return True
        except Exception:
            return False

    def reset(self) -> None:
        self.call("reset", requires_input=False)

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call(
            "get_action", {"observation": observation, "options": options}, requires_input=True
        )
        return tuple(response)

    def wait_until_ready(self, timeout_s: float, logger=None) -> bool:
        """Poll ping() until the server answers or timeout_s elapses.

        The server binds its socket immediately but doesn't start answering "ping"
        until the checkpoint has finished loading onto the GPU (see
        examples/daksha/serve_policy_zmq.py), which can take anywhere from a few
        seconds to ~30s+ depending on policy size. Failing fast here would make the
        bridge racy against run_inference.sh's checkpoint load; waiting means the
        control loop only ever starts once the model is actually ready to serve.
        """
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        attempt = 0
        while True:
            if self.ping():
                return True
            attempt += 1
            if deadline is not None and time.monotonic() >= deadline:
                return False
            if logger is not None and attempt % 5 == 1:
                logger.info(
                    f"Waiting for policy server at tcp://{self.host}:{self.port} "
                    f"(checkpoint still loading?), attempt {attempt}..."
                )
            time.sleep(1.0)


# ════════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════════


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str, expected_len: int, name: str) -> np.ndarray:
    vals = np.array([float(x.strip()) for x in value.split(",") if x.strip()], dtype=np.float32)
    if vals.shape[0] != expected_len:
        raise ValueError(f"{name} must have {expected_len} values, got {vals.shape[0]}: {value}")
    return vals


def joint_name_aliases(name: str) -> set[str]:
    aliases = {name}
    if "/" in name:
        aliases.add(name.split("/", 1)[1])
    return aliases


def side_of(key: str) -> str:
    k = key.lower()
    if "left" in k:
        return "left"
    if "right" in k:
        return "right"
    return "?"


def recursive_add_extra_dim(obs: dict[str, Any]) -> dict[str, Any]:
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


# ════════════════════════════════════════════════════════════════════════════════════
# BRIDGE NODE
# ════════════════════════════════════════════════════════════════════════════════════


class Ros2DakshaB1SmolVLABridge(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("daksha_b1_smolvla_ros2_bridge")
        self.args = args

        self.policy = PolicyZmqClient(
            host=args.policy_host, port=args.policy_port, timeout_ms=args.policy_timeout_ms
        )
        self.get_logger().info(
            f"Waiting for policy server at tcp://{args.policy_host}:{args.policy_port} "
            f"(up to {args.server_wait_timeout_s:.0f}s -- the checkpoint has to finish loading)..."
        )
        if not self.policy.wait_until_ready(args.server_wait_timeout_s, logger=self.get_logger()):
            raise RuntimeError(
                f"Could not reach policy server at tcp://{args.policy_host}:{args.policy_port} "
                f"within {args.server_wait_timeout_s:.0f}s. Is examples/daksha/serve_policy_zmq.py "
                "running (or still loading the checkpoint)?"
            )
        self.get_logger().info("Policy server is up.")

        self.state_obs_keys = parse_csv_list(args.state_obs_keys_csv)
        self.action_keys = parse_csv_list(args.action_keys_csv)

        if len(self.state_obs_keys) != 16:
            raise ValueError("state_obs_keys_csv must contain exactly 16 keys")
        if len(self.action_keys) != 16:
            raise ValueError("action_keys_csv must contain exactly 16 keys")
        for name, keys in (("state_obs_keys", self.state_obs_keys), ("action_keys", self.action_keys)):
            if len(set(keys)) != len(keys):
                raise ValueError(f"{name} contains duplicate keys: {keys}")
            n_gripper = sum(1 for k in keys if "gripper" in k.lower())
            if n_gripper != 2:
                raise ValueError(f"{name} must contain exactly 2 'gripper' keys, found {n_gripper}: {keys}")

        self.active_sides = set(parse_csv_list(args.active_sides_csv))
        if not self.active_sides or not self.active_sides <= {"left", "right"}:
            raise ValueError(
                f"active_sides_csv must contain left and/or right, got {args.active_sides_csv!r}"
            )
        self.inactive_action_indices = np.array(
            [i for i, key in enumerate(self.action_keys) if side_of(key) not in self.active_sides],
            dtype=np.int64,
        )
        if len(self.inactive_action_indices) == 16:
            raise ValueError("active_sides_csv does not match any action keys")
        self.get_logger().info(
            f"Active prediction arms: {','.join(side for side in ('left', 'right') if side in self.active_sides)}; "
            f"holding inactive arm dimensions at live measured positions"
        )

        # Correctness of everything below hinges entirely on this order matching the
        # order the checkpoint was trained on (features.action.names in the training
        # dataset's meta/info.json) -- there is no side-to-slot assumption left to
        # sanity-check it against, so it is logged loudly instead.
        self.get_logger().info(f"state_obs_keys (order sent as observation.state): {self.state_obs_keys}")
        self.get_logger().info(f"action_keys (order the policy's action vector is read as): {self.action_keys}")

        if not 0 <= args.replan_at < args.action_horizon:
            raise ValueError(
                f"--replan-at must be in [0, action_horizon), got {args.replan_at} "
                f"with --action-horizon {args.action_horizon}"
            )
        self.replan_budget_s = args.replan_at / args.control_hz

        self.joint_min = parse_csv_floats(args.joint_min_csv, 16, "joint_min_csv")
        self.joint_max = parse_csv_floats(args.joint_max_csv, 16, "joint_max_csv")
        self.max_step_delta = parse_csv_floats(args.max_step_delta_csv, 16, "max_step_delta_csv")
        self.prev_cmd: np.ndarray | None = None
        self.last_policy_ok_time = 0.0

        # Preserve all 16 ROS command slots so index-based consumers keep the
        # robot convention (left 0..7, right 8..15). Inactive arm values are
        # replaced with measured hold positions before this message is built.
        all_indices = list(range(16))
        if args.publish_joint_order == "as_is":
            self.publish_indices = np.asarray(all_indices, dtype=np.int64)
        else:
            first_side = "left" if args.publish_joint_order == "left_first" else "right"
            first = [i for i, key in enumerate(self.action_keys) if side_of(key) == first_side]
            rest = [i for i in all_indices if i not in first]
            self.publish_indices = np.array(first + rest, dtype=np.int64)

        self.camera_sources = (
            ("zed_right", "zed_right_topic", "zed_right_camera_key"),
            ("zed_left", "zed_left_topic", "zed_left_camera_key"),
            ("wrist_right", "right_wrist_topic", "right_wrist_camera_key"),
            ("wrist_left", "left_wrist_topic", "left_wrist_camera_key"),
        )
        active_sources = [source for source, _, key_attr in self.camera_sources if getattr(args, key_attr)]
        policy_camera_keys = [getattr(args, key_attr) for _, _, key_attr in self.camera_sources if getattr(args, key_attr)]
        if not active_sources:
            raise ValueError("The selected checkpoint has no supported camera inputs")
        if len(set(policy_camera_keys)) != len(policy_camera_keys):
            raise ValueError(f"Duplicate checkpoint camera keys: {policy_camera_keys}")
        self.camera_keys = tuple(active_sources)

        # Receipt timestamps, so a publisher that dies is detected instead of the
        # bridge silently re-sending the last frame it ever saw. Without this the
        # policy keeps scoring a frozen image, predicts "no motion" every chunk, and
        # the arm sits still with no error anywhere -- indistinguishable from a bad
        # checkpoint. Cameras dropping out is exactly the observed failure.
        self.image_stamps: dict[str, float] = {}
        self.joint_stamps: dict[str, float] = {}

        self.latest_joint_map: dict[str, float] = {}
        self.latest_joint_names: list[str] = []
        self.latest_images: dict[str, np.ndarray] = {}
        self.planned_actions: deque[JointState] = deque()
        self.pending_plan_future = None
        self.shape_log_done = False
        self.plan_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.valid_modes = ["hold", "full"]
        if args.run_validation_suite:
            self.phase_modes = list(self.valid_modes)
        else:
            if args.validation_mode not in self.valid_modes:
                raise ValueError(f"validation_mode must be one of {self.valid_modes}")
            self.phase_modes = [args.validation_mode]
        self.phase_idx = 0
        self.phase_start_t = time.monotonic()
        self.finished = False
        self.get_logger().info(f"Validation phases: {self.phase_modes}")

        if args.right_joint_state_topic or args.left_joint_state_topic:
            if args.right_joint_state_topic:
                self.create_subscription(
                    JointState,
                    args.right_joint_state_topic,
                    lambda m: self._joint_cb(m, side="right"),
                    qos_profile_sensor_data,
                )
            if args.left_joint_state_topic:
                self.create_subscription(
                    JointState,
                    args.left_joint_state_topic,
                    lambda m: self._joint_cb(m, side="left"),
                    qos_profile_sensor_data,
                )
        else:
            self.create_subscription(
                JointState, args.joint_state_topic, self._joint_cb, qos_profile_sensor_data
            )

        # The dataset was recorded from the "<topic>/compressed" variants
        # (ros2_topics.json). Keep the raw base topics as the defaults and append
        # the "/compressed" suffix automatically when compressed transport is
        # selected, so both modes subscribe to the topics the data came from.
        image_msg_type = CompressedImage if args.image_transport == "compressed" else Image
        for source, topic_attr, key_attr in self.camera_sources:
            if not getattr(args, key_attr):
                continue
            topic = getattr(args, topic_attr)
            if args.image_transport == "compressed" and not topic.endswith("/compressed"):
                topic = f"{topic}/compressed"
            self.create_subscription(
                image_msg_type,
                topic,
                lambda msg, cam=source: self._image_cb(cam, msg),
                10,
            )

        # Inference publishes a single JointState command topic even if the dataset
        # was originally recorded from separate arm controller topics. Default QoS
        # (reliable, depth 10) -- deliberately not qos_profile_sensor_data; see the
        # module docstring for why a command topic must not be best-effort.
        self.cmd_pub = self.create_publisher(JointState, args.cmd_topic, 10)

        self.timer = self.create_timer(1.0 / args.control_hz, self._control_step)
        self.get_logger().info(
            f"Bridge started. Server=tcp://{args.policy_host}:{args.policy_port}, "
            f"cmd_topic={args.cmd_topic}, image_transport={args.image_transport}"
        )

    def _joint_cb(self, msg: JointState, side: str | None = None) -> None:
        now = time.monotonic()
        if msg.name and msg.position and len(msg.name) == len(msg.position):
            if not self.latest_joint_names:
                self.latest_joint_names = list(msg.name)
                if self.latest_joint_names != self.action_keys:
                    self.get_logger().info(
                        "Incoming joint names differ from checkpoint bridge names; matching names "
                        "where possible and using ordered per-arm positions when needed."
                    )
            for n, p in zip(msg.name, msg.position):
                for alias in joint_name_aliases(n):
                    self.latest_joint_map[alias] = float(p)
                    self.joint_stamps[alias] = now

        # Some converted datasets use follower_joint_1..16 / joint_1..16 names,
        # while ROS publishes ordered per-arm JointState messages. If names did
        # not match this side's canonical bridge keys, use the topic's known order.
        if side in ("left", "right"):
            keys = [k for k in self.state_obs_keys if side_of(k) == side]
            if keys and len(msg.position) == len(keys) and not all(k in self.joint_stamps and self.joint_stamps[k] == now for k in keys):
                for key, value in zip(keys, msg.position):
                    self.latest_joint_map[key] = float(value)
                    self.joint_stamps[key] = now
        elif not msg.name and msg.position:
            for key, value in zip(self.state_obs_keys, msg.position):
                self.latest_joint_map[key] = float(value)
                self.joint_stamps[key] = now

    def _maybe_downscale(self, img: np.ndarray) -> np.ndarray:
        """Cap the image's longest edge at --image-max-edge (0 disables)."""
        limit = self.args.image_max_edge
        if limit <= 0:
            return img
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= limit:
            return img
        scale = limit / float(longest)
        return cv2.resize(
            img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _image_cb(self, cam_key: str, msg: Image | CompressedImage) -> None:
        if isinstance(msg, CompressedImage):
            try:
                compressed = np.frombuffer(msg.data, dtype=np.uint8)
                if len(compressed) == 0:
                    self.get_logger().warn(f"Received empty compressed image data from {cam_key}")
                    return
                img = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
                if img is None:
                    self.get_logger().warn(f"Failed to decode compressed image from {cam_key}")
                    return
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.latest_images[cam_key] = self._maybe_downscale(img)
                self.image_stamps[cam_key] = time.monotonic()
            except Exception as e:
                self.get_logger().warn(f"Error processing compressed image from {cam_key}: {e}")
            return

        if msg.encoding not in ("rgb8", "bgr8", "bgra8", "rgba8"):
            self.get_logger().warn(f"Unsupported raw image encoding for {cam_key}: {msg.encoding}")
            return
        channels = 4 if msg.encoding in ("bgra8", "rgba8") else 3
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, channels))
        if msg.encoding == "bgr8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif msg.encoding == "bgra8":
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif msg.encoding == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        self.latest_images[cam_key] = self._maybe_downscale(img)
        self.image_stamps[cam_key] = time.monotonic()

    def _current_mode(self) -> str | None:
        if self.finished:
            return None
        mode = self.phase_modes[self.phase_idx]
        if self.args.run_validation_suite:
            elapsed = time.monotonic() - self.phase_start_t
            if elapsed >= self.args.phase_seconds:
                self.get_logger().info(f"Completed phase: {mode}")
                self.phase_idx += 1
                if self.phase_idx >= len(self.phase_modes):
                    self.get_logger().info("Validation suite complete.")
                    self.finished = True
                    return None
                self.phase_start_t = time.monotonic()
                mode = self.phase_modes[self.phase_idx]
                self.planned_actions.clear()
                self.get_logger().info(f"Starting phase: {mode}")
        return mode

    def _missing_inputs(self) -> tuple[list[str], list[str]]:
        now = time.monotonic()
        timeout = self.args.input_timeout_s

        def stale(key: str, stamps: dict[str, float]) -> str | None:
            if key not in stamps:
                return key
            age = now - stamps[key]
            return f"{key} (stale {age:.1f}s)" if timeout > 0 and age > timeout else None

        missing_joints = [s for s in (stale(k, self.joint_stamps) for k in self.state_obs_keys) if s]
        missing_cameras = [s for s in (stale(k, self.image_stamps) for k in self.camera_keys) if s]
        return missing_joints, missing_cameras

    def _ready(self) -> bool:
        missing_joints, missing_cameras = self._missing_inputs()
        return not missing_joints and not missing_cameras

    def _log_waiting_inputs(self) -> None:
        missing_joints, missing_cameras = self._missing_inputs()
        parts: list[str] = []
        if missing_joints:
            if self.args.right_joint_state_topic or self.args.left_joint_state_topic:
                topic_desc = (
                    f"right={self.args.right_joint_state_topic or 'disabled'}, "
                    f"left={self.args.left_joint_state_topic or 'disabled'}"
                )
            else:
                topic_desc = self.args.joint_state_topic
            parts.append(f"waiting on {topic_desc} for joint keys: {', '.join(missing_joints)}")
        if missing_cameras:
            camera_topics = {
                "zed_right": self.args.zed_right_topic,
                "zed_left": self.args.zed_left_topic,
                "wrist_right": self.args.right_wrist_topic,
                "wrist_left": self.args.left_wrist_topic,
            }
            missing_cam_desc = ", ".join(f"{k} ({camera_topics[k]})" for k in missing_cameras)
            parts.append(f"waiting on camera topics: {missing_cam_desc}")
        if not parts:
            parts.append("all required ROS2 inputs received, but policy is not ready yet")
        self.get_logger().info("Waiting for inputs -> " + " | ".join(parts))

    def _hold_vec16(self) -> np.ndarray:
        return np.array([self.latest_joint_map[k] for k in self.state_obs_keys], dtype=np.float32)

    def _log_obs_shapes(self, obs: dict[str, Any]) -> None:
        video_shapes = {k: tuple(v.shape) for k, v in obs["video"].items()}
        state_shapes = {k: tuple(np.asarray(v).shape) for k, v in obs["state"].items()}
        language_shapes = {k: (len(v), len(v[0]) if v else 0) for k, v in obs["language"].items()}
        self.get_logger().info(
            f"Policy obs shapes: video={video_shapes}, state={state_shapes}, language={language_shapes}"
        )

    def _obs_for_policy(self, state16: np.ndarray | None = None) -> dict[str, Any]:
        if state16 is None:
            state16 = self._hold_vec16()
        obs = {
            "video": {
                getattr(self.args, key_attr): self.latest_images[source]
                for source, _, key_attr in self.camera_sources
                if getattr(self.args, key_attr)
            },
            # Sent as a single flat vector in --state-obs-keys-csv order, matching
            # examples/daksha/serve_policy_zmq.py's `"observation.state" in state`
            # branch directly -- see the module docstring for why the separate
            # right_arm/left_arm grouped branch is deliberately not used here.
            "state": {"observation.state": state16},
            "language": {"annotation.human.task_description": self.args.lang_instruction},
        }
        obs = recursive_add_extra_dim(obs)
        obs = recursive_add_extra_dim(obs)
        if not self.shape_log_done:
            self._log_obs_shapes(obs)
            self.shape_log_done = True
        return obs

    @staticmethod
    def _extract_action_vec16(action_chunk: dict[str, Any], t: int) -> np.ndarray:
        # examples/daksha/serve_policy_zmq.py always returns a single flat "action"
        # array (postprocessor output is never split into per-group keys) -- see
        # serve_smolvla_zmq.py's `response = [{"action": action}, ...]`.
        if "action" not in action_chunk:
            raise ValueError(f"Unsupported action chunk keys: {list(action_chunk.keys())}")
        vec = np.asarray(action_chunk["action"][0][t], dtype=np.float32).reshape(-1)
        if vec.shape[0] != 16:
            raise ValueError(f"Expected action[16], got {vec.shape[0]}")
        return vec

    def _apply_safety(self, action_vec16: np.ndarray, hold_vec16: np.ndarray, mode: str) -> np.ndarray:
        if self.prev_cmd is None:
            self.prev_cmd = hold_vec16.copy()

        prev = self.prev_cmd
        target = np.clip(action_vec16, self.joint_min, self.joint_max)
        delta = np.clip(target - prev, -self.max_step_delta, self.max_step_delta)
        # Do not re-clip the stepped command: if the measured pose starts outside
        # recorded bounds, that would bypass max_step_delta and cause a sudden jump.
        cmd = prev + delta

        if mode == "hold":
            cmd = hold_vec16.copy()
        elif len(self.inactive_action_indices):
            # Joint-limit clipping can otherwise move an inactive arm when its
            # measured hold position falls outside that arm's configured limits.
            cmd[self.inactive_action_indices] = hold_vec16[self.inactive_action_indices]
        self.prev_cmd = cmd.copy()
        return cmd

    def _to_joint_state_msg(self, cmd16: np.ndarray) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self.action_keys[i] for i in self.publish_indices]
        msg.position = [float(cmd16[i]) for i in self.publish_indices]
        # Many hardware controllers require velocity / effort fields to be present
        # (even if zero) to accept the JointState as a valid command.
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
        return msg

    def _plan_from_policy(self, mode: str) -> list[JointState]:
        hold_vec16 = self._hold_vec16()
        obs = self._obs_for_policy(hold_vec16)
        t_start = time.monotonic()
        action_chunk, _ = self.policy.get_action(obs)
        inference_s = time.monotonic() - t_start
        self.last_policy_ok_time = time.monotonic()

        if self.args.replan_at > 0:
            budget_s = self.replan_budget_s
            note = ""
            if inference_s > budget_s:
                note = f"  OVER BUDGET -- raise --replan-at to {math.ceil(inference_s * self.args.control_hz)}"
            self.get_logger().info(
                f"inference {inference_s * 1000:.0f} ms vs budget {budget_s * 1000:.0f} ms{note}"
            )
        else:
            self.get_logger().info(
                f"inference {inference_s * 1000:.0f} ms "
                f"(~{inference_s * self.args.control_hz:.1f} ticks held between chunks; replanning disabled)"
            )

        horizon = int(np.asarray(action_chunk["action"]).shape[1])
        steps = min(horizon, self.args.action_horizon)
        self.get_logger().info(
            f"Policy action shape: {tuple(np.asarray(action_chunk['action']).shape)}, "
            f"horizon={horizon}, steps={steps}, mode={mode}"
        )

        # Re-anchor the rate limiter to the state this chunk was predicted from -- see
        # the equivalent note in examples/daksha/smolvla/eval_piper_ros2_remote_4cam.py.
        # Actions here are absolute (not relative), so this mainly keeps the rate limit
        # anchored to a state actually measured at plan time rather than a possibly
        # stale `prev_cmd` left over from whatever the previous chunk ended on.
        self.prev_cmd = hold_vec16.copy()

        # Diagnostic: tells apart "the checkpoint isn't predicting motion" from "motion
        # was predicted but the safety clamp ate it" -- both look identical (arm doesn't
        # move) from the ROS log alone otherwise. Compares the *last* chunk step against
        # the hold state this chunk was planned from, before and after _apply_safety,
        # split by arm via the same side_of() key scan used for publish_indices.
        left_idx = [i for i, k in enumerate(self.action_keys) if side_of(k) == "left"]
        right_idx = [i for i, k in enumerate(self.action_keys) if side_of(k) == "right"]

        planned: list[JointState] = []
        raw_last: np.ndarray | None = None
        cmd_last: np.ndarray | None = None
        for t in range(steps):
            raw_vec16 = self._extract_action_vec16(action_chunk, t)
            command_vec16 = raw_vec16.copy()
            if len(self.inactive_action_indices):
                command_vec16[self.inactive_action_indices] = hold_vec16[self.inactive_action_indices]
            cmd16 = self._apply_safety(command_vec16, hold_vec16, mode)
            planned.append(self._to_joint_state_msg(cmd16))
            raw_last, cmd_last = raw_vec16, cmd16

        if raw_last is not None:
            raw_delta = np.abs(raw_last - hold_vec16)
            cmd_delta = np.abs(cmd_last - hold_vec16)
            self.get_logger().info(
                f"Policy-active joints: {[key for key in self.action_keys if side_of(key) in self.active_sides]}; "
            f"ROS command order: {[self.action_keys[i] for i in self.publish_indices]}"
            )
            self.get_logger().info(
                f"chunk-end delta vs hold -- raw policy: left_max={raw_delta[left_idx].max():.4f} "
                f"right_max={raw_delta[right_idx].max():.4f} | commanded after active-arm mask: "
                f"left_max={cmd_delta[left_idx].max():.4f} right_max={cmd_delta[right_idx].max():.4f} "
                f"(raw >> clamped on an arm means the policy wants to move it but the clamp is stopping "
                f"it; raw~0 means the checkpoint itself isn't predicting motion for that arm)"
            )
        return planned

    def _publish_hold(self) -> None:
        hold_vec16 = self._hold_vec16()
        self.cmd_pub.publish(self._to_joint_state_msg(hold_vec16))

    def _control_step(self) -> None:
        mode = self._current_mode()
        if mode is None:
            self.pending_plan_future = None
            if self._ready():
                self._publish_hold()
            return
        if not self._ready():
            self._log_waiting_inputs()
            return

        stale = (
            self.last_policy_ok_time > 0
            and (time.monotonic() - self.last_policy_ok_time) > self.args.watchdog_timeout_s
        )
        if mode == "hold" or stale:
            self.pending_plan_future = None
            self.planned_actions.clear()
            self._publish_hold()
            return

        if self.pending_plan_future is not None and self.pending_plan_future.done():
            try:
                planned = self.pending_plan_future.result()
                self.planned_actions.clear()
                self.planned_actions.extend(planned)
            except Exception as exc:
                self.get_logger().error(f"Policy planning failed: {exc}")
            finally:
                self.pending_plan_future = None

        if self.pending_plan_future is None and len(self.planned_actions) <= self.args.replan_at:
            try:
                self.pending_plan_future = self.plan_executor.submit(self._plan_from_policy, mode)
            except Exception as exc:
                self.get_logger().error(f"Could not start policy planning: {exc}")
                self.pending_plan_future = None

        if self.planned_actions:
            self.cmd_pub.publish(self.planned_actions.popleft())
        else:
            self._publish_hold()

    def destroy_node(self) -> bool:
        self.plan_executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Daksha B1 ROS2 bridge for LeRobot SmolVLA policy checkpoints")

    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=5555)
    p.add_argument("--policy-timeout-ms", type=int, default=15000, help="per-request timeout once connected")
    p.add_argument(
        "--server-wait-timeout-s",
        type=float,
        default=120.0,
        help="how long to wait for the policy server to answer ping before giving up "
        "(0 = wait forever). Covers checkpoint load time, not just process startup.",
    )
    # Must match the training task string exactly (see the dataset's meta/tasks.jsonl).
    p.add_argument("--lang-instruction", default="task1")
    p.add_argument("--action-horizon", type=int, default=16)
    # Must match the training dataset's recording rate (meta/info.json -> fps),
    # otherwise the chunk is played back at the wrong speed. Varies per dataset in
    # this repo (10 or 30 fps observed) -- check the specific checkpoint's dataset.
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--replan-at", type=int, default=0)
    p.add_argument("--watchdog-timeout-s", type=float, default=2.0)
    p.add_argument("--input-timeout-s", type=float, default=1.0)

    p.add_argument("--validation-mode", choices=["hold", "full"], default="full")
    p.add_argument("--run-validation-suite", action="store_true")
    p.add_argument("--phase-seconds", type=float, default=12.0)

    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--right-joint-state-topic", default="/RightArmSystem_ordered_joint_states")
    p.add_argument("--left-joint-state-topic", default="/LeftArmSystem_ordered_joint_states")
    p.add_argument("--cmd-topic", default="/joint_cmd")
    p.add_argument("--publish-joint-order", choices=["as_is", "left_first", "right_first"], default="left_first")
    p.add_argument(
        "--active-sides-csv",
        default="left,right",
        help="which arms were active in this checkpoint's training dataset; inactive arms hold live state",
    )
    # The trained zed_right modality is supplied by the world camera on this robot.
    # Keep the raw base topic here; compressed transport appends /compressed below.
    p.add_argument("--zed-right-topic", default="/camera/world/color/image_raw")
    p.add_argument("--zed-left-topic", default="/zed/zed_node/left/color/rect/image")
    p.add_argument("--right-wrist-topic", default="/right/camera/color/image_raw")
    p.add_argument("--left-wrist-topic", default="/left/camera/color/image_raw")
    p.add_argument("--image-transport", choices=["compressed", "raw"], default="compressed")
    p.add_argument("--image-max-edge", type=int, default=0)

    p.add_argument("--zed-right-camera-key", default="zed_right")
    p.add_argument("--zed-left-camera-key", default="zed_left")
    p.add_argument("--right-wrist-camera-key", default="wrist_right")
    p.add_argument("--left-wrist-camera-key", default="wrist_left")

    # Default order verified against every Daksha B1 dataset's meta/info.json ->
    # features.action.names in this repo (left-first). See the module docstring.
    p.add_argument(
        "--state-obs-keys-csv",
        default=(
            "left_joint_1,left_joint_2,left_joint_3,left_joint_4,left_joint_5,left_joint_6,left_joint_7,left_gripper_left_joint,"
            "right_joint_1,right_joint_2,right_joint_3,right_joint_4,right_joint_5,right_joint_6,right_joint_7,right_gripper_right_joint"
        ),
    )
    p.add_argument(
        "--action-keys-csv",
        default=(
            "left_joint_1,left_joint_2,left_joint_3,left_joint_4,left_joint_5,left_joint_6,left_joint_7,left_gripper_left_joint,"
            "right_joint_1,right_joint_2,right_joint_3,right_joint_4,right_joint_5,right_joint_6,right_joint_7,right_gripper_right_joint"
        ),
    )

    # No default on purpose -- see the module docstring for the derivation recipe.
    # Running with made-up limits risks commanding the arm outside its safe range.
    p.add_argument("--joint-min-csv", required=True, help="16 comma-separated radians, --state-obs-keys-csv order")
    p.add_argument("--joint-max-csv", required=True, help="16 comma-separated radians, --state-obs-keys-csv order")
    p.add_argument(
        "--max-step-delta-csv",
        required=True,
        help="16 comma-separated per-tick rate limits (radians), --state-obs-keys-csv order",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    rclpy.init()
    node = Ros2DakshaB1SmolVLABridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
