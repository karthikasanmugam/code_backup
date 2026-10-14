#!/usr/bin/env python3
"""
ROS2 Piper bridge (4 cameras) for remote smolvla policy inference via ZeroMQ.
"""

from __future__ import annotations

import argparse
from collections import deque
import io
import time
from typing import Any

import cv2
import msgpack
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, JointState
import zmq


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str, expected_len: int, name: str) -> np.ndarray:
    vals = np.array([float(x.strip()) for x in value.split(",") if x.strip()], dtype=np.float32)
    if vals.shape[0] != expected_len:
        raise ValueError(f"{name} must have {expected_len} values, got {vals.shape[0]}: {value}")
    return vals


def recursive_add_extra_dim(obs: dict[str, Any]) -> dict[str, Any]:
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


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
            _ = self.call("ping", requires_input=False)
            return True
        except Exception:
            return False

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call(
            "get_action", {"observation": observation, "options": options}, requires_input=True
        )
        return tuple(response)


class PiperRos2RemoteBridge4Cam(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("piper_smolvla_ros2_remote_bridge_4cam")
        self.args = args
        self.policy = PolicyZmqClient(
            host=args.policy_host, port=args.policy_port, timeout_ms=args.policy_timeout_ms
        )

        if not self.policy.ping():
            raise RuntimeError(
                f"Could not reach policy server at tcp://{args.policy_host}:{args.policy_port}"
            )

        self.arm_obs_keys = parse_csv_list(args.arm_obs_keys_csv)
        self.arm_cmd_keys = parse_csv_list(args.arm_cmd_keys_csv)
        self.gripper_cmd_keys = parse_csv_list(args.gripper_cmd_keys_csv)
        self.gripper_obs_key = args.gripper_obs_key

        if len(self.arm_obs_keys) != 6:
            raise ValueError("arm_obs_keys_csv must contain exactly 6 keys")
        if len(self.arm_cmd_keys) != 6:
            raise ValueError("arm_cmd_keys_csv must contain exactly 6 keys")
        if len(self.gripper_cmd_keys) not in (1, 2):
            raise ValueError("gripper_cmd_keys_csv must contain 1 or 2 keys")

        self.arm_min = parse_csv_floats(args.arm_min_csv, 6, "arm_min_csv")
        self.arm_max = parse_csv_floats(args.arm_max_csv, 6, "arm_max_csv")
        self.max_step_delta = parse_csv_floats(args.max_step_delta_csv, 6, "max_step_delta_csv")
        self.prev_arm_cmd: np.ndarray | None = None
        self.last_policy_ok_time = 0.0

        self.latest_joint_map: dict[str, float] = {}
        self.latest_images: dict[str, np.ndarray] = {}
        self.planned_actions: deque[JointState] = deque()
        self._last_not_ready_log_time = 0.0
        self._warned_image_encodings: set[str] = set()
        self._seen_camera_frames: set[str] = set()
        self._last_decode_warn_time: dict[str, float] = {}

        self.create_subscription(
            JointState, args.joint_state_topic, self._joint_cb, qos_profile_sensor_data
        )
        self._subscribe_image_topic("world", args.world_topic)
        self._subscribe_image_topic("front_left", args.front_left_topic)
        self._subscribe_image_topic("front_right", args.front_right_topic)
        self._subscribe_image_topic("wrist", args.wrist_topic)
        self.cmd_pub = self.create_publisher(JointState, args.cmd_topic, 10)

        self.timer = self.create_timer(1.0 / args.control_hz, self._control_step)
        self.get_logger().info(
            f"Bridge started. Server=tcp://{args.policy_host}:{args.policy_port}, "
            f"control_hz={args.control_hz}, timeout_ms={args.policy_timeout_ms}"
        )

    def _joint_cb(self, msg: JointState) -> None:
        if msg.name and msg.position and len(msg.name) == len(msg.position):
            for n, p in zip(msg.name, msg.position):
                self.latest_joint_map[n] = float(p)
        elif msg.position:
            ordered = self.arm_obs_keys + [self.gripper_obs_key]
            for i, key in enumerate(ordered[: len(msg.position)]):
                self.latest_joint_map[key] = float(msg.position[i])

    def _image_cb(self, cam_key: str, msg: CompressedImage) -> None:
        if len(msg.data) == 0:
            now = time.monotonic()
            prev = self._last_decode_warn_time.get(cam_key, 0.0)
            if now - prev > 2.0:
                self._last_decode_warn_time[cam_key] = now
                self.get_logger().warn(f"CompressedImage for '{cam_key}' has empty payload.")
            return

        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            now = time.monotonic()
            prev = self._last_decode_warn_time.get(cam_key, 0.0)
            if now - prev > 2.0:
                self._last_decode_warn_time[cam_key] = now
                self.get_logger().warn(
                    f"Failed to decode CompressedImage for '{cam_key}' (format='{msg.format}')"
                )
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.latest_images[cam_key] = rgb.astype(np.uint8)
        if cam_key not in self._seen_camera_frames:
            self._seen_camera_frames.add(cam_key)
            self.get_logger().info(
                f"Received first frame for camera '{cam_key}', shape={self.latest_images[cam_key].shape}"
            )

    def _image_raw_cb(self, cam_key: str, msg: Image) -> None:
        if msg.height == 0 or msg.width == 0 or len(msg.data) == 0:
            return

        enc = (msg.encoding or "").lower()
        arr = np.frombuffer(msg.data, dtype=np.uint8)

        try:
            if enc == "rgb8":
                rgb = arr.reshape((msg.height, msg.width, 3))
            elif enc == "bgr8":
                bgr = arr.reshape((msg.height, msg.width, 3))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif enc == "mono8":
                gray = arr.reshape((msg.height, msg.width))
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            elif enc == "rgba8":
                rgba = arr.reshape((msg.height, msg.width, 4))
                rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
            elif enc == "bgra8":
                bgra = arr.reshape((msg.height, msg.width, 4))
                rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
            else:
                rgb = arr.reshape((msg.height, msg.width, 3))
                if enc not in self._warned_image_encodings:
                    self._warned_image_encodings.add(enc)
                    self.get_logger().warn(
                        f"Unsupported image encoding '{msg.encoding}' on camera '{cam_key}'. "
                        "Using generic 3-channel fallback."
                    )
        except Exception:
            if enc not in self._warned_image_encodings:
                self._warned_image_encodings.add(enc)
                self.get_logger().warn(
                    f"Failed to decode image encoding '{msg.encoding}' on camera '{cam_key}'."
                )
            return

        self.latest_images[cam_key] = rgb.astype(np.uint8)
        if cam_key not in self._seen_camera_frames:
            self._seen_camera_frames.add(cam_key)
            self.get_logger().info(
                f"Received first frame for camera '{cam_key}', shape={self.latest_images[cam_key].shape}"
            )

    def _get_topic_type(self, topic_name: str) -> str | None:
        for name, types in self.get_topic_names_and_types():
            if name == topic_name and types:
                return types[0]
        return None

    def _subscribe_image_topic(self, cam_key: str, topic_name: str) -> None:
        topic_type = self._get_topic_type(topic_name)
        if topic_type == "sensor_msgs/msg/Image":
            self.create_subscription(
                Image,
                topic_name,
                lambda m, k=cam_key: self._image_raw_cb(k, m),
                qos_profile_sensor_data,
            )
            self.get_logger().info(f"Subscribed {cam_key} as Image on {topic_name}")
            return

        if topic_type not in (None, "sensor_msgs/msg/CompressedImage"):
            self.get_logger().warn(
                f"Topic {topic_name} has type {topic_type}. Trying CompressedImage subscription."
            )
        self.create_subscription(
            CompressedImage,
            topic_name,
            lambda m, k=cam_key: self._image_cb(k, m),
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Subscribed {cam_key} as CompressedImage on {topic_name}")

    def _ready(self) -> bool:
        for key in self.arm_obs_keys + [self.gripper_obs_key]:
            if key not in self.latest_joint_map:
                return False
        for key in ("world", "front_left", "front_right", "wrist"):
            if key not in self.latest_images:
                return False
        return True

    def _log_not_ready(self) -> None:
        now = time.monotonic()
        if now - self._last_not_ready_log_time < 2.0:
            return
        self._last_not_ready_log_time = now

        required_joints = self.arm_obs_keys + [self.gripper_obs_key]
        missing_joints = [key for key in required_joints if key not in self.latest_joint_map]
        missing_cams = [key for key in ("world", "front_left", "front_right", "wrist") if key not in self.latest_images]

        if missing_joints or missing_cams:
            self.get_logger().warn(
                f"Waiting for inputs. missing_joints={missing_joints} missing_cameras={missing_cams} "
                f"seen_joint_names={list(self.latest_joint_map.keys())[:20]}"
            )

    def _hold_vec7(self) -> np.ndarray:
        arm = np.array([self.latest_joint_map[key] for key in self.arm_obs_keys], dtype=np.float32)
        grip = np.array([self.latest_joint_map[self.gripper_obs_key]], dtype=np.float32)
        return np.concatenate([arm, grip], axis=0)

    def _obs_for_policy(self) -> dict[str, Any]:
        obs = {
            "video": {
                self.args.world_camera_key: self.latest_images["world"],
                self.args.front_left_camera_key: self.latest_images["front_left"],
                self.args.front_right_camera_key: self.latest_images["front_right"],
                self.args.wrist_camera_key: self.latest_images["wrist"],
            },
            "state": {
                "single_arm": np.array(
                    [self.latest_joint_map[key] for key in self.arm_obs_keys], dtype=np.float32
                ),
                "gripper": np.array([self.latest_joint_map[self.gripper_obs_key]], dtype=np.float32),
            },
            "language": {"annotation.human.task_description": self.args.lang_instruction},
        }
        obs = recursive_add_extra_dim(obs)
        obs = recursive_add_extra_dim(obs)
        return obs

    @staticmethod
    def _extract_action_vec7(action_chunk: dict[str, Any], t: int) -> np.ndarray:
        if "single_arm" in action_chunk and "gripper" in action_chunk:
            arm = np.asarray(action_chunk["single_arm"][0][t], dtype=np.float32).reshape(-1)
            grip = np.asarray(action_chunk["gripper"][0][t], dtype=np.float32).reshape(-1)
            if arm.shape[0] != 6 or grip.shape[0] != 1:
                raise ValueError(f"Unexpected action dims: arm={arm.shape}, gripper={grip.shape}")
            return np.concatenate([arm, grip], axis=0)
        if "action.single_arm" in action_chunk and "action.gripper" in action_chunk:
            arm = np.asarray(action_chunk["action.single_arm"][0][t], dtype=np.float32).reshape(-1)
            grip = np.asarray(action_chunk["action.gripper"][0][t], dtype=np.float32).reshape(-1)
            if arm.shape[0] != 6 or grip.shape[0] != 1:
                raise ValueError(f"Unexpected action dims: arm={arm.shape}, gripper={grip.shape}")
            return np.concatenate([arm, grip], axis=0)
        if "action" in action_chunk:
            vec = np.asarray(action_chunk["action"][0][t], dtype=np.float32).reshape(-1)
            if vec.shape[0] != 7:
                raise ValueError(f"Expected action[7], got {vec.shape[0]}")
            return vec
        raise ValueError(f"Unsupported action chunk keys: {list(action_chunk.keys())}")

    def _apply_safety(self, target_vec7: np.ndarray, hold_vec7: np.ndarray) -> tuple[np.ndarray, float]:
        target_arm = np.clip(target_vec7[:6], self.arm_min, self.arm_max)
        if self.prev_arm_cmd is None:
            self.prev_arm_cmd = hold_vec7[:6].copy()
        arm_delta = np.clip(target_arm - self.prev_arm_cmd, -self.max_step_delta, self.max_step_delta)
        arm_cmd = np.clip(self.prev_arm_cmd + arm_delta, self.arm_min, self.arm_max)

        grip = float(target_vec7[6] * self.args.gripper_scale + self.args.gripper_offset)
        grip = float(np.clip(grip, self.args.gripper_min, self.args.gripper_max))
        self.prev_arm_cmd = arm_cmd.copy()
        return arm_cmd, grip

    def _to_joint_state_msg(self, arm_cmd: np.ndarray, grip_cmd: float) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        names = list(self.arm_cmd_keys)
        pos = [float(x) for x in arm_cmd]
        if len(self.gripper_cmd_keys) == 1:
            names.append(self.gripper_cmd_keys[0])
            pos.append(float(grip_cmd))
        else:
            names.append(self.gripper_cmd_keys[0])
            pos.append(float(grip_cmd))
            names.append(self.gripper_cmd_keys[1])
            pos.append(float(self.args.second_gripper_mirror_sign * grip_cmd))

        msg.name = names
        msg.position = pos
        return msg

    def _plan_from_policy(self) -> None:
        hold_vec7 = self._hold_vec7()
        obs = self._obs_for_policy()
        action_chunk, _ = self.policy.get_action(obs)
        self.last_policy_ok_time = time.monotonic()

        any_key = next(iter(action_chunk.keys()))
        horizon = int(np.asarray(action_chunk[any_key]).shape[1])
        steps = min(horizon, self.args.action_horizon)

        for t in range(steps):
            vec7 = self._extract_action_vec7(action_chunk, t)
            arm_cmd, grip_cmd = self._apply_safety(vec7, hold_vec7)
            self.planned_actions.append(self._to_joint_state_msg(arm_cmd, grip_cmd))

    def _publish_hold(self) -> None:
        hold_vec7 = self._hold_vec7()
        msg = self._to_joint_state_msg(hold_vec7[:6], float(hold_vec7[6]))
        self.cmd_pub.publish(msg)

    def _control_step(self) -> None:
        if not self._ready():
            self._log_not_ready()
            return

        stale = (
            self.last_policy_ok_time > 0
            and (time.monotonic() - self.last_policy_ok_time) > self.args.watchdog_timeout_s
        )
        if stale:
            self.planned_actions.clear()
            self._publish_hold()
            return

        if not self.planned_actions:
            try:
                self._plan_from_policy()
            except Exception as exc:
                self.get_logger().error(f"Policy planning failed: {exc}")
                self.planned_actions.clear()
                self._publish_hold()
                return

        if self.planned_actions:
            self.cmd_pub.publish(self.planned_actions.popleft())
        else:
            self._publish_hold()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Piper ROS2 4-cam remote bridge for smolvla")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--policy-timeout-ms", type=int, default=60000)
    parser.add_argument("--lang-instruction", default="task1")
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--watchdog-timeout-s", type=float, default=1.0)

    parser.add_argument("--joint-state-topic", default="/joint_states_right")
    parser.add_argument("--cmd-topic", default="/joint_ctrl_cmd_right")
    parser.add_argument("--world-topic", default="/world/camera/image_raw/compressed")
    parser.add_argument("--front-left-topic", default="/front/left/image_raw/compressed")
    parser.add_argument("--front-right-topic", default="/front/right/image_raw/compressed")
    parser.add_argument("--wrist-topic", default="/waveshare/cam1/image_raw/compressed")

    parser.add_argument("--world-camera-key", default="world")
    parser.add_argument("--front-left-camera-key", default="front_left")
    parser.add_argument("--front-right-camera-key", default="front_right")
    parser.add_argument("--wrist-camera-key", default="wrist")

    parser.add_argument("--arm-obs-keys-csv", default="joint1,joint2,joint3,joint4,joint5,joint6")
    parser.add_argument("--gripper-obs-key", default="gripper")
    parser.add_argument("--arm-cmd-keys-csv", default="joint1,joint2,joint3,joint4,joint5,joint6")
    parser.add_argument("--gripper-cmd-keys-csv", default="gripper")
    parser.add_argument("--second-gripper-mirror-sign", type=float, default=-1.0)

    parser.add_argument(
        "--arm-min-csv",
        default="-0.7161285,-0.03820236,-1.5631742,-1.007391,-0.7551507,-0.8011157",
    )
    parser.add_argument(
        "--arm-max-csv",
        default="0.8281539,2.1425767,0.04277269,0.83614326,1.1717832,2.2785003",
    )
    parser.add_argument("--max-step-delta-csv", default="0.03,0.03,0.03,0.03,0.03,0.03")
    parser.add_argument("--gripper-min", type=float, default=-0.0037)
    parser.add_argument("--gripper-max", type=float, default=0.1030)
    parser.add_argument("--gripper-scale", type=float, default=1.0)
    parser.add_argument("--gripper-offset", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rclpy.init()
    node = PiperRos2RemoteBridge4Cam(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
