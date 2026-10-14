# #!/usr/bin/env python3
# """
# ROS2 Daksha B1 bridge for GR00T policy server.

# Checkpoint: output/pick_place_blue/checkpoint-70000
# Dataset:    /home/s1/Desktop/datasets/pick_place_blue  (134 eps, 30 fps)
# Embodiment: daksha_b1

# Every default in this file is derived from that checkpoint's baked configs
# (processor_config.json, experiment_cfg/dataset_statistics.json) and the
# dataset meta (meta/tasks.jsonl, meta/info.json, ros2_topics.json), so
# inference runs in exactly the space the model was trained on.

# Policy state/action layout (16 DoF), right-first:
#   right_arm(7) + right_gripper(1) + left_arm(7) + left_gripper(1)

# Video modality keys (must match checkpoint processor_config.json ->
# modality_configs -> daksha_b1 -> video -> modality_keys):
#   zed_right, zed_left, wrist_right, wrist_left

# Action horizon: 16 (the checkpoint's 16 action delta_indices).
# Control rate: 30 Hz (dataset meta/info.json fps).

# Note on action representation: the checkpoint trains the two arms as
# RELATIVE actions and the two grippers as ABSOLUTE (processor
# use_relative_action=True). The policy server already converts relative ->
# absolute using the state we send, so the actions returned here are
# ABSOLUTE joint positions and are clamped as such below.

# Start the server first:
#   python gr00t/eval/run_gr00t_server.py \
#       --model-path output/pick_place_blue/checkpoint-70000 \
#       --embodiment-tag daksha_b1

# Commands are published as sensor_msgs/JointState on --cmd-topic (/joint_cmd).
# """

# from __future__ import annotations

# import argparse
# from collections import deque
# import concurrent.futures
# import math
# import time
# from typing import Any

# import cv2
# from gr00t.policy.server_client import PolicyClient
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import qos_profile_sensor_data
# from sensor_msgs.msg import CompressedImage, Image, JointState


# def parse_csv_list(value: str) -> list[str]:
#     return [x.strip() for x in value.split(",") if x.strip()]


# def parse_csv_floats(value: str, expected_len: int, name: str) -> np.ndarray:
#     vals = np.array([float(x.strip()) for x in value.split(",") if x.strip()], dtype=np.float32)
#     if vals.shape[0] != expected_len:
#         raise ValueError(f"{name} must have {expected_len} values, got {vals.shape[0]}: {value}")
#     return vals


# def joint_name_aliases(name: str) -> set[str]:
#     aliases = {name}
#     if "/" in name:
#         aliases.add(name.split("/", 1)[1])
#     elif name.startswith("daksha_right_"):
#         aliases.add(f"right/{name}")
#     elif name.startswith("daksha_left_"):
#         aliases.add(f"left/{name}")
#     return aliases


# def recursive_add_extra_dim(obs: dict[str, Any]) -> dict[str, Any]:
#     for key, val in obs.items():
#         if isinstance(val, np.ndarray):
#             obs[key] = val[np.newaxis, ...]
#         elif isinstance(val, dict):
#             obs[key] = recursive_add_extra_dim(val)
#         else:
#             obs[key] = [val]
#     return obs


# class Gr00TDakshaB1RosBridge(Node):
#     def __init__(self, args: argparse.Namespace):
#         super().__init__("gr00t_daksha_b1_ros2_bridge")
#         self.args = args

#         self.policy = PolicyClient(host=args.policy_host, port=args.policy_port, strict=False)
#         if not self.policy.ping():
#             raise RuntimeError(
#                 f"Could not reach GR00T server at tcp://{args.policy_host}:{args.policy_port}"
#             )

#         self.state_obs_keys = parse_csv_list(args.state_obs_keys_csv)
#         self.action_keys = parse_csv_list(args.action_keys_csv)

#         if len(self.state_obs_keys) != 16:
#             raise ValueError("state_obs_keys_csv must contain exactly 16 keys")
#         if len(self.action_keys) != 16:
#             raise ValueError("action_keys_csv must contain exactly 16 keys")

#         # Guard against the left/right swap regression: the slice indices below and the
#         # action concatenation assume RIGHT-first ordering, and the checkpoint's
#         # `right_arm` group is the physical right arm. If the key CSVs are ever reordered
#         # (e.g. to left-first to mirror the dataset's flat layout) without also updating
#         # the indices, the arms and grippers get silently swapped and the gripper never
#         # actuates. Fail loudly instead.
#         for name, keys in (
#             ("state_obs_keys", self.state_obs_keys),
#             ("action_keys", self.action_keys),
#         ):
#             expected_side = ["right"] * 8 + ["left"] * 8
#             for i, (key, side) in enumerate(zip(keys, expected_side)):
#                 if side not in key.lower():
#                     raise ValueError(
#                         f"{name}[{i}]={key!r} must be a '{side}' joint. Keys must be "
#                         f"right-first: right_arm(0..6), right_gripper(7), left_arm(8..14), "
#                         f"left_gripper(15). Reordering breaks the state/action mapping."
#                     )
#             for i in (7, 15):
#                 if "gripper" not in keys[i].lower():
#                     raise ValueError(f"{name}[{i}]={keys[i]!r} must be a gripper joint.")

#         if not 0 <= args.replan_at < args.action_horizon:
#             raise ValueError(
#                 f"--replan-at must be in [0, action_horizon), got {args.replan_at} "
#                 f"with --action-horizon {args.action_horizon}"
#             )
#         # Wall-clock the policy has to answer before the queue runs dry and the arm
#         # falls back to hold. Reported next to the measured latency each replan.
#         self.replan_budget_s = args.replan_at / args.control_hz

#         self.joint_min = parse_csv_floats(args.joint_min_csv, 16, "joint_min_csv")
#         self.joint_max = parse_csv_floats(args.joint_max_csv, 16, "joint_max_csv")
#         self.max_step_delta = parse_csv_floats(args.max_step_delta_csv, 16, "max_step_delta_csv")
#         self.prev_cmd: np.ndarray | None = None
#         self.last_policy_ok_time = 0.0

#         # Policy state/action layout is RIGHT-first (matches the checkpoint's modality
#         # groups; the key CSVs above are validated to follow this order):
#         # right arm(0..6), right gripper(7), left arm(8..14), left gripper(15).
#         self.policy_right_arm_indices = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
#         self.policy_right_gripper_index = np.array([7], dtype=np.int64)
#         self.policy_left_arm_indices = np.array([8, 9, 10, 11, 12, 13, 14], dtype=np.int64)
#         self.policy_left_gripper_index = np.array([15], dtype=np.int64)
#         self.policy_control_indices = np.arange(16, dtype=np.int64)

#         # Outgoing /joint_cmd ordering. The internal layout is right-first (the
#         # checkpoint's modality order), but every other publisher on this robot's
#         # /joint_cmd -- leader_mirror, home_move_service, ros2_topic_replay -- sends
#         # LEFT-first, which is also the dataset's flat layout (meta/info.json names).
#         # sensor_msgs/JointState carries `name` alongside `position`, so a name-driven
#         # consumer accepts either order; an index-driven one accepts only the
#         # convention already on the wire. Publishing left-first is therefore correct
#         # under both, so it is the default. Names and positions are permuted together
#         # by the same index array, so the pairing can never come apart.
#         if args.publish_joint_order == "left_first":
#             self.publish_indices = np.array(list(range(8, 16)) + list(range(0, 8)), dtype=np.int64)
#         else:
#             self.publish_indices = np.arange(16, dtype=np.int64)

#         # Internal camera keys, in the same order as the checkpoint's video
#         # modality_keys: zed_right, zed_left, wrist_right, wrist_left.
#         self.camera_keys = ("zed_right", "zed_left", "wrist_right", "wrist_left")

#         # Receipt timestamps, so a publisher that dies is detected instead of the
#         # bridge silently re-sending the last frame it ever saw. Without this the
#         # policy keeps scoring a frozen image, predicts "no motion" every chunk, and
#         # the arm sits still with no error anywhere -- indistinguishable from a bad
#         # checkpoint. Cameras dropping out is exactly the observed failure.
#         self.image_stamps: dict[str, float] = {}
#         self.joint_stamps: dict[str, float] = {}

#         self.latest_joint_map: dict[str, float] = {}
#         self.latest_joint_names: list[str] = []
#         self.latest_images: dict[str, np.ndarray] = {}
#         self.planned_actions: deque[JointState] = deque()
#         self.pending_plan_future: concurrent.futures.Future[list[JointState]] | None = None
#         self.pending_plan_mode: str | None = None
#         self.shape_log_done = False
#         self.plan_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

#         self.valid_modes = ["hold", "full"]
#         if args.run_validation_suite:
#             self.phase_modes = list(self.valid_modes)
#         else:
#             if args.validation_mode not in self.valid_modes:
#                 raise ValueError(f"validation_mode must be one of {self.valid_modes}")
#             self.phase_modes = [args.validation_mode]
#         self.phase_idx = 0
#         self.phase_start_t = time.monotonic()
#         self.finished = False
#         self.get_logger().info(f"Validation phases: {self.phase_modes}")

#         if args.right_joint_state_topic or args.left_joint_state_topic:
#             if args.right_joint_state_topic:
#                 self.create_subscription(
#                     JointState,
#                     args.right_joint_state_topic,
#                     lambda m: self._joint_cb(m, side="right"),
#                     qos_profile_sensor_data,
#                 )
#             if args.left_joint_state_topic:
#                 self.create_subscription(
#                     JointState,
#                     args.left_joint_state_topic,
#                     lambda m: self._joint_cb(m, side="left"),
#                     qos_profile_sensor_data,
#                 )
#         else:
#             self.create_subscription(
#                 JointState, args.joint_state_topic, self._joint_cb, qos_profile_sensor_data
#             )
#         # The dataset was recorded from the "<topic>/compressed" variants
#         # (ros2_topics.json). Keep the raw base topics as the defaults and append
#         # the "/compressed" suffix automatically when compressed transport is
#         # selected, so both modes subscribe to the topics the data came from.
#         if args.image_transport == "compressed":
#             for attr in (
#                 "zed_right_topic",
#                 "zed_left_topic",
#                 "right_wrist_topic",
#                 "left_wrist_topic",
#             ):
#                 topic = getattr(args, attr)
#                 if not topic.endswith("/compressed"):
#                     setattr(args, attr, f"{topic}/compressed")

#         image_msg_type = CompressedImage if args.image_transport == "compressed" else Image
#         self.create_subscription(
#             image_msg_type,
#             args.zed_right_topic,
#             lambda m: self._image_cb("zed_right", m),
#             10,
#         )
#         self.create_subscription(
#             image_msg_type,
#             args.zed_left_topic,
#             lambda m: self._image_cb("zed_left", m),
#             10,
#         )
#         self.create_subscription(
#             image_msg_type,
#             args.right_wrist_topic,
#             lambda m: self._image_cb("wrist_right", m),
#             10,
#         )
#         self.create_subscription(
#             image_msg_type,
#             args.left_wrist_topic,
#             lambda m: self._image_cb("wrist_left", m),
#             10,
#         )
#         # Inference publishes a single JointState command topic even if the
#         # dataset was originally recorded from separate arm controller topics.
#         self.cmd_pub = self.create_publisher(JointState, args.cmd_topic, 10)

#         self.timer = self.create_timer(1.0 / args.control_hz, self._control_step)
#         self.get_logger().info(
#             f"Bridge started. Server=tcp://{args.policy_host}:{args.policy_port}, "
#             f"cmd_topic={args.cmd_topic}, image_transport={args.image_transport}"
#         )

#     def _joint_cb(self, msg: JointState, side: str | None = None) -> None:
#         if msg.name and msg.position and len(msg.name) == len(msg.position):
#             if not self.latest_joint_names:
#                 self.latest_joint_names = list(msg.name)
#                 if self.latest_joint_names != self.action_keys:
#                     self.get_logger().info(
#                         "Incoming joint names differ from action_keys. The bridge will match aliases "
#                         "like daksha_right_joint1 and right/daksha_right_joint1."
#                     )
#             now = time.monotonic()
#             for n, p in zip(msg.name, msg.position):
#                 for alias in joint_name_aliases(n):
#                     self.latest_joint_map[alias] = float(p)
#                     self.joint_stamps[alias] = now
#         elif msg.position:
#             if side == "right":
#                 keys = self.state_obs_keys[:8]
#             elif side == "left":
#                 keys = self.state_obs_keys[8:16]
#             else:
#                 keys = self.state_obs_keys
#             now = time.monotonic()
#             for i, key in enumerate(keys[: len(msg.position)]):
#                 self.latest_joint_map[key] = float(msg.position[i])
#                 self.joint_stamps[key] = now

#     def _maybe_downscale(self, img: np.ndarray) -> np.ndarray:
#         """Cap the image's longest edge at --image-max-edge (0 disables)."""
#         limit = self.args.image_max_edge
#         if limit <= 0:
#             return img
#         h, w = img.shape[:2]
#         longest = max(h, w)
#         if longest <= limit:
#             return img
#         scale = limit / float(longest)
#         # INTER_AREA is the correct kernel for downsampling; anything else aliases.
#         return cv2.resize(
#             img,
#             (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
#             interpolation=cv2.INTER_AREA,
#         )

#     def _image_cb(self, cam_key: str, msg: Image | CompressedImage) -> None:
#         if isinstance(msg, CompressedImage):
#             try:
#                 compressed = np.frombuffer(msg.data, dtype=np.uint8)
#                 if len(compressed) == 0:
#                     self.get_logger().warn(f"Received empty compressed image data from {cam_key}")
#                     return
#                 img = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
#                 if img is None:
#                     self.get_logger().warn(f"Failed to decode compressed image from {cam_key}")
#                     return
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 self.latest_images[cam_key] = self._maybe_downscale(img)
#                 self.image_stamps[cam_key] = time.monotonic()
#             except Exception as e:
#                 self.get_logger().warn(f"Error processing compressed image from {cam_key}: {e}")
#             return

#         if msg.encoding not in ("rgb8", "bgr8", "bgra8", "rgba8"):
#             self.get_logger().warn(f"Unsupported raw image encoding for {cam_key}: {msg.encoding}")
#             return
#         channels = 4 if msg.encoding in ("bgra8", "rgba8") else 3
#         img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, channels))
#         if msg.encoding == "bgr8":
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         elif msg.encoding == "bgra8":
#             img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
#         elif msg.encoding == "rgba8":
#             img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
#         self.latest_images[cam_key] = self._maybe_downscale(img)
#         self.image_stamps[cam_key] = time.monotonic()

#     def _current_mode(self) -> str | None:
#         if self.finished:
#             return None
#         mode = self.phase_modes[self.phase_idx]
#         if self.args.run_validation_suite:
#             elapsed = time.monotonic() - self.phase_start_t
#             if elapsed >= self.args.phase_seconds:
#                 self.get_logger().info(f"Completed phase: {mode}")
#                 self.phase_idx += 1
#                 if self.phase_idx >= len(self.phase_modes):
#                     self.get_logger().info("Validation suite complete.")
#                     self.finished = True
#                     return None
#                 self.phase_start_t = time.monotonic()
#                 mode = self.phase_modes[self.phase_idx]
#                 self.planned_actions.clear()
#                 self.get_logger().info(f"Starting phase: {mode}")
#         return mode

#     def _ready(self) -> bool:
#         missing_joints, missing_cameras = self._missing_inputs()
#         return not missing_joints and not missing_cameras

#     def _missing_inputs(self) -> tuple[list[str], list[str]]:
#         """Inputs that never arrived, or whose last message is older than the timeout.

#         A key that went stale is reported as "<key> (stale Ns)" so a publisher that
#         died mid-run is called out by name rather than silently reused forever.
#         """
#         now = time.monotonic()
#         timeout = self.args.input_timeout_s

#         def stale(key: str, stamps: dict[str, float]) -> str | None:
#             if key not in stamps:
#                 return key
#             age = now - stamps[key]
#             return f"{key} (stale {age:.1f}s)" if timeout > 0 and age > timeout else None

#         missing_joints = [
#             s for s in (stale(k, self.joint_stamps) for k in self.state_obs_keys) if s
#         ]
#         missing_cameras = [s for s in (stale(k, self.image_stamps) for k in self.camera_keys) if s]
#         return missing_joints, missing_cameras

#     def _log_waiting_inputs(self) -> None:
#         missing_joints, missing_cameras = self._missing_inputs()
#         parts: list[str] = []
#         if missing_joints:
#             if self.args.right_joint_state_topic or self.args.left_joint_state_topic:
#                 topic_desc = (
#                     f"right={self.args.right_joint_state_topic or 'disabled'}, "
#                     f"left={self.args.left_joint_state_topic or 'disabled'}"
#                 )
#             else:
#                 topic_desc = self.args.joint_state_topic
#             parts.append(f"waiting on {topic_desc} for joint keys: {', '.join(missing_joints)}")
#         if missing_cameras:
#             camera_topics = {
#                 "zed_right": self.args.zed_right_topic,
#                 "zed_left": self.args.zed_left_topic,
#                 "wrist_right": self.args.right_wrist_topic,
#                 "wrist_left": self.args.left_wrist_topic,
#             }
#             missing_cam_desc = ", ".join(f"{k} ({camera_topics[k]})" for k in missing_cameras)
#             parts.append(f"waiting on camera topics: {missing_cam_desc}")
#         if not parts:
#             parts.append("all required ROS2 inputs received, but policy is not ready yet")
#         self.get_logger().info("Waiting for inputs -> " + " | ".join(parts))

#     def _hold_vec16(self) -> np.ndarray:
#         return np.array([self.latest_joint_map[k] for k in self.state_obs_keys], dtype=np.float32)

#     def _log_obs_shapes(self, obs: dict[str, Any]) -> None:
#         video_shapes = {k: tuple(v.shape) for k, v in obs["video"].items()}
#         state_shapes = {k: tuple(v.shape) for k, v in obs["state"].items()}
#         language_shapes = {k: (len(v), len(v[0]) if v else 0) for k, v in obs["language"].items()}
#         self.get_logger().info(
#             f"Policy obs shapes: video={video_shapes}, state={state_shapes}, language={language_shapes}"
#         )

#     def _obs_for_policy(self, state16: np.ndarray | None = None) -> dict[str, Any]:
#         # Accept a precomputed state so the caller can guarantee the rate limiter is
#         # anchored to the exact same vector the policy conditions on.
#         if state16 is None:
#             state16 = self._hold_vec16()
#         obs = {
#             "video": {
#                 self.args.zed_right_camera_key: self.latest_images["zed_right"],
#                 self.args.zed_left_camera_key: self.latest_images["zed_left"],
#                 self.args.right_wrist_camera_key: self.latest_images["wrist_right"],
#                 self.args.left_wrist_camera_key: self.latest_images["wrist_left"],
#             },
#             "state": {
#                 "right_arm": state16[self.policy_right_arm_indices],
#                 "right_gripper": state16[[7]],
#                 "left_arm": state16[self.policy_left_arm_indices],
#                 "left_gripper": state16[[15]],
#             },
#             "language": {"annotation.human.task_description": self.args.lang_instruction},
#         }
#         obs = recursive_add_extra_dim(obs)
#         obs = recursive_add_extra_dim(obs)
#         if not self.shape_log_done:
#             self._log_obs_shapes(obs)
#             self.shape_log_done = True
#         return obs

#     @staticmethod
#     def _extract_action_vec16(action_chunk: dict[str, Any], t: int) -> np.ndarray:
#         if (
#             "right_arm" in action_chunk
#             and "right_gripper" in action_chunk
#             and "left_arm" in action_chunk
#             and "left_gripper" in action_chunk
#         ):
#             right = np.asarray(action_chunk["right_arm"][0][t], dtype=np.float32).reshape(-1)
#             right_gripper = np.asarray(
#                 action_chunk["right_gripper"][0][t], dtype=np.float32
#             ).reshape(-1)
#             left = np.asarray(action_chunk["left_arm"][0][t], dtype=np.float32).reshape(-1)
#             left_gripper = np.asarray(action_chunk["left_gripper"][0][t], dtype=np.float32).reshape(
#                 -1
#             )
#             if (
#                 right.shape[0] != 7
#                 or right_gripper.shape[0] != 1
#                 or left.shape[0] != 7
#                 or left_gripper.shape[0] != 1
#             ):
#                 raise ValueError(
#                     "Unexpected action dims: "
#                     f"right_arm={right.shape}, right_gripper={right_gripper.shape}, "
#                     f"left_arm={left.shape}, left_gripper={left_gripper.shape}"
#                 )
#             return np.concatenate([right, right_gripper, left, left_gripper], axis=0)
#         if (
#             "action.right_arm" in action_chunk
#             and "action.right_gripper" in action_chunk
#             and "action.left_arm" in action_chunk
#             and "action.left_gripper" in action_chunk
#         ):
#             right = np.asarray(action_chunk["action.right_arm"][0][t], dtype=np.float32).reshape(-1)
#             right_gripper = np.asarray(
#                 action_chunk["action.right_gripper"][0][t], dtype=np.float32
#             ).reshape(-1)
#             left = np.asarray(action_chunk["action.left_arm"][0][t], dtype=np.float32).reshape(-1)
#             left_gripper = np.asarray(
#                 action_chunk["action.left_gripper"][0][t], dtype=np.float32
#             ).reshape(-1)
#             if (
#                 right.shape[0] != 7
#                 or right_gripper.shape[0] != 1
#                 or left.shape[0] != 7
#                 or left_gripper.shape[0] != 1
#             ):
#                 raise ValueError(
#                     "Unexpected action dims: "
#                     f"right_arm={right.shape}, right_gripper={right_gripper.shape}, "
#                     f"left_arm={left.shape}, left_gripper={left_gripper.shape}"
#                 )
#             return np.concatenate([right, right_gripper, left, left_gripper], axis=0)
#         if "action" in action_chunk:
#             vec = np.asarray(action_chunk["action"][0][t], dtype=np.float32).reshape(-1)
#             if vec.shape[0] != 16:
#                 raise ValueError(f"Expected action[16], got {vec.shape[0]}")
#             return vec
#         raise ValueError(f"Unsupported action chunk keys: {list(action_chunk.keys())}")

#     def _apply_safety(
#         self, action_vec16: np.ndarray, hold_vec16: np.ndarray, mode: str
#     ) -> np.ndarray:
#         # `prev_cmd` chains within a chunk (step t rate-limits against step t-1) and is
#         # re-anchored to the measured state at the start of each chunk by
#         # `_plan_from_policy`. Both happen on the planner thread, so no locking.
#         if self.prev_cmd is None:
#             self.prev_cmd = hold_vec16.copy()

#         prev_control = self.prev_cmd[self.policy_control_indices]
#         target = np.clip(action_vec16, self.joint_min, self.joint_max)
#         delta = np.clip(target - prev_control, -self.max_step_delta, self.max_step_delta)
#         cmd_control = np.clip(prev_control + delta, self.joint_min, self.joint_max)

#         cmd = hold_vec16.copy()
#         cmd[self.policy_control_indices] = cmd_control
#         if mode == "hold":
#             cmd = hold_vec16.copy()
#         self.prev_cmd = cmd.copy()
#         return cmd

#     def _to_joint_state_msg(self, cmd16: np.ndarray) -> JointState:
#         msg = JointState()
#         msg.header.stamp = self.get_clock().now().to_msg()
#         # Permute names and positions together so each value keeps its own name.
#         msg.name = [self.action_keys[i] for i in self.publish_indices]
#         msg.position = [float(cmd16[i]) for i in self.publish_indices]
#         # Many hardware controllers require velocity / effort fields to be
#         # present (even if zero) to accept the JointState as a valid command.
#         msg.velocity = [0.0] * len(cmd16)
#         msg.effort = [0.0] * len(cmd16)
#         return msg

#     def _plan_from_policy(self, mode: str) -> list[JointState]:
#         hold_vec16 = self._hold_vec16()
#         obs = self._obs_for_policy(hold_vec16)
#         t_start = time.monotonic()
#         action_chunk, _ = self.policy.get_action(obs)
#         inference_s = time.monotonic() - t_start
#         self.last_policy_ok_time = time.monotonic()

#         # Tuning signal for --replan-at: the round trip must fit in the budget, else
#         # the queue drains before the next chunk lands and the arm hold-stutters.
#         # With replanning disabled (--replan-at 0) there is no budget -- the arm simply
#         # holds for the whole round trip by design, so report the cost of that pause.
#         if self.args.replan_at > 0:
#             budget_s = self.replan_budget_s
#             note = ""
#             if inference_s > budget_s:
#                 note = (
#                     f"  OVER BUDGET -- raise --replan-at to "
#                     f"{math.ceil(inference_s * self.args.control_hz)}"
#                 )
#             self.get_logger().info(
#                 f"inference {inference_s * 1000:.0f} ms vs budget {budget_s * 1000:.0f} ms{note}"
#             )
#         else:
#             self.get_logger().info(
#                 f"inference {inference_s * 1000:.0f} ms "
#                 f"(~{inference_s * self.args.control_hz:.1f} ticks held between chunks; "
#                 f"replanning disabled)"
#             )

#         any_key = next(iter(action_chunk.keys()))
#         horizon = int(np.asarray(action_chunk[any_key]).shape[1])
#         steps = min(horizon, self.args.action_horizon)
#         action_shapes = {k: tuple(np.asarray(v).shape) for k, v in action_chunk.items()}
#         self.get_logger().info(
#             f"Policy action keys/shapes: {action_shapes}, horizon={horizon}, steps={steps}, mode={mode}"
#         )

#         # Re-anchor the rate limiter to the state this chunk was predicted from.
#         # The checkpoint emits the arms as RELATIVE actions, so the server returns
#         # absolute targets computed as (hold_vec16 + predicted delta) -- the chunk is
#         # meaningful only relative to hold_vec16. Carrying `prev_cmd` over from the
#         # previous chunk would rate-limit against a reference the model never saw:
#         # `_publish_hold` streams the measured state between chunks without touching
#         # `prev_cmd`, so the two drift apart and the +/-max_step_delta clamp then eats
#         # the leading steps of every chunk and distorts the trajectory.
#         self.prev_cmd = hold_vec16.copy()

#         planned: list[JointState] = []
#         for t in range(steps):
#             vec16 = self._extract_action_vec16(action_chunk, t)
#             cmd16 = self._apply_safety(vec16, hold_vec16, mode)
#             planned.append(self._to_joint_state_msg(cmd16))
#         return planned

#     def _publish_hold(self) -> None:
#         hold_vec16 = self._hold_vec16()
#         self.cmd_pub.publish(self._to_joint_state_msg(hold_vec16))

#     def _control_step(self) -> None:
#         mode = self._current_mode()
#         if mode is None:
#             self.pending_plan_future = None
#             self.pending_plan_mode = None
#             if self._ready():
#                 self._publish_hold()
#             return
#         if not self._ready():
#             self._log_waiting_inputs()
#             return

#         stale = (
#             self.last_policy_ok_time > 0
#             and (time.monotonic() - self.last_policy_ok_time) > self.args.watchdog_timeout_s
#         )
#         if mode == "hold" or stale:
#             self.pending_plan_future = None
#             self.pending_plan_mode = None
#             self.planned_actions.clear()
#             self._publish_hold()
#             return

#         if self.pending_plan_future is not None and self.pending_plan_future.done():
#             try:
#                 planned = self.pending_plan_future.result()
#                 # REPLACE, don't append. The fresh chunk is anchored to the state
#                 # sampled at its own plan time, so any steps still queued from the
#                 # previous chunk are stale targets relative to it.
#                 self.planned_actions.clear()
#                 self.planned_actions.extend(planned)
#             except Exception as exc:
#                 # Keep whatever is still queued: those steps are only slightly stale
#                 # and finishing them beats stuttering to hold. A persistent failure is
#                 # caught by the watchdog above once last_policy_ok_time goes stale.
#                 self.get_logger().error(f"Policy planning failed: {exc}")
#             finally:
#                 self.pending_plan_future = None
#                 self.pending_plan_mode = None

#         # Kick off the next chunk once the queue is down to --replan-at steps, so the
#         # round trip overlaps with executing the tail of the current chunk instead of
#         # stalling the arm. replan_at=0 means "drain fully, then replan".
#         if self.pending_plan_future is None and len(self.planned_actions) <= self.args.replan_at:
#             try:
#                 self.pending_plan_mode = mode
#                 self.pending_plan_future = self.plan_executor.submit(self._plan_from_policy, mode)
#             except Exception as exc:
#                 self.get_logger().error(f"Could not start policy planning: {exc}")
#                 self.pending_plan_future = None
#                 self.pending_plan_mode = None

#         # Keep executing the queue while inference is in flight; only hold if it is dry.
#         if self.planned_actions:
#             self.cmd_pub.publish(self.planned_actions.popleft())
#         else:
#             self._publish_hold()

#     def destroy_node(self) -> bool:
#         self.plan_executor.shutdown(wait=False, cancel_futures=True)
#         return super().destroy_node()


# def build_arg_parser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser("Daksha B1 ROS2 bridge for GR00T policy server")

#     p.add_argument("--policy-host", default="127.0.0.1")
#     p.add_argument("--policy-port", type=int, default=5555)
#     # Must match the training task string exactly. From the dataset's
#     # meta/tasks.jsonl (task_index 0), which is the only task in this dataset.
#     p.add_argument("--lang-instruction", default="Pick the blue box from the table.")
#     # Action horizon must match the number of delta_indices in the checkpoint
#     # (processor_config.json -> modality_configs -> daksha_b1 -> action -> delta_indices).
#     # For checkpoint-70000 this is 16.
#     p.add_argument("--action-horizon", type=int, default=16)
#     # Must match the dataset recording rate (meta/info.json -> fps = 30), otherwise
#     # the 16-step chunk is played back at the wrong speed.
#     p.add_argument("--control-hz", type=float, default=30.0)
#     # Start the next chunk once only this many planned steps remain, so the policy
#     # round trip overlaps with executing the current chunk's tail rather than stalling
#     # the arm at hold. This checkpoint predicts the arms as RELATIVE actions, so a
#     # chunk's displacement accumulates toward its END -- replanning too early throws
#     # away the part of the chunk that carries the motion, and the arm looks sluggish.
#     # Set it to ceil(measured_inference_s * control_hz); read the measurement off the
#     # "inference N ms vs budget M ms" log line. 0 = drain the chunk fully, then replan
#     # (the arm holds during every round trip).
#     p.add_argument("--replan-at", type=int, default=0)
#     p.add_argument("--watchdog-timeout-s", type=float, default=2.0)
#     # Treat a camera/joint input as unusable once its last message is this old. Guards
#     # against a dead publisher: without it the bridge keeps feeding the policy the last
#     # frame forever and the arm sits still with nothing logged. 0 disables the check.
#     p.add_argument("--input-timeout-s", type=float, default=1.0)

#     p.add_argument("--validation-mode", choices=["hold", "full"], default="full")
#     p.add_argument("--run-validation-suite", action="store_true")
#     p.add_argument("--phase-seconds", type=float, default=12.0)

#     p.add_argument("--joint-state-topic", default="/joint_states")
#     p.add_argument("--right-joint-state-topic", default="/RightArmSystem_ordered_joint_states")
#     p.add_argument("--left-joint-state-topic", default="/LeftArmSystem_ordered_joint_states")
#     # /joint_cmd is the real command entry point on this robot: joint_command_limiter
#     # subscribes to it (sensor_msgs/JointState) and fans out to the arm and gripper
#     # *_mit_controller/joint_trajectory topics. The follower_cmd_topic entries in the
#     # dataset's ros2_topics.json are those downstream controller topics, not the input.
#     p.add_argument("--cmd-topic", default="/joint_cmd")
#     # Order of `name`/`position` in the published JointState. See publish_indices.
#     p.add_argument(
#         "--publish-joint-order", choices=["left_first", "right_first"], default="left_first"
#     )
#     p.add_argument("--zed-right-topic", default="/zed/zed_node/right/color/rect/image")
#     p.add_argument("--zed-left-topic", default="/zed/zed_node/left/color/rect/image")
#     # Topic defaults are the base (raw) topics behind the dataset's recorded
#     # camera_topics in ros2_topics.json; "/compressed" is appended automatically
#     # when --image-transport compressed is used.
#     p.add_argument("--right-wrist-topic", default="/right/camera/color/image_raw")
#     p.add_argument("--left-wrist-topic", default="/left/camera/color/image_raw")
#     p.add_argument("--image-transport", choices=["compressed", "raw"], default="raw")
#     # Cap each frame's longest edge before sending it to the policy server; 0 disables.
#     # Only ever downscales. NOTE for this checkpoint: the cameras publish 320x240 and
#     # the server's own preprocessing resizes the SHORTEST edge to 256. A cap of 256
#     # therefore shrinks 320x240 to 256x192, and the server upsamples it back -- a
#     # resampling round trip the training data never went through. 0 (or 320) feeds the
#     # server the same 320x240 frames the dataset was recorded at.
#     p.add_argument("--image-max-edge", type=int, default=0)

#     # Video modality keys the checkpoint's processor expects, verbatim from
#     # processor_config.json -> modality_configs -> daksha_b1 -> video.
#     p.add_argument("--zed-right-camera-key", default="zed_right")
#     p.add_argument("--zed-left-camera-key", default="zed_left")
#     p.add_argument("--right-wrist-camera-key", default="wrist_right")
#     p.add_argument("--left-wrist-camera-key", default="wrist_left")

#     # IMPORTANT: key order MUST be right-first to match the slice indices in
#     # __init__ (policy_right_arm_indices=[0..6], right_gripper=7, left_arm=[8..14],
#     # left_gripper=15) and the action concatenation order in _extract_action_vec16.
#     # The checkpoint's `right_arm` group is the physical right arm (verified against
#     # the baked dataset statistics), so the physical right joints must occupy slots 0..7.
#     p.add_argument(
#         "--state-obs-keys-csv",
#         default=(
#             "right_joint_1,right_joint_2,right_joint_3,right_joint_4,right_joint_5,right_joint_6,right_joint_7,right_gripper_right_joint,"
#             "left_joint_1,left_joint_2,left_joint_3,left_joint_4,left_joint_5,left_joint_6,left_joint_7,left_gripper_left_joint"
#         ),
#     )
#     p.add_argument(
#         "--action-keys-csv",
#         default=(
#             "right_joint_1,right_joint_2,right_joint_3,right_joint_4,right_joint_5,right_joint_6,right_joint_7,right_gripper_right_joint,"
#             "left_joint_1,left_joint_2,left_joint_3,left_joint_4,left_joint_5,left_joint_6,left_joint_7,left_gripper_left_joint"
#         ),
#     )

#     # Safety clamp limits, right-first order to match the key CSVs above:
#     #   right_arm(7) + right_gripper(1) + left_arm(7) + left_gripper(1)
#     #
#     # Derived DIRECTLY from THIS checkpoint's baked training statistics
#     # (checkpoint-70000/experiment_cfg/dataset_statistics.json -> daksha_b1), so
#     # inference is clamped to the same value range the model was trained on. Per
#     # joint the bound is the union of the recorded action and state ranges, widened
#     # by max(10% of range, 0.05 rad) for arms and by 0.01 rad for grippers.
#     #
#     # Note the gripper signs, which are what a hand-written limit usually gets wrong
#     # here: the RIGHT gripper's recorded range is POSITIVE [0.0, 0.044] and the LEFT
#     # gripper's is NEGATIVE [-0.044, 0.0]. Clamping either to the other's sign pins
#     # that gripper to a constant and it never actuates.
#     #
#     # To re-derive for another checkpoint:
#     #   python - <<'EOF'
#     #   import json
#     #   d = json.load(open("<ckpt>/experiment_cfg/dataset_statistics.json"))["daksha_b1"]
#     #   for g in ("right_arm", "right_gripper", "left_arm", "left_gripper"):
#     #       lo = [min(a, s) for a, s in zip(d["action"][g]["min"], d["state"][g]["min"])]
#     #       hi = [max(a, s) for a, s in zip(d["action"][g]["max"], d["state"][g]["max"])]
#     #       m = [0.01 if "gripper" in g else max(0.1 * (h - l), 0.05) for l, h in zip(lo, hi)]
#     #       print(g, [round(l - e, 5) for l, e in zip(lo, m)], [round(h + e, 5) for h, e in zip(hi, m)])
#     #   EOF
#     p.add_argument(
#         "--joint-min-csv",
#         default=(
#             "-1.26631,-0.52212,-0.43299,0.20816,1.05684,-0.23902,-1.72448,-0.01,"
#             "-1.39176,-0.36248,-0.46181,-1.82025,-1.67081,-0.51649,-1.71634,-0.05401"
#         ),
#     )
#     p.add_argument(
#         "--joint-max-csv",
#         default=(
#             "1.48489,0.31187,0.51066,1.78671,1.89044,0.31845,1.73649,0.05405,"
#             "1.31943,0.32413,0.42672,-0.21759,-0.94923,0.32474,1.83418,0.01"
#         ),
#     )
#     # Per-step rate limit, right-first. At the 30 Hz control rate 0.03 rad/step is
#     # ~0.9 rad/s per arm joint. The gripper entries (0.05) exceed the gripper's whole
#     # 0.044 rad travel on purpose, so grip/release is not rate limited.
#     p.add_argument(
#         "--max-step-delta-csv",
#         default="0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.05,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.05",
#     )
#     return p


# def main() -> None:
#     args = build_arg_parser().parse_args()
#     rclpy.init()
#     node = Gr00TDakshaB1RosBridge(args)
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == "__main__":
#     main()
