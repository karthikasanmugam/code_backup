"""
Daksha VLA Training and Inference UI
Lightweight interface for ACT, Diffusion, and SmolVLA policies.
"""

import os
import json
import atexit
import importlib
import logging
import math
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime
from urllib.parse import quote
from urllib.request import urlopen
from flask import Flask, request, jsonify, render_template, Response, send_file, url_for
from flask_cors import CORS
import psutil
import signal
import select

try:
    import rclpy
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# ── DAKSHA ARM CONFIG REFERENCE ───────────────────────────────────────────────
# Mirrors examples/daksha/*/meta/modality.json. Kept here as UI defaults so the
# UI doesn't need to read a dataset's modality.json just to render forms.
DAKSHA_ARM_CONFIG = {
    "video": {
        "modality_keys": ["zed_right", "zed_left", "wrist_right", "wrist_left"],
        "camera_aliases": {
            "zed_right": "zed_right",
            "zed_left": "zed_left",
            "wrist_right": "right_wrist",
            "wrist_left": "left_wrist",
        },
    },
    "state": {
        "modality_keys": ["right_arm", "right_gripper", "left_arm", "left_gripper"],
        "state_dim": 16,
        "state_obs_keys_csv": "daksha_right_joint1,daksha_right_joint2,daksha_right_joint3,daksha_right_joint4,daksha_right_joint5,daksha_right_joint6,daksha_right_joint7,daksha_right_finger_joint1,daksha_left_joint1,daksha_left_joint2,daksha_left_joint3,daksha_left_joint4,daksha_left_joint5,daksha_left_joint6,daksha_left_joint7,daksha_left_finger_joint1",
    },
    "action": {
        "modality_keys": ["right_arm", "right_gripper", "left_arm", "left_gripper"],
        "action_keys_csv": "daksha_right_joint1,daksha_right_joint2,daksha_right_joint3,daksha_right_joint4,daksha_right_joint5,daksha_right_joint6,daksha_right_joint7,daksha_right_finger_joint1,daksha_left_joint1,daksha_left_joint2,daksha_left_joint3,daksha_left_joint4,daksha_left_joint5,daksha_left_joint6,daksha_left_joint7,daksha_left_finger_joint1",
    },
    "language": {
        "task_description_key": "annotation.human.task_description",
    },
}

DEFAULT_CAMERA_KEYS = ["zed_right", "zed_left", "right_wrist", "left_wrist"]

# ════════════════════════════════════════════════════════════════════════════════════
# APP CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════════

CONTAINER_ROOT = str(
    Path(os.environ.get("DAKSHA_PROJECT_ROOT", Path(__file__).resolve().parent))
    .expanduser()
    .resolve()
)
FRONTEND_ROOT = Path(CONTAINER_ROOT) / "ui" / "ui" / "frontend"
app = Flask(
    __name__,
    template_folder=str(FRONTEND_ROOT / "templates"),
    static_folder=str(FRONTEND_ROOT / "static"),
)
CORS(app)

# Base paths — derived from this file's location so the UI keeps working
# regardless of where the repo is checked out.
#
# This file lives at the lerobot_2 root; UI assets and scripts stay under
# <container>/ui/ui, and backend tools stay under <container>/ui.
# container holds exactly three folders:
#   <container>/
#     lerobot/   the LeRobot repo (src/, outputs/, pyproject.toml, tests/, ...)
#     ui/        the Daksha backends (act/ diffusion/ smolvla/ inference/) + this Flask UI (ui/ui/)
#     venv/      legacy Python environment, kept only as a fallback source of the Jetson
#                torch wheel; the UI now runs under the global/user site-packages instead.
# REPO_ROOT points at the LeRobot repo inside the container, so every REPO_ROOT/src
# and REPO_ROOT/outputs reference below stays valid.
# DAKSHA_ROOT is the ui/ folder (the former examples/daksha).
REPO_ROOT = f"{CONTAINER_ROOT}/lerobot"
DAKSHA_ROOT = f"{CONTAINER_ROOT}/ui"
UI_ROOT = f"{DAKSHA_ROOT}/ui"
OUTPUT_ROOT = f"/home/s1/Desktop/outputs/daksha"
SMOLVLA_OUTPUT_ROOT = f"{REPO_ROOT}/outputs/smolvla"
SMOLVLA_DEFAULT_RUN = f"{REPO_ROOT}/outputs/smolvla_daksha"

# Dataset and model paths
# DATASET_ROOT's immediate subdirectories are listed as selectable datasets in
# the UI (see list_datasets()) — it is the parent of dataset folders, not a
# dataset itself. This is where the Daksha data-collection recorder writes
# new episodes (see daksha_data_collection/config/config.yaml: dataset.root_dir),
# so recordings show up here with no manual copy step. Override with the
# DAKSHA_DATASET_ROOT env var if datasets live elsewhere. Keep the default on the
# current workspace dataset folder so it resolves to the actual local data store.
DATASET_ROOT = os.environ.get("DAKSHA_DATASET_ROOT", "/home/s1/Desktop/datasets/")
MODELS_ROOT = os.environ.get("DAKSHA_MODELS_ROOT", f"{REPO_ROOT}/outputs/daksha")


MODEL_VALIDATION_DIR = f"{UI_ROOT}/model_validation"
if MODEL_VALIDATION_DIR not in sys.path:
    sys.path.insert(0, MODEL_VALIDATION_DIR)
REPO_SOURCE_DIR = f"{REPO_ROOT}/src"
if REPO_SOURCE_DIR not in sys.path:
    sys.path.insert(0, REPO_SOURCE_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import uuid  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from vla_eval.datasets.lerobot_custom import LeRobotCustomDataset  # noqa: E402
from vla_eval.evaluation.splits import checkpoint_episode_split
from vla_eval.evaluation.act_episode import calculate_metrics, evaluate_act_episode  # noqa: E402
from vla_eval.core.registry import available_models  # noqa: E402
from vla_eval.models.act.config import ACTCheckpointConfig  # noqa: E402
from vla_eval.models.diffusion.config import DiffusionCheckpointConfig  # noqa: E402
from vla_eval.models.smolvla.config import SmolVLACheckpointConfig  # noqa: E402
from vla_eval.models.runtime import PROFILES, load_evaluation_model  # noqa: E402

# Reused to compute the same train/held-out-eval episode split the training pipeline
# itself uses (lerobot.datasets.factory.make_train_eval_datasets), so the Model
# Validation tab only ever evaluates on episodes the checkpoint didn't train on.
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402

# Log paths
LOGS_ROOT = "/tmp/daksha_logs"
TRAIN_LOG = f"{LOGS_ROOT}/training.log"
SERVER_LOG = f"{LOGS_ROOT}/server.log"
# Dataset preparation / conversion / validation output (dataset paths, codebase-version
# checks, LeRobot converter chatter, validation warnings) is kept off the training tab
# and written here instead. The training tab (TRAIN_LOG) shows only the dataset name,
# the policy, "Training started", and then the raw backend training output (loss, ...).
TRAIN_PREP_LOG = f"{LOGS_ROOT}/training_prep.log"
# Full, unfiltered backend stdout (shell launcher + lerobot_train) is written here. The
# training tab (TRAIN_LOG) gets only the worthwhile lines mirrored from this file -- see
# _keep_training_line / _monitor_training_logs. Keeping the raw stream on disk preserves
# full output for debugging and lets the mirror be rebuilt after a UI restart.
TRAIN_RAW_LOG = f"{LOGS_ROOT}/training_raw.log"
CLIENT_LOG = f"{LOGS_ROOT}/client.log"
INFERENCE_LOG = f"{LOGS_ROOT}/inference.log"
BRIDGE_LOG = f"{LOGS_ROOT}/bridge.log"
PROCESS_STATE_PATH = f"{LOGS_ROOT}/managed_processes.json"

# Create necessary directories
os.makedirs(LOGS_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

app.config['UPLOAD_FOLDER'] = '/tmp/daksha_uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10 GB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════════════
# POLICY TYPE CONFIGURATIONS
# ════════════════════════════════════════════════════════════════════════════════════

POLICY_CONFIGS = {
    "act": {
        "name": "Action Chunking Transformer (ACT)",
        "description": "Transformer-based action chunking for discrete steps",
        "script": f"{DAKSHA_ROOT}/act/train_act_local.sh",
        "pretrained": "facebook/dinov2-base",
        "features": ["action_chunking", "multi_camera", "efficient"],
        "hyperparameters": {
            "batch_size": {"default": 8, "min": 1, "max": 128, "type": "int"},
            "steps": {"default": 50000, "min": 1000, "max": 500000, "type": "int"},
            "lr": {"default": 1e-4, "min": 1e-6, "max": 1e-2, "type": "float"},
            "chunk_size": {"default": 32, "min": 1, "max": 64, "type": "int"},
            "gradient_accumulation": {"default": 1, "min": 1, "max": 16, "type": "int"},
            # Save early enough that a normal operator stop still leaves a usable
            # model. The retention worker keeps only the three newest snapshots.
            "save_freq": {"default": 500, "min": 100, "max": 10000, "type": "int"},
            # Independent of save_freq -- see EVAL_STEPS wiring in _prepare_and_train.
            # Lowering save_freq alone no longer drags eval passes along with it.
            "eval_freq": {"default": 500, "min": 100, "max": 10000, "type": "int"},
        }
    },
    "diffusion": {
        "name": "Diffusion Policy",
        "description": "Latent diffusion model for continuous action prediction",
        "script": f"{DAKSHA_ROOT}/diffusion/train_diffusion_local.sh",
        "pretrained": "stabilityai/stable-diffusion-v1-5",
        "features": ["diffusion_model", "continuous_actions", "smooth_trajectories"],
        "hyperparameters": {
            "batch_size": {"default": 4, "min": 1, "max": 64, "type": "int"},
            "save_freq": {"default": 500, "min": 1, "max": 10000, "type": "int"},
            # Independent of save_freq -- see EVAL_STEPS wiring in _prepare_and_train.
            "eval_freq": {"default": 500, "min": 1, "max": 10000, "type": "int"},
            "steps": {"default": 50000, "min": 1000, "max": 500000, "type": "int"},
            "lr": {"default": 1e-4, "min": 1e-6, "max": 1e-2, "type": "float"},
            "num_diffusion_steps": {"default": 100, "min": 10, "max": 1000, "type": "int"},
            "guidance_scale": {"default": 1.0, "min": 0.0, "max": 10.0, "type": "float"},
        }
    },
    "smolvla": {
        "name": "SmolVLA",
        "description": "Vision-language action policy",
        "script": f"{DAKSHA_ROOT}/smolvla/train_smolvla_local.sh",
        "pretrained": "lerobot/smolvla_base",
        "features": ["vision_language_model", "multi_camera", "action_chunking"],
        "hyperparameters": {
            "pretrained_path": {"default": "lerobot/smolvla_base", "type": "string"},
            "batch_size": {"default": 4, "min": 1, "max": 64, "type": "int"},
            # Action horizon = number of future actions predicted per inference; passed to
            # training as --policy.chunk_size and --policy.n_action_steps (see launchers).
            "action_horizon": {"default": 20, "min": 1, "max": 100, "type": "int"},
            "steps": {"default": 3000, "min": 1, "max": 500000, "type": "int"},
            "save_freq": {"default": 500, "min": 1, "max": 10000, "type": "int"},
            # Independent of save_freq -- see EVAL_STEPS wiring in configure_smolvla_environment.
            "eval_freq": {"default": 500, "min": 1, "max": 10000, "type": "int"},
            "log_freq": {"default": 20, "min": 1, "max": 1000, "type": "int"},
            "num_workers": {"default": 0, "min": 0, "max": 16, "type": "int"},
            "freeze_vision_encoder": {"default": False, "type": "bool"},
            "train_expert_only": {"default": False, "type": "bool"},
            "offline": {"default": False, "type": "bool"},
        },
    },
}


def smolvla_training_config(config):
    """Validate only SmolVLA controls; other policies retain their existing behavior."""
    if not isinstance(config, dict):
        raise ValueError("SmolVLA config must be an object")
    specs = POLICY_CONFIGS["smolvla"]["hyperparameters"]
    unknown = set(config) - set(specs)
    if unknown:
        raise ValueError(f"Unsupported SmolVLA settings: {', '.join(sorted(unknown))}")
    result = {}
    for key, spec in specs.items():
        value = config.get(key, spec["default"])
        if spec["type"] == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not spec["min"] <= value <= spec["max"] or int(value) != value:
                raise ValueError(f"{key} must be an integer from {spec['min']} to {spec['max']}")
            value = int(value)
        elif spec["type"] == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path or model ID")
        result[key] = value.strip() if isinstance(value, str) else value
    return result


def configure_smolvla_environment(env, config):
    for key, value in smolvla_training_config(config).items():
        if key != "offline":
            env[key.upper()] = str(value).lower() if isinstance(value, bool) else str(value)
    env.update({"VIDEO_BACKEND": "pyav", "COMPILE_MODEL": "false", "TOLERANCE_S": "0.001",
                "EVAL_SPLIT": "0.1", "EVAL_STEPS": str(config.get("eval_freq", 500)),
                "MAX_EVAL_SAMPLES": "64",
                "REPAIR_MISSING_QUANTILES": "true",
                "WANDB_ENABLE": "false", "PUSH_TO_HUB": "false", "PYTHONUNBUFFERED": "1"})
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env[key] = "1" if config.get("offline", False) else "0"

# ════════════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ════════════════════════════════════════════════════════════════════════════════════

global_state = {
    "training_process": None,
    "training_active": False,
    "training_policy": None,
    # "preparing" while a checkpoint resume is initialized, "converting" while a dataset
    # is auto-converted to v3.0, and "training" once lerobot_train has been launched.
    "training_stage": None,
    "cancel_requested": False,
    # Per-run cancellation prevents a stale preparation thread from being revived.
    "training_generation": 0,
    "training_cancel_event": None,
    # output_dir of the most recently started/resumed run (fresh or resumed) -- lets the
    # Training tab's "Retrain" button resume the run just stopped with one click, with
    # no need to go find it in the Checkpoints tab first.
    "last_output_dir": None,
    "server_process": None,
    "server_port": None,
    # Resolved from the selected checkpoint's saved train_config.json and its
    # dataset metadata when the policy server starts.  This is deliberately
    # backend-only: operators should never have to transcribe model metadata.
    "inference_metadata": None,
    "inference_active": False,
    # The ROS2 bridge (examples/daksha/inference/eval_daksha_act.py for ACT/SmolVLA,
    # eval_daksha_diffusion.py for Diffusion -- see ACT_BRIDGE_SCRIPT/
    # DIFFUSION_BRIDGE_SCRIPT) that actually commands the robot from the running policy
    # server -- separate from server_process/inference_active, which only track the ZMQ
    # policy server itself (unused for Diffusion, which the bridge loads directly).
    "bridge_process": None,
    "bridge_active": False,
    "ros_node": None,
    "current_step": 0,
    "total_steps": 0,
    "training_resumed_from_step": 0,
    "validation_ui_process": None,
    "validation_ui_port": None,
    "validation_api_process": None,
    "validation_api_port": None,
    "validation_frontend_host": None,
}
# Guards every "is X already running?" check-then-launch sequence below (training,
# inference server, bridge). Without it, two near-simultaneous start requests (a
# double-click, or two browser tabs) can both pass the "not already running" check
# before either sets its process handle, launch two subprocesses, and have the second
# assignment silently clobber the first's handle -- leaking the first process,
# untracked and unstoppable from the UI.
global_state_lock = threading.Lock()


def _read_managed_processes():
    try:
        value = json.loads(Path(PROCESS_STATE_PATH).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_managed_processes(processes):
    """Atomically persist child identities so a restarted UI can reattach controls."""
    temporary = Path(f"{PROCESS_STATE_PATH}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(processes, indent=2))
        temporary.replace(PROCESS_STATE_PATH)
    except OSError as exc:
        log_message(SERVER_LOG, f"Could not persist process state: {exc}")
        temporary.unlink(missing_ok=True)


def _register_managed_process(role, process, **metadata):
    processes = _read_managed_processes()
    if process is None:
        processes.pop(role, None)
    else:
        try:
            created = psutil.Process(process.pid).create_time()
        except (psutil.Error, OSError):
            created = time.time()
        processes[role] = {"pid": process.pid, "pgid": os.getpgid(process.pid),
                           "create_time": created, **metadata}
    _write_managed_processes(processes)


def _managed_process_record(role):
    record = _read_managed_processes().get(role)
    if not record:
        return None
    try:
        process = psutil.Process(int(record["pid"]))
        if process.status() == psutil.STATUS_ZOMBIE or abs(process.create_time() - float(record["create_time"])) > 1.0:
            raise psutil.NoSuchProcess(process.pid)
        return record
    except (KeyError, TypeError, ValueError, psutil.Error, OSError):
        processes = _read_managed_processes()
        processes.pop(role, None)
        _write_managed_processes(processes)
        return None


def _clear_managed_process(role, process=None):
    record = _managed_process_record(role)
    if record is None or process is None or int(record.get("pid", -1)) == process.pid:
        processes = _read_managed_processes()
        processes.pop(role, None)
        _write_managed_processes(processes)


def _stop_managed_process(role, process=None, timeout=10):
    """Stop a tracked process group, including when its original Flask process is gone."""
    record = _managed_process_record(role)
    if process is not None and process.poll() is None:
        pid, pgid = process.pid, os.getpgid(process.pid)
    elif record:
        pid, pgid = int(record["pid"]), int(record.get("pgid", record["pid"]))
    else:
        return False
    try:
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                child = psutil.Process(pid)
                if child.status() == psutil.STATUS_ZOMBIE or not child.is_running():
                    break
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                break
            time.sleep(0.1)
        else:
            os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        _clear_managed_process(role, process)
    return True


def _recover_managed_processes():
    """Restore status fields for surviving child processes after a UI restart."""
    training = _managed_process_record("training")
    if training:
        global_state.update(training_active=True, training_policy=training.get("policy"),
                            training_stage=training.get("stage", "training"),
                            last_output_dir=training.get("output_dir"),
                            total_steps=int(training.get("total_steps") or 0),
                            current_step=int(training.get("current_step") or 0),
                            training_resumed_from_step=int(training.get("resumed_from_step") or 0),
                            cancel_requested=False)
        print(f"[daksha-ui] Reattached to training process {training['pid']} after UI restart")
        # The log mirror (raw -> training tab) is re-attached from main(), once the
        # filtering helpers defined later in this module are available.
        global_state["training_needs_log_reattach"] = True
    try:
        metadata_path = Path(PROCESS_STATE_PATH).with_name("inference_metadata.json")
        if metadata_path.is_file():
            global_state["inference_metadata"] = json.loads(metadata_path.read_text())
            global_state["inference_active"] = True
    except (OSError, ValueError):
        pass
    for role, active_key in (("inference", "inference_active"), ("bridge", "bridge_active")):
        record = _managed_process_record(role)
        if record:
            global_state[active_key] = True
            if role == "inference":
                global_state["server_port"] = record.get("port")
                global_state["inference_metadata"] = record.get("metadata")


_recover_managed_processes()


def _stop_stale_validators():
    """Validator UI/API belong to one app.py run; any recorded at startup are orphans."""
    for role in ("validation_api", "validation_ui"):
        if _managed_process_record(role):
            _stop_managed_process(role, timeout=3)
            print(f"[daksha-ui] Stopped orphaned {role} left by a previous run")


_stop_stale_validators()

_shutdown_lock = threading.Lock()
_shutdown_done = False


def shutdown_all_children(reason="shutdown"):
    """Stop every child process (and free its port) when this app.py exits.

    Training survives UI restarts by default and is reattached from
    managed_processes.json. Set DAKSHA_KEEP_TRAINING=0 to stop it on exit.
    """
    global _shutdown_done
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True

    keep_training = os.environ.get("DAKSHA_KEEP_TRAINING", "1").lower() not in ("0", "false", "no")
    tracked = [
        ("bridge", global_state.get("bridge_process")),
        ("inference", global_state.get("server_process")),
        ("validation_ui", global_state.get("validation_ui_process")),
        ("validation_api", global_state.get("validation_api_process")),
    ]
    if not keep_training:
        tracked.append(("training", global_state.get("training_process")))
        cancel_event = global_state.get("training_cancel_event")
        if cancel_event is not None:
            cancel_event.set()

    for role, process in tracked:
        try:
            if _stop_managed_process(role, process, timeout=5):
                print(f"[daksha-ui] Stopped {role} ({reason})", flush=True)
        except Exception as exc:  # never let one failure block the rest of the cleanup
            print(f"[daksha-ui] Could not stop {role}: {exc}", flush=True)
    Path(PROCESS_STATE_PATH).with_name("inference_metadata.json").unlink(missing_ok=True)


atexit.register(shutdown_all_children)

# ════════════════════════════════════════════════════════════════════════════════════
# GPU / MEMORY MONITORING (tegrastats)
# ════════════════════════════════════════════════════════════════════════════════════
# Jetson has no discrete VRAM -- CPU and GPU share one physical RAM pool. "gpu_percent"
# below is the GR3D (GPU engine) load tegrastats reports; "vram" is that shared RAM
# pool's usage, which is the closest real analogue to VRAM on this hardware. Sampled
# from a long-lived `tegrastats` subprocess instead of spawning one per request, since
# each spawn takes real time to produce its first line.
_tegra_stats = {"gpu_percent": None, "ram_used_mb": None, "ram_total_mb": None}
_tegra_stats_lock = threading.Lock()

_TEGRA_RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
_TEGRA_GPU_RE = re.compile(r"GR3D_FREQ (\d+)%")

def _tegrastats_monitor():
    """Background reader for `tegrastats`; no-op (stats stay None) off Jetson."""
    tegrastats_bin = shutil.which("tegrastats")
    if not tegrastats_bin:
        return
    try:
        proc = subprocess.Popen(
            [tegrastats_bin, "--interval", "1000"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except Exception as e:
        print(f"tegrastats monitor failed to start: {e}")
        return
    for line in proc.stdout:
        ram_match = _TEGRA_RAM_RE.search(line)
        gpu_match = _TEGRA_GPU_RE.search(line)
        with _tegra_stats_lock:
            if ram_match:
                _tegra_stats["ram_used_mb"] = int(ram_match.group(1))
                _tegra_stats["ram_total_mb"] = int(ram_match.group(2))
            if gpu_match:
                _tegra_stats["gpu_percent"] = int(gpu_match.group(1))

threading.Thread(target=_tegrastats_monitor, daemon=True).start()

# ════════════════════════════════════════════════════════════════════════════════════
# ROS2 INTEGRATION
# ════════════════════════════════════════════════════════════════════════════════════

class InstructionPublisher:
    def __init__(self):
        if ROS2_AVAILABLE:
            self.node = rclpy.create_node('daksha_vla_ui_pub')
            self.pub = self.node.create_publisher(String, '/daksha_instruction', 10)
        else:
            self.node = None

    def publish(self, text):
        if ROS2_AVAILABLE and self.node:
            msg = String()
            msg.data = text
            self.pub.publish(msg)
        else:
            print(f'[MOCK ROS2] Published: {text}')

def init_ros2():
    if ROS2_AVAILABLE:
        threading.Thread(target=_ros2_spin, daemon=True).start()

def _ros2_spin():
    try:
        rclpy.init(args=None)
        global_state["ros_node"] = InstructionPublisher()
        rclpy.spin(global_state["ros_node"].node)
    except Exception as e:
        print(f"ROS2 Error: {e}")

# ════════════════════════════════════════════════════════════════════════════════════
# LOG AND PROCESS HELPERS
# ════════════════════════════════════════════════════════════════════════════════════

def log_message(log_path, message):
    """Write message to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    try:
        with open(log_path, 'a') as f:
            f.write(full_message + '\n')
    except Exception as e:
        print(f"Log write error: {e}")

def clear_log(log_path):
    """Clear log file."""
    try:
        open(log_path, 'w').close()
    except Exception as e:
        print(f"Log clear error: {e}")

def stream_log(log_path, active_check=None, line_transform=None):
    """SSE generator for real-time log streaming.

    Without `active_check` this tails forever, which is fine for a single long-lived
    connection but means a stale one from a finished run never closes -- the frontend
    guards against opening a second stream while one is "active", so a connection that
    never terminates would permanently block the next run's log stream. When
    `active_check` is given, the stream closes itself a couple seconds after
    `active_check()` goes false and no new lines have shown up (a short grace period in
    case a few final lines are still being flushed).
    """
    open(log_path, 'a').close()
    with open(log_path, 'r') as f:
        for line in f:
            message = line.rstrip()
            if line_transform:
                message = line_transform(message)
            yield f"data: {message}\n\n"
        idle_after_inactive = 0
        while True:
            line = f.readline()
            if line:
                idle_after_inactive = 0
                message = line.rstrip()
                if line_transform:
                    message = line_transform(message)
                yield f"data: {message}\n\n"
                continue
            if active_check is not None and not active_check():
                idle_after_inactive += 1
                if idle_after_inactive > 6:  # ~3s of quiet after the run ended
                    return
            time.sleep(0.5)

def get_process_stats(process):
    """Get CPU and memory stats for process."""
    try:
        if process and process.poll() is None:
            p = psutil.Process(process.pid)
            return {
                "cpu_percent": p.cpu_percent(interval=0.1),
                "memory_mb": p.memory_info().rss / 1024 / 1024,
                "status": "running"
            }
    except Exception as e:
        print(f"Process stats error: {e}")
    return {"status": "stopped"}

def resolve_repo_python():
    """Return an argv prefix that runs Python with the repo's `src` on PYTHONPATH.

    After the global-libraries migration, the UI, training, inference and dataset
    tooling all share one interpreter instead of a per-container venv: every
    dependency (torch, pandas, pyarrow, msgpack, zmq, jsonlines, ...) is installed
    into the global/user site-packages of the Python this app.py runs under, and
    lerobot is imported from REPO_ROOT/src via PYTHONPATH. The Jetson torch build
    (sm_87) lives in that same global site-packages.

    DAKSHA_PYTHON still overrides the interpreter for non-standard setups (e.g. a
    machine that deliberately keeps lerobot's deps in a separate environment).
    """
    python_bin = os.environ.get("DAKSHA_PYTHON") or sys.executable
    return [python_bin], {"PYTHONPATH": f"{REPO_ROOT}/src"}


def resolve_repo_python_executable():
    """Single interpreter path for shell launchers (SMOLVLA_PYTHON) that can't take an argv prefix."""
    return resolve_repo_python()[0][0]


# Which bridge script runs depends on the *checkpoint's own* saved policy type (see
# start_inference_server), never the UI's policy-type filter dropdown:
#   - ACT checkpoints go through the generic ZMQ server (serve_policy_zmq.py) and
#     eval_daksha_act.py connects to it over the network.
#   - SmolVLA checkpoints go through the same ZMQ protocol, served by
#     smolvla/serve_smolvla_zmq.py (started with an extra --robot-type flag -- see
#     start_inference_server() below), and eval_daksha_smolvla.py connects to it. It is
#     a separate file from eval_daksha_act.py -- forked rather than shared -- so
#     SmolVLA-specific bridge behavior can diverge later without touching the ACT path;
#     today the two are functionally identical.
#   - Diffusion checkpoints cannot go through that server (it calls
#     predict_action_chunk() with a single timestep and never populates
#     DiffusionPolicy's n_obs_steps observation-history queues -- see
#     eval_daksha_diffusion.py's module docstring), so eval_daksha_diffusion.py loads
#     the checkpoint directly in-process instead and needs no separate server.
ACT_BRIDGE_SCRIPT = f"{DAKSHA_ROOT}/inference/eval_daksha_act.py"
SMOLVLA_BRIDGE_SCRIPT = f"{DAKSHA_ROOT}/inference/eval_daksha_smolvla.py"
DIFFUSION_BRIDGE_SCRIPT = f"{DAKSHA_ROOT}/inference/eval_daksha_diffusion.py"
ROS_PYTHON_BIN = os.environ.get("DAKSHA_ROS_PYTHON", "/usr/bin/python3")
ROS_SETUP = os.environ.get("DAKSHA_ROS_SETUP", "/opt/ros/humble/setup.bash")
# eval_daksha_diffusion.py needs both rclpy (ROS) and lerobot's real deps (torch, ...)
# importable in the *same* interpreter. After the global-libraries migration the system
# python3 has the full lerobot stack, and sourcing ROS_SETUP makes rclpy importable from
# it too, so the two coexist in one interpreter. Point this at the ROS install's
# site-packages (e.g. /opt/ros/humble/lib/python3.10/site-packages) only if rclpy isn't
# already importable after sourcing ROS_SETUP.
_ROS_DEFAULT_SITE_PACKAGES = (
    f"{Path(ROS_SETUP).parent}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
)
ROS_SITE_PACKAGES = os.environ.get(
    "DAKSHA_ROS_SITE_PACKAGES",
    _ROS_DEFAULT_SITE_PACKAGES if os.path.isdir(_ROS_DEFAULT_SITE_PACKAGES) else "",
)


def build_ros_bridge_cmd(script_path: str, args: list[str], needs_lerobot: bool = False) -> list[str]:
    """Build the argv to run a Daksha ROS2 bridge script (eval_daksha_act.py,
    eval_daksha_smolvla.py, or eval_daksha_diffusion.py).

    The ACT/SmolVLA bridge only needs rclpy/cv2 (system/ROS packages) rather than
    lerobot's own deps -- resolve_repo_python()'s .venv does not have rclpy installed --
    so it runs under the system ROS interpreter. Mirrors
    examples/daksha/inference/common.sh's inference_ros_python(): source the ROS2 setup
    script, then run the system python.

    The Diffusion bridge additionally loads the checkpoint itself in-process (see
    eval_daksha_diffusion.py's module docstring), so it needs lerobot's real deps
    (torch, ...) importable too -- pass `needs_lerobot=True` to run it under
    resolve_repo_python()'s interpreter instead, with ROS_SITE_PACKAGES prepended to
    PYTHONPATH so rclpy stays importable from there.

    Routed through `bash -c` (with each argument individually quoted) since sourcing a
    shell script isn't expressible as a plain argv list.
    """
    if needs_lerobot:
        python_argv, extra_env = resolve_repo_python()
        pythonpath = ":".join(p for p in (ROS_SITE_PACKAGES, extra_env.get("PYTHONPATH")) if p)
        inner = python_argv + [script_path, *args]
    else:
        pythonpath = ""
        inner = [ROS_PYTHON_BIN, script_path, *args]
    quoted = " ".join(shlex.quote(a) for a in inner)
    export_line = f'export PYTHONPATH="{pythonpath}:${{PYTHONPATH:-}}"; ' if pythonpath else ""
    script = f'if [ -f "{ROS_SETUP}" ]; then source "{ROS_SETUP}"; fi; {export_line}exec {quoted}'
    return ["bash", "-c", script]


def resolve_pretrained_model_dir(checkpoint_path):
    """Resolve a UI-selected checkpoint/run path to an actual pretrained_model dir.

    The inference dropdown passes an exact saved step directory such as
    `<output_dir>/checkpoints/006500`. The real model weights live in that
    directory's `pretrained_model` child (see
    lerobot.common.train_utils.save_checkpoint). Run directories are still
    accepted for resume/backward compatibility and resolve to their newest step.
    If checkpoint_path is already a pretrained_model dir (has config.json), it
    is returned as-is.
    """
    path = Path(checkpoint_path)
    if (path / "config.json").exists():
        return path

    # Exact checkpoint step selected by the inference UI. Resolve this before
    # looking for a nested `checkpoints/` directory, which only run roots have.
    step_model = path / "pretrained_model"
    if (step_model / "config.json").exists() and _is_checkpoint_complete(path):
        return step_model

    checkpoints_dir = path / "checkpoints"
    if not checkpoints_dir.is_dir():
        return path

    last_dir = checkpoints_dir / "last" / "pretrained_model"
    if last_dir.exists() and _is_checkpoint_complete(last_dir.parent):
        return last_dir

    step_dirs = sorted(
        (d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit() and _is_checkpoint_complete(d)),
        key=lambda d: int(d.name),
    )
    if step_dirs:
        candidate = step_dirs[-1] / "pretrained_model"
        if candidate.exists():
            return candidate

    return path


def find_available_port(start_port=5555):
    """Find an available port."""
    import socket
    port = start_port
    while port < start_port + 1000:
        try:
            s = socket.socket()
            s.bind(('', port))
            s.close()
            return port
        except OSError:
            port += 1
    raise RuntimeError("No available ports found")


# ── LE ROBOT V3 DATASET VALIDATION UI ─────────────────────────────────────────
# The modular validator under data_validation/ consumes the standard LeRobot v3.0
# layout. Both the async Validate-button flow and its direct URL endpoint call the
# same idempotent ensure_dataset_v30() conversion used by training before opening it.
VALIDATION_UI_DIR = f"{UI_ROOT}/data_validation"
VALIDATION_BACKEND_DIR = f"{VALIDATION_UI_DIR}/backend_v3"
VALIDATION_FRONTEND_DIR = f"{VALIDATION_UI_DIR}/frontend"
# No-cache static server so frontend edits (index.html/CSS/JS) are always picked
# up by the browser without a hard refresh. Falls back to http.server if missing.
VALIDATION_FRONTEND_SERVER = f"{VALIDATION_UI_DIR}/serve_frontend.py"
VALIDATION_UI_DEFAULT_PORT = int(os.environ.get("DAKSHA_VALIDATION_UI_PORT", 8124))
VALIDATION_API_DEFAULT_PORT = int(os.environ.get("DAKSHA_VALIDATION_API_PORT", 8000))
validation_prepare_jobs = {}
validation_prepare_lock = threading.Lock()


def resolve_validation_python():
    """Return the interpreter that runs the v3 validator API/static server.

    After the global-libraries migration the validator shares the same global Python
    as the rest of the UI -- its dependencies (fastapi, uvicorn, numpy, pandas,
    pyarrow, yaml, cv2) are installed globally alongside everything else.
    DAKSHA_VALIDATION_PYTHON still overrides the interpreter if needed.
    """
    python_bin = os.environ.get("DAKSHA_VALIDATION_PYTHON") or sys.executable
    required = "import fastapi, uvicorn, numpy, pandas, pyarrow, yaml, cv2"
    try:
        subprocess.run(
            [python_bin, "-c", required],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"The validator dependencies (FastAPI, Uvicorn, PyArrow, OpenCV, ...) are not "
            f"importable from {python_bin}. Install them globally, or set "
            f"DAKSHA_VALIDATION_PYTHON to a python executable that has them."
        ) from exc
    log_message(SERVER_LOG, f"Using validator Python environment: {python_bin}")
    return [python_bin]


def _dataset_layout_kind(dataset_path: Path) -> str:
    """"v3.0" (real converted layout), "raw" (per-episode, pre-conversion), or "unknown"."""
    if _dataset_is_v30_ready(dataset_path):
        return "v3.0"
    if list(dataset_path.glob("data/chunk-*/episode_*.parquet")):
        return "raw"
    return "unknown"


def _raw_backup_path(dataset_path: Path) -> Path | None:
    """Find a historical pre-conversion dump when one still exists.

    New conversions remove "<dataset>_old" after the promoted v3.0 dataset passes
    verification. Older conversions may still have a backup in the raw layout.
    """
    candidate = dataset_path.parent / f"{dataset_path.name}_old"
    if candidate.is_dir() and _dataset_layout_kind(candidate) == "raw":
        return candidate
    return None


def _validator_services_ready(frontend_port: int, api_port: int) -> bool:
    """Return true only after both the frontend and API answer HTTP requests."""
    try:
        with urlopen(f"http://127.0.0.1:{api_port}/api/health", timeout=0.5) as response:
            health = json.loads(response.read().decode("utf-8"))
            if response.status != 200 or health.get("status") != "ok":
                return False
        with urlopen(f"http://127.0.0.1:{frontend_port}/", timeout=0.5) as response:
            return response.status == 200 and b"<html" in response.read(1024).lower()
    except Exception:
        return False


def _stop_validation_child(process) -> None:
    """Stop a validator child and anything it spawned (each runs in its own session)."""
    if process is None or process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        process.kill()
        process.wait(timeout=2)


def ensure_validation_ui_running(frontend_host: str = "127.0.0.1") -> int | None:
    """Start the modular v3 validator UI/API once and wait until both are ready."""
    with global_state_lock:
        proc = global_state.get("validation_ui_process")
        api_proc = global_state.get("validation_api_process")
        ui_port = global_state.get("validation_ui_port")
        api_port = global_state.get("validation_api_port")
        if (
            proc is not None
            and proc.poll() is None
            and api_proc is not None
            and api_proc.poll() is None
            and ui_port
            and api_port
            and global_state.get("validation_frontend_host") == frontend_host
            and _validator_services_ready(ui_port, api_port)
        ):
            return ui_port

        if not (os.path.isdir(VALIDATION_BACKEND_DIR) and os.path.isdir(VALIDATION_FRONTEND_DIR)):
            log_message(SERVER_LOG, f"Validator directories not found under {VALIDATION_UI_DIR}")
            return None

        _stop_validation_child(proc)
        _stop_validation_child(api_proc)
        global_state["validation_ui_process"] = None
        global_state["validation_api_process"] = None
        global_state["validation_ui_port"] = None
        global_state["validation_api_port"] = None

        port = find_available_port(VALIDATION_UI_DEFAULT_PORT)
        api_port = find_available_port(VALIDATION_API_DEFAULT_PORT)
        python_argv = resolve_validation_python()
        env = os.environ.copy()
        api_env = dict(env)
        # The validator API imports both its own package (lerobot_validator, under
        # backend_v3/src) and lerobot itself (e.g. deletion.py's
        # `from lerobot.datasets import dataset_tools`). After the global-libraries
        # migration lerobot is NOT pip-installed -- it is imported from REPO_ROOT/src
        # via PYTHONPATH, exactly like resolve_repo_python() does -- so that path must
        # be present here too, or every dataset-editing operation (delete episodes, ...)
        # dies with "No module named 'lerobot.datasets'". Keep backend_v3/src first so
        # lerobot_validator resolves, then lerobot's src, then anything inherited.
        api_env["PYTHONPATH"] = os.pathsep.join(
            p for p in (
                str(Path(VALIDATION_BACKEND_DIR) / "src"),
                f"{REPO_ROOT}/src",
                env.get("PYTHONPATH", ""),
            ) if p
        )
        api_env["DAKSHA_VALIDATION_FRONTEND_ORIGIN"] = f"http://{frontend_host}:{port}"
        api_proc = None
        proc = None

        try:
            # Redirect to a real file: unconsumed PIPE output eventually fills and
            # blocks long-running validation/API and static-server child processes.
            with open(SERVER_LOG, "a", encoding="utf-8", buffering=1) as child_log:
                api_proc = subprocess.Popen(
                    python_argv + [
                        "-m", "uvicorn", "lerobot_validator.api.app:app",
                        "--host", "0.0.0.0", "--port", str(api_port),
                    ],
                    env=api_env,
                    stdout=child_log,
                    stderr=subprocess.STDOUT,
                    cwd=VALIDATION_BACKEND_DIR,
                    start_new_session=True,
                )
                if os.path.isfile(VALIDATION_FRONTEND_SERVER):
                    frontend_argv = python_argv + [
                        VALIDATION_FRONTEND_SERVER, str(port), VALIDATION_FRONTEND_DIR, "0.0.0.0",
                    ]
                else:
                    frontend_argv = python_argv + [
                        "-m", "http.server", str(port), "--bind", "0.0.0.0",
                    ]
                proc = subprocess.Popen(
                    frontend_argv,
                    env=env,
                    stdout=child_log,
                    stderr=subprocess.STDOUT,
                    cwd=VALIDATION_FRONTEND_DIR,
                    start_new_session=True,
                )

            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if proc.poll() is not None or api_proc.poll() is not None:
                    break
                if _validator_services_ready(port, api_port):
                    global_state["validation_ui_process"] = proc
                    global_state["validation_ui_port"] = port
                    global_state["validation_api_process"] = api_proc
                    global_state["validation_api_port"] = api_port
                    global_state["validation_frontend_host"] = frontend_host
                    # Persist identities so shutdown (or the next startup) can always
                    # find and stop these, even if this process is killed uncleanly.
                    _register_managed_process("validation_api", api_proc, port=api_port)
                    _register_managed_process("validation_ui", proc, port=port)
                    log_message(SERVER_LOG, f"Modular validator ready: UI={port}, API={api_port}")
                    return port
                time.sleep(0.15)

            log_message(SERVER_LOG, "Modular validator failed its 15-second startup health check")
        except Exception as exc:
            log_message(SERVER_LOG, f"Could not start modular validator: {exc}")

        _stop_validation_child(proc)
        _stop_validation_child(api_proc)
        return None

def format_bytes(num_bytes):
    """Convert bytes to a human-readable string."""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == 'B':
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def get_dir_size(path):
    """Recursively sum file sizes for a directory."""
    total = 0
    for root, _, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                pass
    return total


def resolve_dataset_path(value):
    """Resolve a dataset ID or path while staying under DATASET_ROOT."""
    candidate = Path(str(value or '').strip()).expanduser()
    if not candidate.is_absolute():
        candidate = Path(DATASET_ROOT) / candidate
    candidate = candidate.resolve()
    allowed = Path(DATASET_ROOT).resolve()
    if candidate != allowed and allowed not in candidate.parents:
        raise PermissionError('Dataset path is outside the configured dataset root.')
    return candidate


# Every root a legitimate checkpoint can live under -- ACT/Diffusion runs go under
# OUTPUT_ROOT (== MODELS_ROOT by default, but each is independently overridable via
# env), and SmolVLA runs can additionally live under SMOLVLA_OUTPUT_ROOT or at the
# single default path SMOLVLA_DEFAULT_RUN (see list_checkpoints()).
CHECKPOINT_ROOTS = [OUTPUT_ROOT, MODELS_ROOT, SMOLVLA_OUTPUT_ROOT, SMOLVLA_DEFAULT_RUN]


def resolve_checkpoint_path(value):
    """Resolve a checkpoint path while confining it to a known checkpoint root."""
    candidate = Path(str(value or '').strip()).expanduser().resolve()
    for root in CHECKPOINT_ROOTS:
        allowed = Path(root).resolve()
        if candidate == allowed or allowed in candidate.parents:
            return candidate
    raise PermissionError('Checkpoint path is outside the configured checkpoint roots.')


VALIDATE_DATASET_SCRIPT = f"{UI_ROOT}/validate_dataset_cli.py"


def validate_dataset_with_lerobot(dataset_path, repo_id=None):
    """Validate a local dataset using the real LeRobot validation module.

    Runs as a subprocess under resolve_repo_python() (the global interpreter with the
    lerobot stack installed) rather than importing lerobot in-process, to keep dataset
    validation isolated from the Flask server process.
    """
    dataset_path = os.path.abspath(dataset_path)
    dataset_name = Path(dataset_path).name
    repo_id = repo_id or f"local/{dataset_name}"

    python_argv, extra_env = resolve_repo_python()
    env = os.environ.copy()
    env.update(extra_env)

    proc = subprocess.run(
        python_argv + [VALIDATE_DATASET_SCRIPT, "--dataset-path", dataset_path, "--repo-id", repo_id],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Dataset validation failed: {(proc.stderr or proc.stdout).strip()}")
        
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise RuntimeError(f"Could not parse validation output: {proc.stdout}\n{proc.stderr}") from e
        
    result["size_gb"] = round(get_dir_size(dataset_path) / (1024**3), 2)
    return result


# ── DATASET CONVERSION PIPELINE ───────────────────────────────────────────────
# The UI's combined converter prepares metadata/statistics, builds and validates the
# LeRobot v3.0 layout in a sibling directory, then promotes it while temporarily
# keeping the source at "<dataset>_old" until the promoted dataset passes verification.
# A legacy converter remains for SmolVLA's already-normalized
# intermediate layout (see _prepare_and_train).
DATASET_V30_SCRIPT = f"{UI_ROOT}/v3_convertor.py"
DATASET_V30_LEGACY_SCRIPT = f"{DAKSHA_ROOT}/convert_daksha_dataset_to_v30.py"
dataset_conversion_lock = threading.Lock()


def _dataset_is_v30_ready(dataset_path: Path) -> bool:
    """True if `dataset_path` already looks like a valid LeRobot v3.0 dataset."""
    info_path = dataset_path / "meta" / "info.json"
    tasks_parquet = dataset_path / "meta" / "tasks.parquet"
    episodes_dir = dataset_path / "meta" / "episodes"
    stats_path = dataset_path / "meta" / "stats.json"
    if not (info_path.is_file() and tasks_parquet.is_file() and stats_path.is_file() and episodes_dir.is_dir()):
        return False
    episode_files = sorted(episodes_dir.glob("*/*.parquet"))
    if not episode_files:
        return False
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        info = json.loads(info_path.read_text())
        stats = json.loads(stats_path.read_text())
        for episode_file in episode_files:
            schema = pq.read_schema(episode_file)
            if "tasks" not in schema.names or not pa.types.is_list(schema.field("tasks").type):
                return False
    except (json.JSONDecodeError, OSError):
        return False
    return info.get("codebase_version") == "v3.0" and isinstance(stats, dict) and bool(stats)


def _repair_v30_episode_tasks(dataset_path: Path, log_path: str) -> bool:
    """Migrate older UI-generated v3.0 episode files from scalar `task` to `tasks: list[str]`."""
    import pyarrow.parquet as pq

    episode_files = sorted((dataset_path / "meta" / "episodes").glob("*/*.parquet"))
    if not episode_files:
        return False

    repaired_files = 0
    for episode_file in episode_files:
        columns = pq.read_schema(episode_file).names
        if "tasks" in columns:
            continue
        if "task" not in columns:
            raise RuntimeError(
                f"Episode metadata has neither the required 'tasks' column nor a legacy 'task' column: {episode_file}"
            )

        episodes = pd.read_parquet(episode_file)
        episodes["tasks"] = episodes.pop("task").map(
            lambda task: [] if task is None or pd.isna(task) else [str(task)]
        )
        temporary_file = episode_file.with_name(f".{episode_file.name}.{uuid.uuid4().hex}.tmp.parquet")
        try:
            episodes.to_parquet(temporary_file, index=False)
            temporary_file.replace(episode_file)
        finally:
            temporary_file.unlink(missing_ok=True)
        repaired_files += 1

    if repaired_files:
        log_message(log_path, f"Repaired LeRobot episode task metadata in {repaired_files} parquet file(s).")
    return repaired_files > 0


def _run_conversion_step(cmd, env, cwd, log_path, register_process=None, progress_callback=None):
    """Log conversion output to disk and tail it for progress callbacks."""
    with open(log_path, "a+", encoding="utf-8", buffering=1) as child_log:
        child_log.seek(0, os.SEEK_END)
        start_offset = child_log.tell()
        proc = subprocess.Popen(
            cmd, env=env, cwd=cwd, stdout=child_log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if register_process:
        register_process(proc)
    try:
        if progress_callback:
            with open(log_path, "r", encoding="utf-8", errors="replace") as progress_log:
                progress_log.seek(start_offset)
                while proc.poll() is None:
                    line = progress_log.readline()
                    if line:
                        progress_callback(line.rstrip())
                    else:
                        time.sleep(0.1)
                for line in progress_log:
                    progress_callback(line.rstrip())
        returncode = proc.wait()
    finally:
        if register_process:
            register_process(None)
    if returncode != 0:
        raise RuntimeError(f"Command failed (exit {returncode}): {' '.join(cmd)}")


def _archive_dataset_conversion_dir(path: Path, label: str, log_path: str) -> Path:
    """Move a stale converter-owned directory aside without deleting user data."""
    archived = path.with_name(f".{path.name}.{label}.{uuid.uuid4().hex[:10]}")
    shutil.move(str(path), str(archived))
    log_message(log_path, f"Preserved stale dataset conversion data at {archived}")
    return archived


def _ensure_dataset_v30(dataset_path_str: str, log_path: str, register_process=None, progress_callback=None) -> dict:
    """Convert a Daksha dataset in place into real LeRobot v3.0 layout, if it isn't already.

    Idempotent: returns immediately (no-op) if the dataset already looks like valid v3.0.
    The combined UI converter prepares metadata and statistics, writes a validated v3.0
    dataset beside the source, and only then replaces the source. The original contents
    stay at "<dataset>_old" until the promoted dataset passes its final readiness check.
    """
    dataset_path = Path(dataset_path_str)
    if _dataset_is_v30_ready(dataset_path):
        log_message(log_path, f"Dataset already in LeRobot v3.0 format: {dataset_path}")
        return {"status": "success", "converted": False}

    old_backup = dataset_path.parent / f"{dataset_path.name}_old"
    converted_path = dataset_path.parent / f"{dataset_path.name}_v30"

    # Recover the original directory when a previous run was interrupted after
    # moving it to the temporary backup but before restoring/promoting a result.
    if not dataset_path.exists() and old_backup.is_dir():
        shutil.move(str(old_backup), str(dataset_path))
        log_message(log_path, f"Restored dataset source from an interrupted preparation: {dataset_path}")
        if _dataset_is_v30_ready(dataset_path):
            log_message(log_path, f"Dataset already in LeRobot v3.0 format: {dataset_path}")
            return {"status": "success", "converted": False}

    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(
            f"Dataset at {dataset_path} has no meta/info.json — this doesn't look like a "
            "Daksha-recorded dataset and can't be auto-converted."
        )

    info = json.loads(info_path.read_text())
    if info.get("codebase_version") == "v3.0":
        repaired = _repair_v30_episode_tasks(dataset_path, log_path)
        if _dataset_is_v30_ready(dataset_path):
            log_message(log_path, f"Dataset already in LeRobot v3.0 format: {dataset_path}")
            return {"status": "success", "converted": False, "repaired": repaired}

    # Preserve stale converter-owned folders instead of failing or accidentally
    # treating an old staging output as the current dataset. Then v3_convertor.py
    # gets a clean output path for the source that failed the readiness check.
    if old_backup.exists():
        _archive_dataset_conversion_dir(old_backup, "old-recovery", log_path)
    if converted_path.exists():
        _archive_dataset_conversion_dir(converted_path, "v30-incomplete", log_path)

    python_argv, extra_env = resolve_repo_python()
    env = os.environ.copy()
    env.update(extra_env)
    env["PYTHONUNBUFFERED"] = "1"
    fps = int(info.get("fps") or 30)
    if fps <= 0:
        raise RuntimeError(f"Dataset FPS must be positive, got {fps}.")

    log_message(log_path, "Converting dataset to LeRobot v3.0 and generating stats.json...")
    _run_conversion_step(
        python_argv + [
            DATASET_V30_SCRIPT,
            "--root", str(dataset_path),
            "--output", str(converted_path),
            "--fps", str(fps),
        ],
        env,
        REPO_ROOT,
        log_path,
        register_process,
        progress_callback,
    )

    if not _dataset_is_v30_ready(converted_path):
        raise RuntimeError(
            "v3.0 conversion finished but the output is missing required metadata, "
            "episode files, or a nonempty stats.json."
        )

    shutil.move(str(dataset_path), str(old_backup))
    try:
        shutil.move(str(converted_path), str(dataset_path))
    except Exception:
        if not dataset_path.exists() and old_backup.exists():
            shutil.move(str(old_backup), str(dataset_path))
        raise

    if not _dataset_is_v30_ready(dataset_path):
        shutil.move(str(dataset_path), str(converted_path))
        shutil.move(str(old_backup), str(dataset_path))
        raise RuntimeError("Promoted dataset failed the LeRobot v3.0 readiness check; original dataset restored.")

    # Keep the source backup through conversion and promotion checks so failures can
    # restore the original. Remove it only once the final in-place dataset is verified.
    shutil.rmtree(old_backup)
    log_message(log_path, f"Removed pre-conversion backup after successful verification: {old_backup}")
    log_message(log_path, f"Dataset ready in LeRobot v3.0 format: {dataset_path}")
    return {"status": "success", "converted": True}


def ensure_dataset_v30(dataset_path_str: str, log_path: str, register_process=None, progress_callback=None) -> dict:
    """Serialize conversion requests so Train and Validate cannot race on one dataset."""
    with dataset_conversion_lock:
        return _ensure_dataset_v30(dataset_path_str, log_path, register_process, progress_callback)


@app.route('/api/datasets/import', methods=['POST'])
def import_dataset():
    """Import a raw Daksha dataset from a server path into DATASET_ROOT, convert it, and validate it.

    JSON body:
      - source_path: absolute path to raw dataset folder on the server
      - dest_name: optional name to use under DATASET_ROOT (defaults to basename of source)
      - modality_template: optional path to modality.json template
      - chunks_size: optional int
    """
    data = request.json or {}
    source = data.get('source_path')
    if not source:
        return jsonify({"status": "error", "message": "source_path is required"}), 400

    source_path = Path(source).expanduser().resolve()
    if not source_path.exists() or not source_path.is_dir():
        return jsonify({"status": "error", "message": f"Source path not found: {source_path}"}), 404

    # .name strips any directory components (leading "/", "../", etc.) so an
    # attacker-controlled dest_name can't escape DATASET_ROOT -- Path's "/" operator
    # silently discards the left side entirely when the right side is absolute, and a
    # raw ".." segment would otherwise walk out of DATASET_ROOT once resolved.
    dest_name = Path(data.get('dest_name') or source_path.name).name
    if not dest_name or dest_name in ('.', '..'):
        return jsonify({"status": "error", "message": "Invalid dest_name"}), 400
    dest_path = Path(DATASET_ROOT) / dest_name
    allowed_root = Path(DATASET_ROOT).resolve()
    if dest_path.resolve() != allowed_root and allowed_root not in dest_path.resolve().parents:
        return jsonify({"status": "error", "message": "dest_name escapes the dataset root"}), 400
    if dest_path.exists():
        dest_path = Path(str(dest_path) + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

    try:
        log_message(TRAIN_LOG, f"Importing dataset from {source_path} to {dest_path}")
        shutil.copytree(str(source_path), str(dest_path))
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to copy dataset: {e}"}), 500

    # Run the fused metadata-normalize + v3.0 conversion pipeline
    try:
        conv_result = ensure_dataset_v30(str(dest_path), TRAIN_LOG)
    except Exception as e:
        log_message(TRAIN_LOG, f"Conversion failed: {e}")
        return jsonify({"status": "error", "message": f"Conversion failed: {e}"}), 500
    log_message(TRAIN_LOG, f"Conversion result: {conv_result.get('status')}")

    # Validate converted dataset
    try:
        validation = validate_dataset_with_lerobot(str(dest_path), repo_id=f"local/{dest_path.name}")
    except Exception as e:
        return jsonify({"status": "error", "message": f"Validation error: {e}"}), 500

    return jsonify({"status": "success", "dataset_id": dest_path.name, "validation": validation})

# ════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - Dashboard
# ════════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve main UI."""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system and training status."""
    with _tegra_stats_lock:
        gpu_percent = _tegra_stats["gpu_percent"]
        vram_used_mb = _tegra_stats["ram_used_mb"]
        vram_total_mb = _tegra_stats["ram_total_mb"]
    vram_percent = (
        (vram_used_mb / vram_total_mb * 100) if vram_used_mb and vram_total_mb else None
    )
    return jsonify({
        "training": {
            "active": bool(_managed_process_record("training")) or global_state["training_active"],
            "policy": global_state["training_policy"],
            "current_step": global_state["current_step"],
            "total_steps": global_state["total_steps"],
        },
        "inference": {
            "active": global_state["inference_active"],
            "server_running": bool(_managed_process_record("inference")) or (global_state["server_process"] is not None and global_state["server_process"].poll() is None),
            "server_port": global_state["server_port"] or ((_managed_process_record("inference") or {}).get("port")),
            "bridge_active": bool(_managed_process_record("bridge")) or global_state["bridge_active"],
            "bridge_running": bool(_managed_process_record("bridge")) or (global_state["bridge_process"] is not None and global_state["bridge_process"].poll() is None),
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "gpu_percent": gpu_percent,
            "vram_percent": vram_percent,
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
        }
    })

@app.route('/api/policies', methods=['GET'])
def get_policies():
    """Get available policy types."""
    policies = {}
    for key, config in POLICY_CONFIGS.items():
        policies[key] = {
            "name": config["name"],
            "description": config["description"],
            "features": config["features"],
            # NOTE: keep "type" in what's sent to the frontend -- static/app.js's
            # updateHyperparameters() switches rendering (checkbox/text/number) on it.
            "hyperparameters": config["hyperparameters"],
            "modality_contract": {
                "state_dim": DAKSHA_ARM_CONFIG["state"]["state_dim"],
                "action_dim": DAKSHA_ARM_CONFIG["state"]["state_dim"],
                "state_modality_keys": DAKSHA_ARM_CONFIG["state"]["modality_keys"],
                "action_modality_keys": DAKSHA_ARM_CONFIG["action"]["modality_keys"],
                "camera_keys": DEFAULT_CAMERA_KEYS,
                "language_key": DAKSHA_ARM_CONFIG["language"]["task_description_key"],
            },
        }
    return jsonify(policies)

# ════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - Dataset Management
# ════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List available datasets."""
    datasets = []
    if os.path.exists(DATASET_ROOT):
        for item in os.listdir(DATASET_ROOT):
            # "<name>_old" / "<name>_v30" are backup/scratch dirs the conversion pipeline
            # itself creates (see ensure_dataset_v30) -- not real selectable datasets.
            if item.startswith(".") or item.endswith("_old") or item.endswith("_v30"):
                continue
            item_path = os.path.join(DATASET_ROOT, item)
            if os.path.isdir(item_path):
                try:
                    total_size = sum(os.path.getsize(os.path.join(dirpath, f)) 
                                   for dirpath, dirnames, filenames in os.walk(item_path) 
                                   for f in filenames)
                    datasets.append({
                        "name": item,
                        "dataset_id": item,
                        "path": item_path,
                        "size_gb": round(total_size / (1024**3), 2),
                        "created": datetime.fromtimestamp(os.path.getctime(item_path)).isoformat(),
                    })
                except Exception as e:
                    print(f"Error analyzing dataset {item}: {e}")
    return jsonify(datasets)


# ── Read-only dataset viewer API ─────────────────────────────────────────────
# Viewer requests resolve a catalog ID to one direct child of DATASET_ROOT. They
# never accept an arbitrary filesystem path, even though the older mutation APIs
# also accept backend-resolved paths from the catalog.
viewer_metadata_cache_lock = threading.RLock()
viewer_metadata_cache = {}


def _viewer_dataset_root(dataset_id):
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id or dataset_id in {".", ".."} or Path(dataset_id).name != dataset_id or "/" in dataset_id or "\\" in dataset_id:
        raise ValueError("Invalid dataset ID.")
    if dataset_id.endswith("_old") or dataset_id.endswith("_v30"):
        raise FileNotFoundError(f"Dataset not found: {dataset_id}")
    root = Path(DATASET_ROOT).expanduser().resolve()
    candidate = (root / dataset_id).resolve()
    if candidate.parent != root:
        raise PermissionError("Dataset path is outside the configured dataset root.")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_id}")
    return candidate


def _viewer_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_viewer_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_viewer_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _viewer_jsonable(item) for key, item in value.items()}
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _viewer_read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _viewer_read_info(root):
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError("This folder has no LeRobot meta/info.json file.")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read dataset metadata: {exc}") from exc
    if not isinstance(info, dict):
        raise ValueError("Dataset info.json must contain an object.")
    return info


def _viewer_episode_metadata_rows(root):
    parquet_root = root / "meta" / "episodes"
    parquet_files = sorted(parquet_root.rglob("*.parquet")) if parquet_root.is_dir() else []
    if parquet_files:
        try:
            frames = [pd.read_parquet(path) for path in parquet_files]
            if frames:
                frame = pd.concat(frames, ignore_index=True)
                return [
                    {str(key): _viewer_jsonable(value) for key, value in row.items()}
                    for row in frame.to_dict(orient="records")
                ]
        except Exception as exc:
            raise ValueError(f"Could not read episode metadata Parquet: {exc}") from exc
    jsonl_path = root / "meta" / "episodes.jsonl"
    if jsonl_path.is_file():
        try:
            return _viewer_read_jsonl(jsonl_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read episode metadata JSONL: {exc}") from exc
    raise ValueError("No supported episode metadata was found (expected meta/episodes Parquet or episodes.jsonl).")


def _viewer_task_map(root):
    tasks = {}
    parquet_path = root / "meta" / "tasks.parquet"
    if parquet_path.is_file():
        try:
            rows = pd.read_parquet(parquet_path).to_dict(orient="records")
            for row in rows:
                index = row.get("task_index")
                text = row.get("task") or row.get("prompt")
                if index is not None and text:
                    tasks[int(index)] = str(text)
        except Exception:
            pass
    jsonl_path = root / "meta" / "tasks.jsonl"
    if jsonl_path.is_file() and not tasks:
        try:
            for row in _viewer_read_jsonl(jsonl_path):
                if row.get("task_index") is not None:
                    tasks[int(row["task_index"])] = str(row.get("task") or row.get("prompt") or "")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return tasks


def _viewer_field(row, group, camera_key, field):
    nested = row.get(group)
    if isinstance(nested, dict):
        camera = nested.get(camera_key)
        if isinstance(camera, dict) and field in camera:
            return camera.get(field)
    for key in (f"{group}/{camera_key}/{field}", f"{group}.{camera_key}.{field}"):
        if key in row:
            return row.get(key)
    return None


def _viewer_group_field(row, group, field):
    nested = row.get(group)
    if isinstance(nested, dict) and field in nested:
        return nested.get(field)
    for key in (f"{group}/{field}", f"{group}.{field}"):
        if key in row:
            return row.get(key)
    return None


def _viewer_int(value, default=None):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _viewer_float(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _viewer_episode_index(row):
    return _viewer_int(row.get("episode_index", row.get("episode")))


def _viewer_episode_length(row):
    return _viewer_int(row.get("length", row.get("frame_count", row.get("frames"))), 0) or 0


def _viewer_episode_task(row, task_map):
    task = row.get("task") or row.get("prompt") or row.get("task_description")
    if isinstance(task, (list, tuple)):
        task = task[0] if task else ""
    if not task:
        tasks = row.get("tasks")
        if isinstance(tasks, (list, tuple)) and tasks:
            task = tasks[0]
    if not task:
        task_index = _viewer_int(row.get("task_index"))
        if task_index is None and isinstance(row.get("tasks"), (list, tuple)):
            task_index = _viewer_int(row["tasks"][0]) if row["tasks"] else None
        task = task_map.get(task_index, "")
    return str(task or "")


def _viewer_camera_keys(info):
    raw = info.get("camera_keys") or info.get("video_keys") or []
    keys = [str(key) for key in raw if isinstance(key, str)]
    features = info.get("features") or {}
    if isinstance(features, dict):
        for key, feature in features.items():
            if isinstance(feature, dict) and (feature.get("dtype") == "video" or str(key).startswith("observation.images.")):
                if str(key) not in keys:
                    keys.append(str(key))
    return keys


def _viewer_camera_label(camera_key):
    camera = camera_key.replace("observation.images.", "")
    lowered = camera.lower()
    if "wrist_left" in lowered or "left_wrist" in lowered:
        return "Left wrist camera"
    if "wrist_right" in lowered or "right_wrist" in lowered:
        return "Right wrist camera"
    if "world" in lowered or "center" in lowered:
        return "World center camera"
    return camera.replace("_", " ").replace(".", " · ").title()


def _viewer_action_info(info):
    feature = (info.get("features") or {}).get("action", {})
    names = feature.get("names") if isinstance(feature, dict) else None
    if not isinstance(names, list) or not names:
        names = []
    shape = feature.get("shape") if isinstance(feature, dict) else None
    dimension = _viewer_int(shape[0], len(names)) if isinstance(shape, list) and shape else len(names)
    dimension = max(0, dimension or len(names))
    if len(names) < dimension:
        names = names + [f"Action {index + 1}" for index in range(len(names), dimension)]
    return names[:dimension], dimension


def _viewer_safe_child(root, relative):
    try:
        candidate = (root / str(relative)).resolve()
        if candidate == root or root not in candidate.parents:
            return None
        return candidate
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _viewer_camera_has_media(root, camera_key):
    camera_root = _viewer_safe_child(root, Path("videos") / camera_key)
    if camera_root is None or not camera_root.is_dir():
        return False
    try:
        return next(camera_root.rglob("*.mp4"), None) is not None
    except OSError:
        return False


def _viewer_indices(row, group):
    chunk = _viewer_int(_viewer_group_field(row, group, "chunk_index"))
    if chunk is None:
        chunk = _viewer_int(_viewer_group_field(row, group, "episode_chunk"))
    file_index = _viewer_int(_viewer_group_field(row, group, "file_index"))
    return chunk, file_index


def _viewer_format_template(root, template, values):
    if not isinstance(template, str) or not template:
        return None
    try:
        relative = template.format(**values)
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    return _viewer_safe_child(root, relative)


def _viewer_media_record(root, info, row, camera_key, fps):
    episode_index = _viewer_episode_index(row)
    if episode_index is None:
        return {"camera_key": camera_key, "available": False, "error": "Episode index is missing."}
    chunks_size = _viewer_int(info.get("chunks_size"), 1000) or 1000
    episode_chunk = _viewer_int(_viewer_group_field(row, "data", "episode_chunk"), episode_index // chunks_size)
    data_chunk, data_file = _viewer_indices(row, "data")
    camera_chunk, camera_file = _viewer_indices(row, "videos",)
    camera_chunk = _viewer_int(_viewer_field(row, "videos", camera_key, "chunk_index"), camera_chunk)
    camera_file = _viewer_int(_viewer_field(row, "videos", camera_key, "file_index"), camera_file)
    chunk_index = camera_chunk if camera_chunk is not None else data_chunk if data_chunk is not None else episode_chunk
    file_index = camera_file if camera_file is not None else data_file
    values = {
        "video_key": camera_key, "camera_key": camera_key,
        "episode_index": episode_index, "episode_chunk": episode_chunk,
        "chunk_index": chunk_index, "file_index": file_index if file_index is not None else episode_index,
        "video_chunk_index": chunk_index, "video_file_index": file_index if file_index is not None else episode_index,
        "chunks_size": chunks_size,
    }
    candidates = []
    templated = _viewer_format_template(root, info.get("video_path"), values)
    if templated is not None:
        candidates.append(templated)
    camera_root = _viewer_safe_child(root, Path("videos") / camera_key)
    if camera_root is not None:
        chunks = {episode_chunk, chunk_index}
        if camera_chunk is not None:
            chunks.add(camera_chunk)
        filenames = {f"episode_{episode_index:06d}.mp4"}
        if file_index is not None:
            filenames.add(f"file-{file_index:03d}.mp4")
        for chunk in chunks:
            if chunk is None:
                continue
            for filename in filenames:
                candidates.append(camera_root / f"chunk-{chunk:03d}" / filename)
        try:
            exact_episode_files = camera_root.glob(f"chunk-*/episode_{episode_index:06d}.mp4")
            candidates.extend(exact_episode_files)
        except OSError:
            pass
    media_path = next((candidate for candidate in candidates if candidate and candidate.is_file() and root in candidate.resolve().parents), None)
    start = _viewer_float(_viewer_field(row, "videos", camera_key, "from_timestamp"), 0.0)
    end = _viewer_float(_viewer_field(row, "videos", camera_key, "to_timestamp"))
    duration = _viewer_episode_length(row) / fps if fps > 0 else 0.0
    if end is None or end <= start:
        end = start + duration
    available = media_path is not None and end > start
    record = {
        "camera_key": camera_key,
        "label": _viewer_camera_label(camera_key),
        "available": bool(available),
        "start_time": start,
        "end_time": end,
    }
    if available:
        record["_resolved_path"] = str(media_path.resolve())
        record["url"] = url_for("dataset_viewer_video", dataset_id=root.name, episode_index=episode_index, camera_key=camera_key)
    else:
        record["error"] = "No matching video segment is available for this episode."
    return record


def _viewer_dataset_signature(root):
    meta = root / "meta"
    files = [meta / "info.json", meta / "tasks.parquet", meta / "tasks.jsonl", meta / "episodes.jsonl"]
    episodes_dir = meta / "episodes"
    if episodes_dir.is_dir():
        files.extend(episodes_dir.rglob("*.parquet"))
    signature = []
    for path in sorted(set(files)):
        try:
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            continue
    return tuple(signature)


def _viewer_dataset_context(dataset_id):
    root = _viewer_dataset_root(dataset_id)
    cache_key = str(root)
    signature = _viewer_dataset_signature(root)
    with viewer_metadata_cache_lock:
        cached = viewer_metadata_cache.get(cache_key)
        if cached and cached["signature"] == signature:
            return root, cached["info"], cached["rows"], cached["task_map"]

        info = _viewer_read_info(root)
        rows = _viewer_episode_metadata_rows(root)
        rows = sorted(
            (row for row in rows if _viewer_episode_index(row) is not None),
            key=lambda row: _viewer_episode_index(row),
        )
        task_map = _viewer_task_map(root)
        episode_map = {_viewer_episode_index(row): row for row in rows}
        viewer_metadata_cache[cache_key] = {
            "signature": signature, "info": info, "rows": rows,
            "task_map": task_map, "episodes_by_index": episode_map,
        }
        while len(viewer_metadata_cache) > 6:
            viewer_metadata_cache.pop(next(iter(viewer_metadata_cache)))
        return root, info, rows, task_map


def _viewer_find_episode_row(root, episode_index):
    _viewer_dataset_context(root.name)
    with viewer_metadata_cache_lock:
        cached = viewer_metadata_cache.get(str(root))
        if cached:
            return cached["episodes_by_index"].get(int(episode_index))
    return None


def _viewer_episode_payload(root, info, row, task_map):
    fps = _viewer_float(info.get("fps"), 0.0) or 0.0
    length = _viewer_episode_length(row)
    task = _viewer_episode_task(row, task_map)
    task_index = _viewer_int(row.get("task_index"))
    if task_index is None and isinstance(row.get("tasks"), (list, tuple)) and row["tasks"]:
        task_index = _viewer_int(row["tasks"][0])
    videos = [_viewer_media_record(root, info, row, key, fps) for key in _viewer_camera_keys(info)]
    for video in videos:
        video.pop("_resolved_path", None)
    return {
        "episode_index": _viewer_episode_index(row),
        "task_index": task_index,
        "task": task,
        "prompt": task,
        "frame_count": length,
        "duration_s": length / fps if fps > 0 else None,
        "fps": fps or None,
        "videos": videos,
    }


@app.route('/api/datasets/<dataset_id>/viewer', methods=['GET'])
def dataset_viewer_metadata(dataset_id):
    try:
        root, info, episode_rows, task_map = _viewer_dataset_context(dataset_id)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Could not inspect dataset metadata: {exc}"}), 500

    fps = _viewer_float(info.get("fps"))
    action_names, action_dim = _viewer_action_info(info)
    camera_keys = _viewer_camera_keys(info)
    totals = {}
    for row in episode_rows:
        task = _viewer_episode_task(row, task_map)
        if task:
            totals[task] = totals.get(task, 0) + 1
    tasks = [{"task_index": index, "prompt": prompt, "count": count} for index, (prompt, count) in enumerate(sorted(totals.items()))]
    if not tasks:
        tasks = [{"task_index": index, "prompt": task, "count": 0} for index, task in sorted(task_map.items())]
    total_frames = _viewer_int(info.get("total_frames"))
    if total_frames is None:
        total_frames = sum(_viewer_episode_length(row) for row in episode_rows)
    media_features = info.get("features") or {}
    camera_dimensions = {}
    for key in camera_keys:
        feature = media_features.get(key, {}) if isinstance(media_features, dict) else {}
        shape = feature.get("shape") if isinstance(feature, dict) else None
        camera_dimensions[key] = shape
    return jsonify({
        "dataset_id": root.name,
        "version": info.get("codebase_version") or info.get("version") or "unknown",
        "layout": "v3" if (root / "meta" / "episodes").is_dir() else "legacy episode layout",
        "fps": fps,
        "robot_type": info.get("robot_type"),
        "total_episodes": _viewer_int(info.get("total_episodes"), len(episode_rows)),
        "total_frames": total_frames,
        "total_tasks": _viewer_int(info.get("total_tasks"), len(tasks)),
        "tasks": tasks,
        "action_dim": action_dim,
        "action_names": action_names,
        "features": list(media_features.keys()) if isinstance(media_features, dict) else [],
        "cameras": [{
            "key": key,
            "label": _viewer_camera_label(key),
            "available": _viewer_camera_has_media(root, key),
            "shape": camera_dimensions.get(key),
        } for key in camera_keys],
    })


@app.route('/api/datasets/<dataset_id>/episodes', methods=['GET'])
def dataset_viewer_episodes(dataset_id):
    try:
        root, info, rows, task_map = _viewer_dataset_context(dataset_id)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Could not read episode metadata: {exc}"}), 500
    task_filter = request.args.get("task", "").strip()
    if task_filter:
        rows = [row for row in rows if _viewer_episode_task(row, task_map) == task_filter]
    try:
        offset = int(request.args.get("offset", "0"))
        limit = int(request.args.get("limit", "300"))
    except (TypeError, ValueError):
        return jsonify({"error": "offset and limit must be integers"}), 400
    if offset < 0:
        return jsonify({"error": "offset must be a nonnegative integer"}), 400
    if limit < 1:
        return jsonify({"error": "limit must be a positive integer"}), 400
    limit = min(limit, 1000)
    page_rows = rows[offset:offset + limit]
    episodes = [_viewer_episode_payload(root, info, row, task_map) for row in page_rows]
    next_offset = offset + len(page_rows)
    return jsonify({"episodes": episodes, "total": len(rows), "offset": offset, "limit": limit, "next_offset": next_offset})


def _viewer_data_files(root, info, row, episode_index):
    chunks_size = _viewer_int(info.get("chunks_size"), 1000) or 1000
    episode_chunk = _viewer_int(_viewer_group_field(row, "data", "episode_chunk"), episode_index // chunks_size)
    data_chunk, data_file = _viewer_indices(row, "data")
    chunk = data_chunk if data_chunk is not None else episode_chunk
    file_index = data_file if data_file is not None else episode_index // chunks_size
    values = {
        "episode_index": episode_index, "episode_chunk": episode_chunk,
        "chunk_index": chunk, "file_index": file_index,
        "chunks_size": chunks_size,
    }
    candidates = []
    templated = _viewer_format_template(root, info.get("data_path"), values)
    if templated is not None and templated.is_file() and root in templated.resolve().parents:
        return [templated.resolve()]
    if templated is not None:
        candidates.append(templated)
    data_root = root / "data"
    if data_root.is_dir():
        if chunk is not None:
            candidates.extend((data_root / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet", data_root / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"))
        candidates.extend(data_root.glob(f"chunk-*/episode_{episode_index:06d}.parquet"))
        candidates.extend(data_root.glob(f"**/episode_{episode_index}.parquet"))
    result = []
    for path in candidates:
        if path is None or not path.is_file():
            continue
        resolved = path.resolve()
        if root not in resolved.parents or resolved in result:
            continue
        result.append(resolved)
    return result


@app.route('/api/datasets/<dataset_id>/actions', methods=['GET'])
def dataset_viewer_actions(dataset_id):
    raw_episode = request.args.get("episode", "")
    try:
        episode_index = int(raw_episode)
        if episode_index < 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "episode must be a nonnegative episode index"}), 400
    try:
        root = _viewer_dataset_root(dataset_id)
        info = _viewer_read_info(root)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Could not read episode metadata: {exc}"}), 500
    try:
        episode_row = _viewer_find_episode_row(root, episode_index)
    except Exception as exc:
        return jsonify({"error": f"Could not read episode metadata: {exc}"}), 500
    if episode_row is None:
        return jsonify({"error": f"Episode {episode_index} was not found in dataset {dataset_id}."}), 404
    files = _viewer_data_files(root, info, episode_row, episode_index)
    if not files:
        return jsonify({"error": "No Parquet data file could be resolved for this episode."}), 404

    try:
        import pyarrow.parquet as pq
        samples = []
        for path in files:
            schema = pq.read_schema(path)
            columns = [name for name in ("episode_index", "frame_index", "timestamp", "action") if name in schema.names]
            if "action" not in columns:
                continue
            is_episode_file = path.name in {f"episode_{episode_index:06d}.parquet", f"episode_{episode_index}.parquet"}
            if "episode_index" not in columns and not is_episode_file:
                return jsonify({"error": "This Parquet layout has no episode index column; the selected episode cannot be isolated safely."}), 422
            filters = [("episode_index", "=", episode_index)] if "episode_index" in columns else None
            table = pq.read_table(path, columns=columns, filters=filters)
            for ordinal, row in enumerate(table.to_pylist()):
                row_episode = _viewer_int(row.get("episode_index"), episode_index)
                if row_episode != episode_index:
                    continue
                action = _viewer_jsonable(row.get("action"))
                if not isinstance(action, list):
                    try:
                        action = list(action)
                    except TypeError:
                        continue
                values = [_viewer_float(value) for value in action]
                if any(value is None for value in values):
                    continue
                frame_index = _viewer_int(row.get("frame_index"), ordinal)
                timestamp = _viewer_float(row.get("timestamp"))
                samples.append({"frame_index": frame_index, "timestamp": timestamp, "action": values})
        if not samples:
            return jsonify({"error": "No action rows were found for the selected episode."}), 422
        fps = _viewer_float(info.get("fps"), 0.0) or 0.0
        first_recorded_timestamp = min(
            (item["timestamp"] for item in samples if item["timestamp"] is not None),
            default=None,
        )
        for ordinal, sample in enumerate(samples):
            if sample["frame_index"] is None:
                sample["frame_index"] = ordinal
            if sample["timestamp"] is None:
                frame_time = sample["frame_index"] / fps if fps > 0 else float(ordinal)
                sample["timestamp"] = first_recorded_timestamp + frame_time if first_recorded_timestamp is not None else frame_time
        # The client uses binary nearest-time lookup, so normalize all missing
        # timestamps first, then return a strictly time-ordered sample series.
        samples.sort(key=lambda item: (item["timestamp"], item["frame_index"]))
        first_timestamp = samples[0]["timestamp"] if samples else None
        for sample in samples:
            sample["episode_time"] = max(0.0, sample["timestamp"] - first_timestamp) if first_timestamp is not None else sample["timestamp"]
        max_samples = 2500
        stride = max(1, math.ceil(len(samples) / max_samples))
        selected_samples = samples[::stride]
        if selected_samples[-1] is not samples[-1]:
            selected_samples.append(samples[-1])
        action_names, action_dim = _viewer_action_info(info)
        sample_dim = len(selected_samples[0]["action"])
        if sample_dim != action_dim:
            action_dim = sample_dim
            action_names = (action_names + [f"action_{index + 1}" for index in range(len(action_names), action_dim)])[:action_dim]
        if any(len(item["action"]) != action_dim for item in selected_samples):
            return jsonify({"error": "Action dimension changes within the selected episode."}), 422
        return jsonify({
            "episode_index": episode_index,
            "action_names": action_names,
            "action_dim": action_dim,
            "sample_count": len(samples),
            "sample_stride": stride,
            "duration_s": _viewer_episode_length(episode_row) / fps if fps > 0 else None,
            "samples": selected_samples,
        })
    except ImportError:
        return jsonify({"error": "PyArrow is required to read dataset action Parquet files."}), 501
    except Exception as exc:
        return jsonify({"error": f"Could not read action samples: {exc}"}), 500


@app.route('/api/datasets/<dataset_id>/video/<int:episode_index>/<camera_key>', methods=['GET'])
def dataset_viewer_video(dataset_id, episode_index, camera_key):
    try:
        root = _viewer_dataset_root(dataset_id)
        info = _viewer_read_info(root)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Could not inspect dataset metadata: {exc}"}), 500
    if camera_key not in _viewer_camera_keys(info):
        return jsonify({"error": "Camera key is not advertised by this dataset."}), 404
    try:
        episode_row = _viewer_find_episode_row(root, episode_index)
    except Exception as exc:
        return jsonify({"error": f"Could not read episode metadata: {exc}"}), 500
    if episode_row is None:
        return jsonify({"error": f"Episode {episode_index} was not found in dataset {dataset_id}."}), 404
    media = _viewer_media_record(root, info, episode_row, camera_key, _viewer_float(info.get("fps"), 0.0) or 0.0)
    if not media.get("available"):
        return jsonify({"error": media.get("error", "Video media is unavailable.")}), 404
    media_path = Path(media.get("_resolved_path", "")).resolve()
    if not media_path.is_file() or root not in media_path.parents:
        return jsonify({"error": "Video media could not be resolved safely."}), 404
    return send_file(media_path, mimetype="video/mp4", conditional=True, etag=True, last_modified=media_path.stat().st_mtime)

@app.route('/api/validate_dataset', methods=['POST'])
def validate_dataset():
    """Validate dataset path and structure using the real LeRobot dataset validator."""
    data = request.json or {}
    repo_id = data.get('repo_id', '').strip()

    raw_path = data.get('path', '').strip() or data.get('dataset_id', '').strip()
    if not raw_path:
        return jsonify({"status": "error", "message": "No path provided."}), 400
    try:
        path = str(resolve_dataset_path(raw_path))
    except PermissionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 403
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": f"Path does not exist: {path}"}), 404
    if not os.path.isdir(path):
        return jsonify({"status": "error", "message": f"Path is not a directory: {path}"}), 400

    try:
        result = validate_dataset_with_lerobot(path, repo_id=repo_id or None)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Validation error: {str(e)}"}), 500

@app.route('/api/validation_ui/url', methods=['GET'])
def get_validation_ui_url():
    """Resolve which validator the "Validate" button should use for a dataset.

    Verify or convert the dataset to v3.0, then return its modular validator deep link.
    """
    dataset_id = request.args.get('dataset_id', '').strip()
    if not dataset_id:
        return jsonify({"status": "error", "message": "dataset_id is required"}), 400

    try:
        dataset_path = resolve_dataset_path(dataset_id)
    except PermissionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 403
    if not dataset_path.exists():
        return jsonify({"status": "error", "message": f"Dataset not found: {dataset_path}"}), 404

    try:
        if not _dataset_is_v30_ready(dataset_path):
            ensure_dataset_v30(str(dataset_path), TRAIN_LOG)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Dataset preparation failed: {exc}"}), 500

    target_path = dataset_path
    host = request.host.split(':')[0]
    port = ensure_validation_ui_running(frontend_host=host)
    if port is None:
        return jsonify({"status": "error", "message": "Validation UI failed to start; check the server log."}), 500

    api_port = global_state["validation_api_port"]
    return_url = quote(request.host_url.rstrip("/"), safe="")
    url = f"http://{host}:{port}/?dataset={quote(str(target_path))}&api={api_port}&return={return_url}"
    return jsonify({"status": "success", "mode": "rich", "url": url})


def _validation_prepare_progress(job_id, progress, message):
    with validation_prepare_lock:
        job = validation_prepare_jobs.get(job_id)
        if job and progress >= job.get("progress", 0):
            job.update(progress=min(100, int(progress)), message=message)


def _prepare_dataset_target(job_id, dataset_path, frontend_host, return_url, open_validator=True):
    log_path = f"{LOGS_ROOT}/validation_{job_id}.log"
    try:
        _validation_prepare_progress(job_id, 2, "Preparing dataset for validation…")
        info = json.loads((dataset_path / "meta" / "info.json").read_text())
        total_episodes = max(1, len(list(dataset_path.glob("data/chunk-*/episode_*.parquet"))))
        video_keys = sorted(
            key for key, value in (info.get("features") or {}).items()
            if isinstance(value, dict) and value.get("dtype") == "video"
        )
        completed_data = 0
        completed_video_groups = 0
        total_video_groups = max(1, len(video_keys))
        camera_group_counts = {}

        def conversion_progress(line):
            nonlocal completed_data, completed_video_groups, total_video_groups
            data_match = re.search(r"\[data\] episode ", line)
            if data_match:
                completed_data += 1
                pct = 5 + (25 * completed_data / total_episodes)
                _validation_prepare_progress(job_id, pct, "Preparing dataset for validation…")
                return
            group_match = re.search(r"\[video:([^]]+)\] group (\d+)/(\d+)", line)
            if group_match:
                key, _, groups = group_match.groups()
                camera_group_counts[key] = int(groups)
                max_groups = max(camera_group_counts.values(), default=1)
                total_video_groups = max(1, len(video_keys) * max_groups)
                return
            if "validated frames=" in line and "[video:" in line:
                completed_video_groups += 1
                pct = 30 + (55 * completed_video_groups / max(total_video_groups, completed_video_groups))
                _validation_prepare_progress(job_id, pct, "Preparing dataset for validation…")
            elif "FINAL DATASET VALIDATION" in line:
                _validation_prepare_progress(job_id, 90, "Finishing dataset checks…")

        if not _dataset_is_v30_ready(dataset_path):
            ensure_dataset_v30(
                str(dataset_path), log_path,
                progress_callback=conversion_progress,
            )
        _validation_prepare_progress(job_id, 86, "Finishing dataset checks…")
        if not _dataset_is_v30_ready(dataset_path):
            raise RuntimeError("Dataset did not pass the LeRobot v3.0 readiness check.")

        url = None
        if open_validator:
            _validation_prepare_progress(job_id, 92, "Opening validation tools…")
            port = ensure_validation_ui_running(frontend_host=frontend_host)
            if port is None:
                raise RuntimeError("Validation UI failed to start; check the server log.")
            api_port = global_state["validation_api_port"]
            url = f"http://{frontend_host}:{port}/?dataset={quote(str(dataset_path))}&api={api_port}&return={return_url}"
        _validation_prepare_progress(job_id, 100, "Dataset is ready.")
        with validation_prepare_lock:
            validation_prepare_jobs[job_id].update(status="complete", url=url)
    except Exception as exc:
        log_message(log_path, f"Validation preparation failed: {exc}")
        with validation_prepare_lock:
            if job_id in validation_prepare_jobs:
                validation_prepare_jobs[job_id].update(status="error", message=str(exc))


@app.route('/api/validation_ui/prepare', methods=['POST'])
def prepare_validation_ui():
    data = request.get_json(silent=True) or {}
    dataset_id = str(data.get("dataset_id") or "").strip()
    if not dataset_id:
        return jsonify({"status": "error", "message": "dataset_id is required"}), 400
    try:
        dataset_path = resolve_dataset_path(dataset_id)
    except PermissionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 403
    if not dataset_path.is_dir():
        return jsonify({"status": "error", "message": f"Dataset not found: {dataset_path}"}), 404

    job_id = uuid.uuid4().hex[:12]
    with validation_prepare_lock:
        validation_prepare_jobs[job_id] = {
            "job_id": job_id, "status": "preparing", "progress": 0,
            "message": "Preparing dataset for validation…", "url": None,
        }
    frontend_host = request.host.split(":")[0]
    return_url = quote(request.host_url.rstrip("/"), safe="")
    threading.Thread(
        target=_prepare_dataset_target,
        args=(job_id, dataset_path, frontend_host, return_url), daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route('/api/datasets/prepare', methods=['POST'])
def prepare_dataset_for_action():
    """Ensure a catalog dataset is in LeRobot v3.0 before viewing or validating it."""
    data = request.get_json(silent=True) or {}
    dataset_id = str(data.get("dataset_id") or "").strip()
    purpose = str(data.get("purpose") or "viewer").strip().lower()
    if purpose not in {"viewer", "validate"}:
        return jsonify({"status": "error", "message": "purpose must be viewer or validate"}), 400
    if not dataset_id:
        return jsonify({"status": "error", "message": "dataset_id is required"}), 400
    try:
        dataset_path = resolve_dataset_path(dataset_id)
    except PermissionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 403
    if not dataset_path.is_dir():
        return jsonify({"status": "error", "message": f"Dataset not found: {dataset_path}"}), 404

    job_id = uuid.uuid4().hex[:12]
    with validation_prepare_lock:
        validation_prepare_jobs[job_id] = {
            "job_id": job_id, "status": "preparing", "progress": 0,
            "message": "Preparing dataset…", "url": None, "dataset_id": dataset_id,
        }
    frontend_host = request.host.split(":")[0]
    return_url = quote(request.host_url.rstrip("/"), safe="")
    threading.Thread(
        target=_prepare_dataset_target,
        args=(job_id, dataset_path, frontend_host, return_url, purpose == "validate"), daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route('/api/datasets/prepare/<job_id>', methods=['GET'])
def dataset_prepare_status(job_id):
    with validation_prepare_lock:
        job = validation_prepare_jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "message": "Preparation job not found"}), 404
        return jsonify(job)


@app.route('/api/validation_ui/prepare/<job_id>', methods=['GET'])
def validation_prepare_status(job_id):
    with validation_prepare_lock:
        job = validation_prepare_jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "message": "Preparation job not found"}), 404
        return jsonify(job)

# ── MODEL VALIDATION: per-frame ACT/Diffusion/SmolVLA evaluation job manager ──
# Runs a whole-dataset evaluation (predicted vs. recorded actions) in a background
# thread and reports progress/results for the Evaluation tab's "Model Validation"
# card to poll and chart. Mirrors the training/inference job-tracking pattern
# above (global_state + a background thread) but keyed by job_id since a
# checkpoint/dataset pair, not a single global run, identifies each evaluation.
mv_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-validation")
mv_jobs: dict[str, dict] = {}
mv_jobs_lock = threading.RLock()


class ModelValidationCancelled(Exception):
    """Raised when a running model-validation evaluation is asked to stop."""


def validate_mv_checkpoint_path(checkpoint_value: str, model_type: str) -> Path:
    """Normalize and validate a checkpoint before starting a background job."""
    checkpoint = Path(str(checkpoint_value).strip().strip("\"'")).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint path must be a directory containing its model files: {checkpoint}")

    config_types = {
        "act": ACTCheckpointConfig,
        "diffusion": DiffusionCheckpointConfig,
        "smolvla": SmolVLACheckpointConfig,
    }
    config = config_types[model_type].from_checkpoint(checkpoint)
    config.validate()
    return checkpoint


def mv_dataset_episode_metadata(dataset_path: str) -> list[dict]:
    """Lightweight per-episode metadata (index/length/task), no video decoding."""
    dataset = LeRobotCustomDataset(dataset_path)
    try:
        return [{"episode_index": ep.episode_index, "length": ep.length, "task": ep.task} for ep in dataset.episodes]
    finally:
        dataset.close()


def compute_mv_test_episodes(dataset_path: str, checkpoint_path: str, episodes_meta: list[dict]) -> dict:
    """Reconstruct the saved split; a moved dataset remains explicitly unverified."""
    config = json.loads((Path(checkpoint_path) / "train_config.json").read_text())
    return checkpoint_episode_split(dataset_path, config["dataset"], episodes_meta)


def update_mv_job(job_id, **updates):
    with mv_jobs_lock:
        if job_id in mv_jobs:
            mv_jobs[job_id].update(updates)


def run_mv_evaluation(job_id: str, payload: dict) -> None:
    model = None
    try:
        update_mv_job(
            job_id, status="loading", progress=0,
            message=f"Loading dataset and {payload.get('model_type', 'act')} checkpoint...",
        )

        def progress(completed: int, total: int) -> None:
            with mv_jobs_lock:
                if mv_jobs.get(job_id, {}).get("cancel_requested"):
                    raise ModelValidationCancelled()
            update_mv_job(
                job_id, status="running", progress=int(round(completed * 100 / total)),
                message=f"Evaluating frame {completed}/{total}",
            )

        stride = max(1, int(payload.get("stride", 1)))
        max_frames = (
            int(payload["max_frames"])
            if payload.get("max_frames") not in (None, "", 0)
            else None
        )
        with mv_jobs_lock:
            if mv_jobs.get(job_id, {}).get("cancel_requested"):
                raise ModelValidationCancelled()
        model = load_evaluation_model(
            payload.get("model_type", "act"), payload["checkpoint"],
            payload.get("device", "cuda"), payload.get("inference_profile", "checkpoint"),
        )
        runtime = model.get_metadata().extra.get("runtime", {"profile": "checkpoint"})
        update_mv_job(job_id, runtime=runtime)
        progress(0, 1)

        if payload.get("scope") == "dataset":
            # Precomputed by start_model_validation() -- only the held-out test-split
            # episodes for this dataset/checkpoint pair, never the training episodes.
            episodes = payload["_test_episodes"]
            length_by_episode = payload["_episode_lengths"]
            total_frames = sum(
                min((length_by_episode[idx] + stride - 1) // stride, max_frames or length_by_episode[idx])
                for idx in episodes
            )

            completed_frames = 0
            episode_results = []

            def dataset_progress(done, _total):
                progress(completed_frames + done, total_frames)

            for episode_index in episodes:
                episode_result = evaluate_act_episode(
                    dataset_path=payload["dataset"],
                    checkpoint_path=payload["checkpoint"],
                    episode_index=int(episode_index),
                    device=payload.get("device", "cuda"),
                    stride=stride,
                    max_frames=max_frames,
                    progress_callback=dataset_progress,
                    model_type=payload.get("model_type", "act"),
                    model=model,
                    seed=payload.get("seed", 0),
                )
                episode_results.append(episode_result)
                completed_frames += episode_result["evaluated_frames"]

            actual = np.concatenate([np.asarray(item["actual"], dtype=np.float32) for item in episode_results])
            predicted = np.concatenate([np.asarray(item["predicted"], dtype=np.float32) for item in episode_results])
            observed_state = np.concatenate(
                [np.asarray(item["observed_state"], dtype=np.float32) for item in episode_results]
            )
            global_indices = np.concatenate(
                [np.asarray(item["global_indices"], dtype=np.int64) for item in episode_results]
            )
            latencies = [value for item in episode_results for value in item["inference_latencies_s"]]
            metrics = calculate_metrics(actual, predicted)
            total_latency = max(float(np.sum(latencies)), 1e-9)
            result = {
                "dataset": episode_results[0]["dataset"],
                "checkpoint": episode_results[0]["checkpoint"],
                "scope": "dataset",
                "episode_count": len(episode_results),
                "evaluated_frames": int(len(actual)),
                "action_dim": episode_results[0]["action_dim"],
                "model_type": payload.get("model_type", "act"),
                "runtime": runtime,
                "inference_latencies_s": latencies,
                "evaluation_wall_s": sum(item["evaluation_wall_s"] for item in episode_results),
                "evaluated_frames_per_second": len(actual) / max(
                    sum(item["evaluation_wall_s"] for item in episode_results), 1e-9
                ),
                "prediction_alignment": episode_results[0]["prediction_alignment"],
                "joint_names": episode_results[0]["joint_names"],
                "joint_names_source": episode_results[0]["joint_names_source"],
                "action_units": "dataset native units",
                "comparison": episode_results[0]["comparison"],
                "seed": payload.get("seed", 0),
                "episode_indices": [idx for item in episode_results for idx in item["episode_indices"]],
                "episode_frame_indices": [idx for item in episode_results for idx in item["frame_indices"]],
                "frame_indices": list(range(len(actual))),
                "global_indices": global_indices.tolist(),
                "actual": actual.tolist(),
                "predicted": predicted.tolist(),
                "observed_state": observed_state.tolist(),
                "metrics": metrics,
                "latency": {
                    "mean_s": float(np.mean(latencies)),
                    "p50_s": float(np.percentile(latencies, 50)),
                    "p95_s": float(np.percentile(latencies, 95)),
                    "p99_s": float(np.percentile(latencies, 99)),
                    "actions_per_second": float(len(actual) / total_latency),
                },
                "episode_summaries": [
                    {
                        "episode_index": item["episode_index"],
                        "task": item["task"],
                        "frames": item["evaluated_frames"],
                        "mean_mae": item["metrics"]["mean_mae"],
                        "mean_rmse": item["metrics"]["mean_rmse"],
                        "p95_latency_s": item["latency"]["p95_s"],
                    }
                    for item in episode_results
                ],
            }
        else:
            result = evaluate_act_episode(
                dataset_path=payload["dataset"],
                checkpoint_path=payload["checkpoint"],
                episode_index=int(payload["episode"]),
                device=payload.get("device", "cuda"),
                stride=stride,
                max_frames=max_frames,
                progress_callback=progress,
                model_type=payload.get("model_type", "act"),
                model=model,
                seed=payload.get("seed", 0),
            )

        result["split"] = payload.get("_split")
        run_dir = Path(MODEL_VALIDATION_DIR) / "artifacts" / "evaluations" / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / f"{result.get('scope', 'episode')}_{result.get('episode_index', 'all')}.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        update_mv_job(
            job_id, status="complete", progress=100, message="Evaluation complete",
            result=result, result_path=str(result_path),
        )

    except ModelValidationCancelled:
        update_mv_job(job_id, status="cancelled", progress=0, message="Evaluation stopped")
    except Exception as exc:
        update_mv_job(job_id, status="error", progress=0, message=str(exc))
    finally:
        # Keep GPU weights alive across episodes, but release them when the job ends.
        del model


@app.route('/api/model_validation/dataset_info', methods=['POST'])
def model_validation_dataset_info():
    """Load a dataset's frame/episode/camera summary, filtered to only the episodes held
    out for evaluation for the given checkpoint (same train/test split the training
    pipeline itself uses -- see compute_mv_test_episodes) -- these populate the Episode
    dropdown, so a training episode can never accidentally be "validated" against.
    """
    payload = request.get_json(silent=True) or {}
    dataset_path = str(payload.get("dataset", "")).strip()
    checkpoint_path = str(payload.get("checkpoint", "")).strip()
    if not dataset_path or not checkpoint_path:
        return jsonify({"error": "dataset and checkpoint are both required"}), 400
    try:
        dataset_path = str(resolve_dataset_path(dataset_path))
        checkpoint_path = str(resolve_checkpoint_path(checkpoint_path))
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403

    try:
        all_episodes = mv_dataset_episode_metadata(dataset_path)
        total_frames = sum(ep["length"] for ep in all_episodes)
        dataset_meta = LeRobotCustomDataset(dataset_path)
        try:
            fps, cameras = dataset_meta.fps, dataset_meta.camera_keys
        finally:
            dataset_meta.close()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        resolved_checkpoint = resolve_pretrained_model_dir(checkpoint_path)
        split = compute_mv_test_episodes(dataset_path, str(resolved_checkpoint), all_episodes)
    except Exception as exc:
        return jsonify({"error": f"Could not determine the train/test split from this checkpoint: {exc}"}), 400

    test_set = set(split["test_episodes"])
    test_episodes = [ep for ep in all_episodes if ep["episode_index"] in test_set]
    if not test_episodes:
        return jsonify({"error": "No held-out test episodes for this dataset/checkpoint pair."}), 400

    return jsonify({
        "dataset": str(Path(dataset_path).expanduser().resolve()),
        "frames": sum(ep["length"] for ep in test_episodes),
        "total_frames": total_frames,
        "fps": fps,
        "cameras": cameras,
        "episodes": test_episodes,
        "total_episodes": len(all_episodes),
        "eval_split": split["eval_split"],
        "held_out_during_training": split["held_out_during_training"],
        "split_source": split["split_source"],
    })


@app.route('/api/model_validation/evaluate', methods=['POST'])
def start_model_validation():
    """Start an ACT/Diffusion/SmolVLA evaluation job (a chosen episode, or the whole
    test split), restricted to the held-out test episodes (see compute_mv_test_episodes).
    """
    payload = request.get_json(silent=True) or {}

    model_type = str(payload.get("model_type", "act")).lower()
    if model_type not in available_models():
        return jsonify({"error": f"Unknown model_type '{model_type}'. Available: {available_models()}"}), 400
    payload["model_type"] = model_type
    if payload.get("inference_profile", "checkpoint") not in PROFILES:
        return jsonify({"error": f"inference_profile must be one of {PROFILES}"}), 400

    try:
        for field, default in (("stride", 1), ("seed", 0)):
            value = payload.get(field, default)
            if isinstance(value, bool) or str(value) != str(int(value)):
                raise ValueError(f"{field} must be an integer")
            payload[field] = int(value)
        if payload["stride"] < 1 or not 0 <= payload["seed"] < 2**32:
            raise ValueError("stride must be >= 1 and seed must be in [0, 2**32)")
        if payload.get("max_frames") not in (None, ""):
            value = payload["max_frames"]
            if isinstance(value, bool) or str(value) != str(int(value)) or int(value) < 1:
                raise ValueError("max_frames must be a positive integer")
            payload["max_frames"] = int(value)
        if payload.get("scope") != "dataset":
            value = payload.get("episode")
            if isinstance(value, bool) or str(value) != str(int(value)) or int(value) < 0:
                raise ValueError("episode must be a nonnegative integer")
            payload["episode"] = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify({"error": str(exc)}), 400

    required = ["dataset", "checkpoint"]
    if payload.get("scope") != "dataset":
        required.append("episode")

    for field in required:
        if not str(payload.get(field, "")).strip():
            return jsonify({"error": f"{field} is required"}), 400

    try:
        payload["dataset"] = str(resolve_dataset_path(payload["dataset"]))
        payload["checkpoint"] = str(resolve_checkpoint_path(payload["checkpoint"]))
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403

    try:
        payload["checkpoint"] = str(
            validate_mv_checkpoint_path(resolve_pretrained_model_dir(payload["checkpoint"]), model_type)
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        all_episodes = mv_dataset_episode_metadata(payload["dataset"])
        split = compute_mv_test_episodes(payload["dataset"], payload["checkpoint"], all_episodes)
    except Exception as exc:
        return jsonify({"error": f"Could not determine the train/test split from this checkpoint: {exc}"}), 400

    payload["_split"] = split
    test_episode_indices = set(split["test_episodes"])
    if payload.get("scope") == "dataset":
        if not test_episode_indices:
            return jsonify({"error": "No held-out test episodes for this dataset/checkpoint pair."}), 400
        payload["_test_episodes"] = sorted(test_episode_indices)
        payload["_episode_lengths"] = {ep["episode_index"]: ep["length"] for ep in all_episodes}
    elif int(payload["episode"]) not in test_episode_indices:
        return jsonify({
            "error": f"Episode {payload['episode']} is not in the held-out test split "
                     f"for this checkpoint (eval_split={split['eval_split']})."
        }), 400

    with mv_jobs_lock:
        active = [j for j in mv_jobs.values() if j["status"] in {"queued", "loading", "running"}]
        if active:
            return jsonify({"error": "A model validation evaluation is already running."}), 409

        job_id = uuid.uuid4().hex[:12]
        mv_jobs[job_id] = {
            "job_id": job_id, "status": "queued", "progress": 0,
            "message": "Queued", "cancel_requested": False,
        }

    mv_executor.submit(run_mv_evaluation, job_id, payload)
    return jsonify({"job_id": job_id})


@app.route('/api/model_validation/job/<job_id>/stop', methods=['POST'])
def stop_model_validation(job_id):
    with mv_jobs_lock:
        job = mv_jobs.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        if job["status"] not in {"queued", "loading", "running"}:
            return jsonify({"error": "job is not running", "status": job["status"]}), 409
        job["cancel_requested"] = True
        job["message"] = "Stopping evaluation..."
    return jsonify({"job_id": job_id, "status": "stopping"})


@app.route('/api/model_validation/jobs/active', methods=['GET'])
def active_model_validation_job():
    with mv_jobs_lock:
        active = next((job for job in reversed(list(mv_jobs.values()))
                       if job.get("status") in {"queued", "loading", "running", "stopping"}), None)
        return jsonify(active or {})


@app.route('/api/model_validation/job/<job_id>', methods=['GET'])
def model_validation_job_status(job_id):
    with mv_jobs_lock:
        job = mv_jobs.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        return jsonify(job)

@app.route('/api/datasets/convert', methods=['POST'])
def convert_dataset():
    """Convert a raw Daksha dataset in place into the real LeRobot v3.0 layout."""
    data = request.json or {}
    raw_path = data.get('path', '').strip() or data.get('dataset_id', '').strip()

    if not raw_path:
        return jsonify({"status": "error", "message": "No path provided."}), 400
    try:
        dataset_path = resolve_dataset_path(raw_path)
    except PermissionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 403
    if not dataset_path.exists() or not dataset_path.is_dir():
        return jsonify({"status": "error", "message": f"Path does not exist or is not a directory: {dataset_path}"}), 404

    try:
        log_message(TRAIN_LOG, f"Converting dataset in place: {dataset_path}")
        conv_result = ensure_dataset_v30(str(dataset_path), TRAIN_LOG)
        log_message(TRAIN_LOG, f"Conversion result: {conv_result.get('status')}")

        validation = validate_dataset_with_lerobot(str(dataset_path), repo_id=f"local/{dataset_path.name}")
        return jsonify({
            "status": "success",
            "dataset_id": dataset_path.name,
            "conversion": conv_result,
            "validation": validation,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Conversion error: {str(e)}"}), 500

@app.route('/api/datasets/delete', methods=['POST'])
def delete_dataset():
    """Permanently delete a dataset."""
    data = request.json or {}
    path = data.get('path', '').strip()
    
    if not path:
        return jsonify({"status": "error", "message": "No path provided."}), 400
    
    try:
        path = str(resolve_dataset_path(path))
    except PermissionError:
        return jsonify({"status": "error", "message": "Invalid dataset path (security check failed)."}), 403
    
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": f"Dataset not found: {path}"}), 404
    
    if not os.path.isdir(path):
        return jsonify({"status": "error", "message": f"Path is not a directory: {path}"}), 400
    
    try:
        dataset_name = os.path.basename(path)
        total_size = get_dir_size(path)
        size_gb = total_size / (1024**3)
        
        shutil.rmtree(path)
        log_message(TRAIN_LOG, f"Dataset deleted: {dataset_name} ({size_gb:.2f} GB)")
        
        return jsonify({
            "status": "success",
            "message": f"Dataset '{dataset_name}' permanently deleted ({size_gb:.2f} GB freed)",
        })
    except Exception as e:
        log_message(TRAIN_LOG, f"Error deleting dataset {path}: {str(e)}")
        return jsonify({"status": "error", "message": f"Deletion failed: {str(e)}"}), 500

# ════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - Training
# ════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Validate the request and kick off dataset-prep + training in the background.

    Dataset conversion (see ensure_dataset_v30) and video/stat computation can take
    minutes on a large dataset, so this only does fast synchronous validation and then
    hands off to a background thread. Progress is visible via /api/train/status
    ("training_stage": "converting" | "training") and the live /api/train/logs stream.
    """
    global global_state

    # Cheap early-exit, not the real guard -- see the atomic check-and-claim below,
    # right before launch, which is what actually prevents two concurrent requests
    # from both passing this check and launching two training subprocesses.
    if global_state["training_active"]:
        return jsonify({"status": "error", "message": "Training already running."}), 400

    try:
        data = request.json
        policy_type = data.get('policy_type')
        dataset_path = data.get('dataset_path') or data.get('dataset_id')
        config = data.get('config', {})
        job_name = data.get('job_name', f"{policy_type}_daksha")

        if policy_type not in POLICY_CONFIGS:
            return jsonify({"status": "error", "message": f"Invalid policy type: {policy_type}"}), 400

        if policy_type == "smolvla":
            try:
                config = smolvla_training_config(config)
            except ValueError as exc:
                return jsonify({"status": "error", "message": str(exc)}), 400

        try:
            dataset_path = str(resolve_dataset_path(dataset_path))
        except PermissionError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 403

        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({"status": "error", "message": f"Invalid dataset path: {dataset_path}"}), 400

        policy_cfg = POLICY_CONFIGS[policy_type]
        script_path = policy_cfg['script']
        if not os.path.exists(script_path):
            return jsonify({"status": "error", "message": f"Training script not found: {script_path}"}), 404

        dataset_name = Path(dataset_path).name
        safe_job_name = ''.join(ch if ch.isalnum() or ch in ['_', '-', '.'] else '_' for ch in job_name)
        safe_job_name = safe_job_name.strip('._') or f"{policy_type}_run"
        output_dir = f"{OUTPUT_ROOT}/{policy_type}/{dataset_name}/{safe_job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Atomic check-and-claim: with the lock held, no other request can observe
        # "not active" and also start a second training run before this one claims it.
        with global_state_lock:
            if global_state["training_active"] or _managed_process_record("training"):
                return jsonify({"status": "error", "message": "Training already running."}), 400
            global_state["training_generation"] += 1
            generation = global_state["training_generation"]
            cancel_event = threading.Event()
            global_state["training_cancel_event"] = cancel_event
            global_state["training_process"] = None
            global_state["training_active"] = True
            global_state["training_policy"] = policy_type
            global_state["training_stage"] = "converting"
            global_state["cancel_requested"] = False
            global_state["current_step"] = 0
            global_state["total_steps"] = int(config.get('steps', policy_cfg['hyperparameters']['steps']['default']))
            global_state["training_resumed_from_step"] = 0
            global_state["last_output_dir"] = output_dir

        clear_log(TRAIN_LOG)
        clear_log(TRAIN_PREP_LOG)
        clear_log(TRAIN_RAW_LOG)

        threading.Thread(
            target=_prepare_and_train,
            args=(policy_type, dataset_path, dataset_name, config, job_name, safe_job_name,
                  output_dir, policy_cfg, script_path, generation, cancel_event),
            daemon=True,
        ).start()

        return jsonify({
            "status": "success",
            "message": f"Preparing dataset and starting {policy_type} training",
            "output_dir": output_dir,
            "job_name": job_name,
        })

    except Exception as e:
        global_state["training_active"] = False
        global_state["training_stage"] = None
        return jsonify({"status": "error", "message": f"Training start error: {str(e)}"}), 500


# ════════════════════════════════════════════════════════════════════════════════════
# CHECKPOINT RETENTION
#
# lerobot.scripts.lerobot_train (see lerobot.common.train_utils.save_checkpoint) writes
# a new checkpoints/<step>/ every save_freq steps and never deletes old ones -- only a
# `last` symlink tracks the newest. Left alone, a long run fills the disk with every
# intermediate checkpoint. This keeps only the CHECKPOINT_KEEP_LAST most recent complete
# ones per run, for both ACT and Diffusion alike (same lerobot_train checkpoint layout
# regardless of policy type, so one pruning routine covers both).
# ════════════════════════════════════════════════════════════════════════════════════

CHECKPOINT_KEEP_LAST = 3


def _is_checkpoint_complete(step_dir: Path) -> bool:
    model = step_dir / "pretrained_model"
    try:
        config = json.loads((model / "config.json").read_text())
        if config.get("type") == "smolvla":
            return all((model / name).is_file() for name in (
                "model.safetensors", "train_config.json", "policy_preprocessor.json",
                "policy_postprocessor.json"))
        return True
    except (OSError, ValueError):
        return False


def _list_run_checkpoint_steps(run_dir) -> list[int]:
    """Complete checkpoint step numbers under `run_dir/checkpoints`, newest first."""
    checkpoints_dir = Path(run_dir) / "checkpoints"
    if not checkpoints_dir.is_dir():
        return []
    steps = [
        int(d.name)
        for d in checkpoints_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and _is_checkpoint_complete(d)
    ]
    return sorted(steps, reverse=True)


def _checkpoint_step_options(run_dir) -> list[dict]:
    """Exact complete checkpoint directories under a run, newest first."""
    checkpoints_dir = Path(run_dir) / "checkpoints"
    if not checkpoints_dir.is_dir():
        return []
    step_dirs = sorted(
        (
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and path.name.isdigit() and _is_checkpoint_complete(path)
        ),
        key=lambda path: int(path.name),
        reverse=True,
    )
    return [
        {
            "step": int(step_dir.name),
            "label": step_dir.name,
            "path": str(step_dir),
        }
        for step_dir in step_dirs
    ]


def _prune_old_checkpoints(output_dir: str, keep: int = CHECKPOINT_KEEP_LAST) -> None:
    """Delete all but the `keep` most recent complete checkpoints under output_dir/checkpoints.

    Only ever removes a checkpoint once it looks fully written (has pretrained_model/config.json),
    so one still being saved is never touched. The `last` symlink always points at the newest
    checkpoint, which is always kept, so it's never left dangling.
    """
    checkpoints_dir = Path(output_dir) / "checkpoints"
    if not checkpoints_dir.is_dir() or keep <= 0:
        return
    step_dirs = sorted(
        (d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit() and _is_checkpoint_complete(d)),
        key=lambda d: int(d.name),
    )
    for old in step_dirs[:-keep]:
        try:
            shutil.rmtree(old)
            log_message(TRAIN_PREP_LOG, f"Pruned old checkpoint (keeping last {keep}): {old.name}")
        except Exception as e:
            log_message(TRAIN_PREP_LOG, f"Failed to prune checkpoint {old}: {e}")


def _prune_checkpoints_loop(output_dir: str) -> None:
    """Background sweep, run alongside a training subprocess for as long as it's active."""
    while global_state["training_active"]:
        _prune_old_checkpoints(output_dir)
        time.sleep(15)
    # One last sweep in case a final checkpoint landed just as training stopped.
    _prune_old_checkpoints(output_dir)


def _prepare_and_train(policy_type, dataset_path, dataset_name, config, job_name, safe_job_name,
                       output_dir, policy_cfg, script_path, generation=None, cancel_event=None):
    """Background worker: convert dataset to v3.0 if needed, validate, then train."""
    if cancel_event is None:
        cancel_event = global_state.get("training_cancel_event") or threading.Event()
    if generation is None:
        generation = global_state.get("training_generation", 0)

    def cancelled():
        return cancel_event.is_set() or generation != global_state.get("training_generation")

    def register_process(proc):
        if cancelled():
            if proc is not None:
                _stop_managed_process("training", proc, timeout=1)
            return
        if proc is None:
            global_state["training_process"] = None
            _clear_managed_process("training")
            return
        global_state["training_process"] = proc
        _register_managed_process(
            "training", proc, policy=policy_type,
            stage=global_state.get("training_stage") or "converting",
            output_dir=output_dir, total_steps=global_state.get("total_steps", 0),
            current_step=global_state.get("current_step", 0),
        )

    try:
        if cancelled():
            return

        dataset_root = Path(dataset_path)
        if (policy_type == "smolvla" and not _dataset_is_v30_ready(dataset_root)
                and (dataset_root / "meta/tasks.parquet").is_file()
                and any((dataset_root / "meta/episodes").glob("*/*.parquet"))):
            python_argv, extra_env = resolve_repo_python()
            env = os.environ.copy()
            env.update(extra_env)
            _run_conversion_step(python_argv + [DATASET_V30_LEGACY_SCRIPT, "--root", dataset_path],
                                 env, REPO_ROOT, TRAIN_PREP_LOG, register_process)
        else:
            ensure_dataset_v30(dataset_path, TRAIN_PREP_LOG, register_process=register_process)

        if cancelled():
            log_message(TRAIN_PREP_LOG, "Training cancelled before dataset validation.")
            return

        # Validate dataset through the real LeRobot validator before launching training.
        # Its output (dataset paths, codebase-version checks, corrupted-episode warnings)
        # is diagnostic prep noise, so it goes to the prep log, never the training tab.
        try:
            validation_result = validate_dataset_with_lerobot(dataset_path, repo_id=f"local/{dataset_name}")
            if validation_result.get("validation_status") == "warning":
                log_message(
                    TRAIN_PREP_LOG,
                    f"Dataset validation found {len(validation_result.get('corrupted_episodes', []))} corrupted episodes in {dataset_path}",
                )
                for issue in validation_result.get("issues", [])[:5]:
                    log_message(TRAIN_PREP_LOG, f"  episode {issue['episode']}: {issue['reason']}")
            else:
                log_message(TRAIN_PREP_LOG, f"Dataset validation passed for {dataset_path}")
        except Exception as validation_error:
            log_message(TRAIN_PREP_LOG, f"Dataset validation warning: {validation_error}")

        if cancelled():
            log_message(TRAIN_PREP_LOG, "Training cancelled before launch.")
            return

        # Build environment variables
        env = os.environ.copy()
        env.update({
            'REPO_ID': f'local/{dataset_name}',
            'DATASET_ROOT': dataset_path,
            'OUTPUT_DIR': output_dir,
            'JOB_NAME': safe_job_name,
            'BATCH_SIZE': str(config.get('batch_size', policy_cfg['hyperparameters']['batch_size']['default'])),
            'STEPS': str(config.get('steps', policy_cfg['hyperparameters']['steps']['default'])),
            'DEVICE': 'cuda',
            'RUN_DATASET_CHECK': 'true',
            'EVAL_SPLIT': '0.1',
            'EVAL_STEPS': str(config.get('eval_freq', 5000 if policy_type == 'diffusion' else 500)),
            # eval_freq is independent of save_freq, so lowering save_freq no longer drags
            # eval frequency along with it. Without the MAX_EVAL_SAMPLES cap below, each eval
            # pass runs uncapped over the whole 10% held-out split with num_workers=0 -- many
            # minutes of video decoding that looks like training hung, worse the lower
            # eval_freq is set. See train_*_local.sh.
            'MAX_EVAL_SAMPLES': '64',
            'PUSH_TO_HUB': 'false',
        })
        env.pop('HUB_REPO_ID', None)
        if policy_type in ("act", "smolvla"):
            # Both launchers source smolvla/common.sh for their dataset check and
            # training command; keep them on the UI's dependency-complete interpreter.
            env["SMOLVLA_PYTHON"] = resolve_repo_python_executable()
        # This flow is fully local (REPO_ID is "local/...") and doesn't need the Hub.
        env['HF_HUB_OFFLINE'] = '1'
        env['HF_HUB_DISABLE_TELEMETRY'] = '1'

        for key in ['chunk_size', 'gradient_accumulation', 'lr', 'num_diffusion_steps', 'guidance_scale', 'hidden_dim', 'save_freq', 'log_freq']:
            if key in config:
                env[key.upper()] = str(config[key])

        if policy_type == "smolvla":
            configure_smolvla_environment(env, config)
        # Split / path / output / full-config details are prep diagnostics -> prep log.
        log_message(TRAIN_PREP_LOG, f"{policy_type}: reserving 10% of whole episodes per task for validation; "
                    "training uses the remaining episodes. Split is saved in train_config.json.")
        log_message(TRAIN_PREP_LOG, f"Dataset path: {dataset_path}")
        log_message(TRAIN_PREP_LOG, f"Output: {output_dir}")
        log_message(TRAIN_PREP_LOG, f"Config: {json.dumps(config, indent=2)}")

        # Training tab shows only: dataset name, policy, and "Training started (<steps>)".
        # Everything after this line is the raw backend training output (loss, ...).
        total_steps = global_state.get("total_steps", 0)
        log_message(TRAIN_LOG, f"Dataset: {dataset_name}")
        log_message(TRAIN_LOG, f"Policy: {policy_type}")
        log_message(TRAIN_LOG, f"Training started ({total_steps} steps)")

        # Do not pre-create this directory. The ACT launcher treats an existing
        # OUTPUT_DIR as a collision and silently changes it to a timestamp suffix;
        # that left the UI tracking an empty folder while weights went elsewhere.
        # lerobot_train creates the requested output directory itself.
        threading.Thread(target=_prune_checkpoints_loop, args=(output_dir,), daemon=True).start()

        # Start the training script as a new session so child processes are placed in
        # the same process group, allowing us to reliably terminate the whole tree later.
        # Serialize the final cancellation check with Stop and process registration.
        with global_state_lock:
            if cancelled() or not global_state["training_active"]:
                return
            # Backend stdout goes to the raw log; _monitor_training_logs mirrors only the
            # worthwhile lines into TRAIN_LOG (the training tab).
            with open(TRAIN_RAW_LOG, "a", encoding="utf-8", buffering=1) as child_log:
                process = subprocess.Popen(
                    ['bash', script_path], env=env, stdout=child_log,
                    stderr=subprocess.STDOUT, cwd=REPO_ROOT, start_new_session=True,
                )
            # Flip stage to "training" only after the subprocess is in hand, so the
            # status endpoint never sees stage="training" with no process registered
            # and misreports a run that is still starting as "stopped".
            global_state["training_stage"] = "training"
            global_state["training_process"] = process
            _register_managed_process(
                "training", process, policy=policy_type, stage="training",
                output_dir=output_dir, total_steps=global_state.get("total_steps", 0),
                current_step=global_state.get("current_step", 0),
                dataset_name=dataset_name,
            )
        _monitor_training_logs(process)

    except Exception as e:
        log_message(TRAIN_PREP_LOG, f"Dataset preparation / training failed: {e}")
        log_message(TRAIN_PREP_LOG, traceback.format_exc())
        with global_state_lock:
            if generation == global_state.get("training_generation"):
                global_state["training_active"] = False
                global_state["training_stage"] = None
                global_state["training_process"] = None


def _iter_process_output(process, idle_timeout_s=1.0):
    """Yield a subprocess's stdout lines one at a time, without blocking forever past
    the process's own exit.

    A plain "while True: readline()" only unblocks once the pipe's write end is
    closed by *every* process holding it, not just the one Popen() started -- if that
    process forks a grandchild that inherits the same stdout fd and isn't reaped
    alongside it (a backgrounded helper, an orphaned DataLoader worker, etc.),
    readline() can block forever even though process.poll() already shows the
    tracked process as long gone, leaving the monitor thread -- and whatever
    *_active flag its finally-block is responsible for clearing -- stuck forever.
    select() with a timeout lets us notice that case and give up instead.
    """
    while True:
        ready, _, _ = select.select([process.stdout], [], [], idle_timeout_s)
        if ready:
            line = process.stdout.readline()
            if not line:
                return  # Real EOF: every holder of the pipe's write end has closed it.
            yield line
            continue
        if process.poll() is not None:
            return


# The training tab shows only backend lines that carry real training signal: lerobot's
# per-log_freq metric lines. Those are the MetricsTracker line (logged every log_freq
# steps, e.g. 20 -- "... step:20 ... loss:3.565 grdn:... lr:..."; it arrives with the
# tqdm bar prefixed to it), the machine-readable "loss_metric step=... train_loss=..."
# line, and eval-loss summaries -- all of which contain "loss". The bare per-step tqdm
# progress bars (one per step, no metrics) are NOT mirrored -- they are pure spam in the
# tab -- but they still advance the UI progress bar because _consume_training_line parses
# their "N/total" from the raw stream regardless of whether the line is displayed.
# Everything else is dropped too: the shell launcher's banner / "Dataset Root:" /
# "Output Dir:" / dataset-check echoes, lerobot's full-config pretty-print dump and its
# "Creating .../Output dir:" housekeeping, and library warnings (scipy NumPy, torchcodec
# fallback, "Unknown fields in DatasetInfo"). The complete unfiltered output is always
# available in TRAIN_RAW_LOG.
def _keep_training_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # lerobot metric lines: MetricsTracker ("... loss:3.565 ..."), the loss_metric line,
    # and eval-loss summaries all contain "loss". Bare tqdm progress bars do not.
    return 'loss' in s.lower()


def _append_display_line(line: str) -> None:
    """Append a raw backend line to the training tab verbatim (it already carries its
    own formatting; no [timestamp] prefix like log_message adds)."""
    try:
        with open(TRAIN_LOG, 'a', encoding='utf-8') as f:
            f.write(line if line.endswith('\n') else line + '\n')
    except Exception:
        pass


def _training_step_from_line(line: str, total_steps: int, resumed_from_step: int = 0) -> int | None:
    metric = re.search(r"\bloss_metric\s+step=(\d+)\b", line)
    if metric:
        step = int(metric.group(1))
        return step if 0 <= step <= total_steps else None

    progress = re.search(r"(\d+)\s*/\s*(\d+)", line)
    if not progress:
        return None
    completed, denominator = map(int, progress.groups())
    if denominator == total_steps:
        return completed
    remaining_steps = max(0, total_steps - resumed_from_step)
    if resumed_from_step and denominator == remaining_steps:
        return resumed_from_step + completed
    return None


def _consume_training_line(line: str) -> None:
    """Update step progress from any line, then mirror it to the tab if worth keeping."""
    step = _training_step_from_line(
        line,
        int(global_state.get("total_steps") or 0),
        int(global_state.get("training_resumed_from_step") or 0),
    )
    if step is not None:
        global_state["current_step"] = max(global_state.get("current_step", 0), step)
        record = _managed_process_record("training")
        if record:
            record["current_step"] = global_state["current_step"]
            processes = _read_managed_processes()
            processes["training"] = record
            _write_managed_processes(processes)
    if _keep_training_line(line):
        _append_display_line(line)


def _consume_training_log_chunk(chunk: str, pending: str = "") -> str:
    """Consume complete newline- or carriage-return-delimited backend records."""
    records = re.split(r"[\r\n]", pending + chunk)
    for record in records[:-1]:
        if record:
            _consume_training_line(record)
    return records[-1]


def _tail_training_log(log_file, should_continue) -> None:
    """Tail a raw log while treating tqdm carriage returns as record boundaries."""
    pending = ""
    while should_continue():
        chunk = log_file.read(8192)
        if chunk:
            pending = _consume_training_log_chunk(chunk, pending)
        else:
            time.sleep(0.2)

    while True:
        chunk = log_file.read(8192)
        if not chunk:
            break
        pending = _consume_training_log_chunk(chunk, pending)
    if pending:
        _consume_training_line(pending)


def _rebuild_training_display_log(record: dict) -> int:
    """Rebuild TRAIN_LOG (header + filtered raw) after a UI restart. Returns the raw
    byte offset consumed, so the re-attached mirror can continue without gaps/dupes."""
    clear_log(TRAIN_LOG)
    policy = record.get("policy")
    total = record.get("total_steps", 0)
    if record.get("resumed"):
        log_message(TRAIN_LOG, f"Policy: {policy}")
        log_message(TRAIN_LOG, f"Training resumed ({record.get('current_step', 0)} / {total} steps)")
    else:
        if record.get("dataset_name"):
            log_message(TRAIN_LOG, f"Dataset: {record['dataset_name']}")
        log_message(TRAIN_LOG, f"Policy: {policy}")
        log_message(TRAIN_LOG, f"Training started ({total} steps)")
    offset = 0
    try:
        with open(TRAIN_RAW_LOG, 'r', encoding='utf-8', errors='replace') as raw:
            pending = ""
            while chunk := raw.read(8192):
                pending = _consume_training_log_chunk(chunk, pending)
            offset = raw.tell()
            if pending:
                _consume_training_line(pending)
    except OSError:
        pass
    return offset


def _reattach_training_log_mirror(start_offset: int) -> None:
    """After a UI restart, keep mirroring raw->tab for a surviving training process
    until it exits (tracked via the managed-process record)."""
    def run():
        try:
            with open(TRAIN_RAW_LOG, 'r', encoding='utf-8', errors='replace') as raw:
                raw.seek(start_offset)
                _tail_training_log(
                    raw,
                    should_continue=lambda: _managed_process_record("training") is not None,
                )
        except Exception as e:
            log_message(TRAIN_PREP_LOG, f"Log reattach error: {e}")
        finally:
            with global_state_lock:
                if _managed_process_record("training") is None:
                    global_state["training_active"] = False
                    global_state["training_stage"] = None
    threading.Thread(target=run, daemon=True).start()


def _monitor_training_logs(process):
    """Tail the raw backend stream, update step progress, and mirror only the
    worthwhile lines into the training tab. The job itself survives a UI restart;
    _recover_managed_processes re-attaches this mirror for a surviving process."""
    try:
        with open(TRAIN_RAW_LOG, "r", encoding="utf-8", errors="replace") as log_file:
            _tail_training_log(log_file, should_continue=lambda: process.poll() is None)
    except Exception as e:
        log_message(TRAIN_PREP_LOG, f"Log monitoring error: {e}")
    finally:
        returncode = process.poll()
        with global_state_lock:
            # Only this thread's own process is still the tracked one -- a newer
            # training run replacing it in the meantime must not have its state
            # clobbered by this (stale) monitor finishing up late.
            if global_state["training_process"] is process:
                global_state["training_active"] = False
                global_state["training_stage"] = None
                _clear_managed_process("training", process)
        if returncode == 0:
            log_message(TRAIN_LOG, "Training completed successfully")
        else:
            log_message(TRAIN_LOG, f"Training failed (exit code {returncode})")


# ════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - Training: resume from a saved checkpoint
#
# lerobot.scripts.lerobot_train has native resume support (TrainPipelineConfig.resume +
# --config_path, see lerobot/configs/train.py): pointed at a checkpoint's
# pretrained_model/ dir (which holds train_config.json), it reloads that run's full
# config -- dataset, policy type, hyperparameters, optimizer/scheduler/rng state, and
# step count -- and continues writing into the SAME output_dir/checkpoints tree. So
# unlike a fresh run, resume needs no policy_type/dataset_path/hyperparameters from the
# caller at all: just which checkpoint. Works identically for ACT and Diffusion since
# the policy type comes from the checkpoint's own saved config.
# ════════════════════════════════════════════════════════════════════════════════════


@app.route('/api/train/resume', methods=['POST'])
def resume_training():
    """Resume training from a saved checkpoint (defaults to the run's most recent one)."""
    global global_state

    # Cheap early-exit only -- see the atomic check-and-claim below, right before
    # launch, which is the real guard against two concurrent resume requests.
    if global_state["training_active"]:
        return jsonify({"status": "error", "message": "Training already running."}), 400

    try:
        data = request.json or {}
        # "Retrain" in the Training tab omits checkpoint_path entirely -- it always means
        # "continue the run I just started/stopped", tracked in last_output_dir regardless
        # of whether that run was itself fresh or already a resume.
        checkpoint_path = data.get('checkpoint_path') or global_state.get('last_output_dir')
        if not checkpoint_path:
            return jsonify({
                "status": "error",
                "message": "No checkpoint to retrain: no training run this session yet. "
                           "Pick one from the Checkpoints tab, or pass checkpoint_path explicitly.",
            }), 400
        try:
            checkpoint_path = str(resolve_checkpoint_path(checkpoint_path))
        except PermissionError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 403
        if not os.path.exists(checkpoint_path):
            return jsonify({"status": "error", "message": f"Invalid checkpoint path: {checkpoint_path}"}), 400

        # resolve_pretrained_model_dir() checks checkpoints/last/pretrained_model first --
        # "last" always points at the most recently *completed* checkpoint (see
        # update_last_checkpoint in lerobot.common.train_utils), so passing a run's
        # output_dir here resumes from its latest checkpoint automatically.
        pretrained_dir = resolve_pretrained_model_dir(checkpoint_path)
        train_config_path = pretrained_dir / "train_config.json"
        if not train_config_path.exists():
            return jsonify({
                "status": "error",
                "message": f"No train_config.json found under checkpoint: {checkpoint_path} "
                           f"(looked in {pretrained_dir}) -- can't resume without it.",
            }), 400

        train_config = json.loads(train_config_path.read_text())
        original_steps = int(train_config.get('steps') or 0)
        extra_steps = data.get('steps')
        total_steps = int(extra_steps) if extra_steps else original_steps

        training_step_path = pretrained_dir.parent / "training_state" / "training_step.json"
        resumed_from_step = 0
        if training_step_path.exists():
            try:
                resumed_from_step = int(json.loads(training_step_path.read_text()).get('step') or 0)
            except (json.JSONDecodeError, ValueError):
                pass

        # output_dir/checkpoints/<step>/pretrained_model -> output_dir, three levels up.
        output_dir = str(pretrained_dir.parent.parent.parent)
        policy_type = ((train_config.get('policy') or {}).get('type')) or 'unknown'

        with global_state_lock:
            if global_state["training_active"] or _managed_process_record("training"):
                return jsonify({"status": "error", "message": "Training already running."}), 400
            global_state["training_generation"] += 1
            generation = global_state["training_generation"]
            cancel_event = threading.Event()
            global_state["training_cancel_event"] = cancel_event
            global_state["training_process"] = None
            global_state["training_active"] = True
            global_state["training_policy"] = policy_type
            global_state["training_stage"] = "preparing"
            global_state["cancel_requested"] = False
            global_state["current_step"] = resumed_from_step
            global_state["total_steps"] = total_steps
            global_state["training_resumed_from_step"] = resumed_from_step
            global_state["last_output_dir"] = output_dir

        clear_log(TRAIN_LOG)
        clear_log(TRAIN_PREP_LOG)
        clear_log(TRAIN_RAW_LOG)
        # Checkpoint path / output dir are prep diagnostics -> prep log. The training tab
        # shows only the policy and the resume step, then the raw backend output.
        log_message(TRAIN_PREP_LOG, f"Resuming {policy_type} training from checkpoint: {pretrained_dir}")
        log_message(TRAIN_PREP_LOG, f"Output: {output_dir}")
        log_message(TRAIN_LOG, f"Policy: {policy_type}")
        log_message(TRAIN_LOG, f"Training resumed ({resumed_from_step} / {total_steps} steps)")

        threading.Thread(
            target=_prepare_and_resume_train,
            args=(pretrained_dir, output_dir, total_steps if extra_steps else None,
                  generation, cancel_event),
            daemon=True,
        ).start()

        return jsonify({
            "status": "success",
            "message": f"Resuming {policy_type} training from step {resumed_from_step}",
            "output_dir": output_dir,
            "resumed_from_step": resumed_from_step,
            "total_steps": total_steps,
            "policy_type": policy_type,
            "stage": "preparing",
            # The original parameters this run was trained with, mapped to the UI's
            # hyperparameter controls so the Training-configuration panel reflects them.
            "hyperparameters": _hyperparameters_from_train_config(train_config, policy_type),
        })

    except Exception as e:
        global_state["training_active"] = False
        global_state["training_stage"] = None
        return jsonify({"status": "error", "message": f"Resume error: {str(e)}"}), 500


# Where each UI hyperparameter (POLICY_CONFIGS[...]["hyperparameters"]) is found inside a
# saved train_config.json. Each entry is a list of (section, key) candidates tried in
# order; section None means the top level. Used to reflect a resumed run's original
# parameters back into the Training-configuration panel. Unlisted names fall back to the
# top-level key of the same name; names that map to nothing stored (e.g. "offline")
# simply keep the UI default.
_RESUME_CONFIG_SOURCES = {
    "batch_size": [(None, "batch_size")],
    "steps": [(None, "steps")],
    "save_freq": [(None, "save_freq")],
    "log_freq": [(None, "log_freq")],
    "num_workers": [(None, "num_workers")],
    "eval_freq": [(None, "eval_steps")],
    "lr": [("optimizer", "lr")],
    "chunk_size": [("policy", "chunk_size")],
    "action_horizon": [("policy", "n_action_steps"), ("policy", "chunk_size")],
    "num_diffusion_steps": [("policy", "num_train_timesteps")],
    "guidance_scale": [("policy", "guidance_scale")],
    "gradient_accumulation": [(None, "gradient_accumulation_steps"), (None, "gradient_accumulation")],
    "pretrained_path": [("policy", "pretrained_path"), ("policy", "path")],
    "freeze_vision_encoder": [("policy", "freeze_vision_encoder")],
    "train_expert_only": [("policy", "train_expert_only")],
}


def _hyperparameters_from_train_config(train_config: dict, policy_type: str) -> dict:
    """Pull the UI-visible hyperparameters for `policy_type` out of a saved train_config.json.

    Returns only the parameters that are actually present in the config, each coerced to
    the type its UI spec expects, so the Training-configuration panel can reflect a
    resumed run's original settings (anything missing keeps its UI default).
    """
    specs = POLICY_CONFIGS.get(policy_type, {}).get("hyperparameters", {})
    result = {}
    for name, spec in specs.items():
        for section, key in _RESUME_CONFIG_SOURCES.get(name, [(None, name)]):
            container = train_config if section is None else (train_config.get(section) or {})
            if not isinstance(container, dict) or container.get(key) is None:
                continue
            value = container[key]
            spec_type = spec.get("type")
            try:
                if spec_type == "int":
                    value = int(value)
                elif spec_type == "float":
                    value = float(value)
                elif spec_type == "bool":
                    value = bool(value)
                else:
                    value = str(value)
            except (TypeError, ValueError):
                break
            result[name] = value
            break
    return result


def _resolved_local_dataset_root(saved_config: dict) -> str | None:
    """Map a checkpoint's saved dataset.root to this machine's dataset root, if needed.

    A checkpoint trained on another machine stores an absolute dataset.root that does
    not exist here (e.g. /home/thor/Desktop/Dinesh/Dataset/pick_box_1_v2). When that
    saved path is missing locally but a dataset of the same name exists under
    DATASET_ROOT, return the local path so resume reads the local copy instead of
    falling through to the (offline) Hugging Face Hub and erroring out.

    Returns None when the saved root is already valid on this machine (same-machine
    resume -- leave it untouched) or when no local match can be found.
    """
    dataset_cfg = saved_config.get("dataset") or {}
    saved_root = dataset_cfg.get("root")
    if saved_root and (Path(saved_root) / "meta" / "info.json").is_file():
        return None  # saved path resolves here; nothing to remap

    name = Path(saved_root).name if saved_root else None
    if not name:
        repo_id = dataset_cfg.get("repo_id") or ""
        name = repo_id.split("/")[-1] if repo_id else None
    if not name:
        return None

    candidate = Path(DATASET_ROOT) / name
    if (candidate / "meta" / "info.json").is_file():
        return str(candidate)
    return None


def _prepare_and_resume_train(pretrained_dir: Path, output_dir: str, steps_override: int | None,
                              generation=None, cancel_event=None):
    """Background worker: run lerobot_train --resume against a saved checkpoint."""
    try:
        if cancel_event is None:
            cancel_event = global_state.get("training_cancel_event") or threading.Event()
        if generation is None:
            generation = global_state.get("training_generation", 0)
        if cancel_event.is_set() or generation != global_state.get("training_generation"):
            return

        python_argv, extra_env = resolve_repo_python()
        env = os.environ.copy()
        env.update(extra_env)
        # Same as a fresh run: this UI is fully local and never needs the Hub for resume.
        env['HF_HUB_OFFLINE'] = '1'
        env['HF_HUB_DISABLE_TELEMETRY'] = '1'
        env['PYTHONUNBUFFERED'] = '1'

        saved_config = json.loads((pretrained_dir / "train_config.json").read_text())
        # A checkpoint trained on another machine stores an absolute dataset.root (and
        # output_dir) from that machine. Remap the dataset to this machine's copy under
        # DATASET_ROOT, and pin output_dir to the local run dir so resumed checkpoints
        # keep saving in place rather than under the original machine's path.
        dataset_root_override = _resolved_local_dataset_root(saved_config)
        if dataset_root_override:
            log_message(TRAIN_PREP_LOG, f"Remapping dataset root for resume -> {dataset_root_override}")

        cmd = python_argv + [
            '-m', 'lerobot.scripts.lerobot_train',
            '--resume=true',
            f'--config_path={pretrained_dir}',
            f'--output_dir={output_dir}',
        ]
        if dataset_root_override:
            cmd.append(f'--dataset.root={dataset_root_override}')
        if steps_override:
            cmd.append(f'--steps={steps_override}')

        threading.Thread(target=_prune_checkpoints_loop, args=(output_dir,), daemon=True).start()

        with global_state_lock:
            if (cancel_event.is_set() or generation != global_state.get("training_generation")
                    or not global_state["training_active"]):
                return
            # Backend stdout -> raw log; _monitor_training_logs mirrors the kept lines.
            with open(TRAIN_RAW_LOG, "a", encoding="utf-8", buffering=1) as child_log:
                process = subprocess.Popen(
                    cmd, env=env, stdout=child_log, stderr=subprocess.STDOUT,
                    cwd=REPO_ROOT, start_new_session=True,
                )
            global_state["training_stage"] = "training"
            global_state["training_process"] = process
            _register_managed_process(
                "training", process, policy=global_state.get("training_policy"),
                stage="training", output_dir=output_dir,
                total_steps=global_state.get("total_steps", 0),
                current_step=global_state.get("current_step", 0),
                resumed_from_step=global_state.get("training_resumed_from_step", 0),
                resumed=True,
            )
        _monitor_training_logs(process)

    except Exception as e:
        log_message(TRAIN_LOG, f"Resume failed: {e}")
        log_message(TRAIN_PREP_LOG, traceback.format_exc())
        with global_state_lock:
            if generation == global_state.get("training_generation"):
                global_state["training_active"] = False
                global_state["training_stage"] = None
                global_state["training_process"] = None


def _terminate_training_subprocess(reason: str) -> bool:
    """Stop training whether its Popen handle belongs to this or a prior UI process."""
    proc = global_state.get("training_process")
    if not proc and not _managed_process_record("training"):
        return False
    log_message(TRAIN_LOG, f"Stopping training subprocess ({reason})")
    stopped = _stop_managed_process("training", proc)
    if proc is not None:
        try:
            proc.wait(timeout=1)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass
    global_state["training_active"] = False
    global_state["training_stage"] = None
    global_state["training_process"] = None
    return stopped


@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    """Stop dataset conversion or training, whichever is currently running."""
    global global_state

    if not global_state["training_active"] and not _managed_process_record("training"):
        return jsonify({"status": "error", "message": "No training running."}), 400

    try:
        with global_state_lock:
            global_state["cancel_requested"] = True
            cancel_event = global_state.get("training_cancel_event")
            if cancel_event is not None:
                cancel_event.set()
            # Invalidate this run before Start becomes available again.
            global_state["training_generation"] += 1
        _terminate_training_subprocess("stopped by user")
        log_message(TRAIN_LOG, "Training stopped by user")

        return jsonify({"status": "success", "message": "Training stopped."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Stop error: {str(e)}"}), 500

@app.route('/api/train/reset', methods=['POST'])
def reset_training():
    """Clear transient training state without deleting saved checkpoints or outputs."""
    with global_state_lock:
        if global_state["training_active"] or _managed_process_record("training"):
            return jsonify({"status": "error", "message": "Stop the active training run before resetting."}), 409
        global_state["training_generation"] += 1
        cancel_event = global_state.get("training_cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        global_state.update(
            training_process=None,
            training_active=False,
            training_policy=None,
            training_stage=None,
            cancel_requested=False,
            training_cancel_event=None,
            current_step=0,
            total_steps=0,
            training_resumed_from_step=0,
            last_output_dir=None,
        )
        _clear_managed_process("training")
    clear_log(TRAIN_LOG)
    return jsonify({"status": "success", "message": "Training environment reset. Saved outputs were kept."})

@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """Get training status and progress."""
    record = _managed_process_record("training")
    process = global_state.get("training_process")
    if record and process is None:
        global_state["training_active"] = True
        global_state["training_policy"] = record.get("policy")
        global_state["training_stage"] = record.get("stage", "training")
        global_state["last_output_dir"] = record.get("output_dir")
        global_state["total_steps"] = int(record.get("total_steps") or 0)
        global_state["current_step"] = int(record.get("current_step") or 0)
        if global_state["total_steps"] > 0:
            try:
                with open(TRAIN_LOG, "rb") as log_file:
                    log_file.seek(0, os.SEEK_END)
                    log_file.seek(max(0, log_file.tell() - 65536))
                    recent_log = log_file.read().decode("utf-8", errors="replace")
                for line in reversed(recent_log.splitlines()):
                    step = _training_step_from_line(
                        line,
                        global_state["total_steps"],
                        int(record.get("resumed_from_step") or 0),
                    )
                    if step is not None:
                        global_state["current_step"] = max(global_state["current_step"], step)
                        record["current_step"] = global_state["current_step"]
                        processes = _read_managed_processes()
                        if processes.get("training", {}).get("pid") == record.get("pid"):
                            processes["training"] = record
                            _write_managed_processes(processes)
                        break
            except OSError:
                pass
        stats = {"status": "running"}
    elif process is None and global_state["training_active"]:
        # Preparation/validation and the brief window between claim and Popen() run
        # in this Flask worker before a managed child exists. Keep that claimed job
        # active so polling cannot cancel its launch; the real "stopped" signal is
        # owned by _monitor_training_logs' finally block, not this status endpoint.
        stats = {"status": "preparing" if global_state.get("training_stage") in ("converting", "preparing") else "starting"}
    else:
        stats = get_process_stats(process)
        if not record and process is None and not global_state["training_active"]:
            global_state["training_stage"] = None
    progress = 0
    if global_state["total_steps"] > 0:
        progress = min(100, (global_state["current_step"] / global_state["total_steps"]) * 100)

    last_output_dir = global_state["last_output_dir"]
    retrain = {"available": False, "name": None, "latest_step": None}
    if last_output_dir and os.path.isdir(last_output_dir):
        steps = _list_run_checkpoint_steps(last_output_dir)
        retrain = {
            "available": bool(steps),
            "name": os.path.basename(last_output_dir),
            "latest_step": steps[0] if steps else None,
        }

    return jsonify({
        "active": global_state["training_active"],
        "policy": global_state["training_policy"],
        "stage": global_state["training_stage"],
        "current_step": global_state["current_step"],
        "total_steps": global_state["total_steps"],
        "progress": progress,
        "stats": stats,
        "retrain": retrain,
    })

@app.route('/api/train/logs', methods=['GET'])
def get_training_logs():
    """Stream training logs in real-time."""
    return Response(
        stream_log(TRAIN_LOG, active_check=lambda: global_state["training_active"]),
        mimetype='text/event-stream',
    )

# ════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - Inference
# ════════════════════════════════════════════════════════════════════════════════════

def _csv(values):
    """Format numeric metadata for eval_daksha.py's stable CLI interface."""
    return ",".join(f"{float(value):.6g}" for value in values)


def _dataset_instruction(dataset_root):
    """Return the recorded task text when raw task metadata is available."""
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        for line in tasks_path.read_text().splitlines():
            try:
                task = json.loads(line)
                value = task.get("task") or task.get("task_description")
                if value:
                    return str(value)
            except json.JSONDecodeError:
                continue
    return "task1"  # Existing Daksha datasets use this task when no text metadata remains.


def _resolve_checkpoint_dataset_root(dataset_cfg):
    """Find the training dataset on this machine, even if the checkpoint moved.

    Checkpoints store an absolute ``dataset.root`` from the training host. When
    inference runs on another system that path may not exist, so try the saved
    local repo ID under this UI's configured DATASET_ROOT before failing.
    """
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}

    candidates = []
    saved_root = dataset_cfg.get("root")
    if saved_root:
        candidates.append(Path(saved_root).expanduser())

    repo_id = str(dataset_cfg.get("repo_id") or "")
    if repo_id.startswith("local/"):
        candidates.append(Path(DATASET_ROOT) / repo_id.split("/", 1)[1])

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "meta" / "info.json").is_file() and (resolved / "meta" / "stats.json").is_file():
            return resolved

    tried = ", ".join(str(path.resolve()) for path in candidates) or "no local dataset paths"
    raise ValueError(
        "Training dataset metadata is unavailable. Checked: " + tried
        + ". Copy the checkpoint's training dataset under DATASET_ROOT or set DAKSHA_DATASET_ROOT."
    )


def _bridge_joint_keys(
    dataset_root: Path, names: list, modality_section: dict, label: str, robot_type: str | None
) -> list:
    """Return ROS-facing joint labels in the policy vector's exact index order.

    Generic Daksha joint_1..joint_16 feature names do not carry arm identity.
    For these datasets, the robot's inference layout is left arm/gripper at 0..7
    followed by right arm/gripper at 8..15. Named feature keys remain authoritative;
    modality ranges are used for other generic layouts.
    """
    if len(names) != 16:
        raise ValueError(f"Training dataset {label} must contain exactly 16 names")

    generic_names = all(
        name == f"joint_{index}" or name == f"follower_joint_{index}"
        for index, name in enumerate(names, start=1)
    )
    if generic_names and robot_type in {
        "bimanual_leader_follower",
        "leader_follower_6dof_4camera",
    }:
        # The selected Daksha inference setup maps left arm/gripper to indices
        # 0..7 and right arm/gripper to indices 8..15. Some converted datasets carry
        # a conflicting modality.json; keep this explicit inference mapping stable.
        return [
            *(f"left_joint_{i}" for i in range(1, 8)),
            "left_gripper_left_joint",
            *(f"right_joint_{i}" for i in range(1, 8)),
            "right_gripper_right_joint",
        ]

    if sum("gripper" in name.lower() for name in names) == 2:
        return list(names)

    modality_path = dataset_root / "meta" / "modality.json"
    if not modality_section:
        raise ValueError(
            f"Training dataset {label} names do not identify the two grippers and "
            f"{modality_path} has no {label} mapping for robot_type={robot_type!r}"
        )

    ordered = [None] * 16
    for side in ("left", "right"):
        for group, expected_len in ((f"{side}_arm", 7), (f"{side}_gripper", 1)):
            bounds = modality_section.get(group)
            if not isinstance(bounds, dict):
                raise ValueError(f"{modality_path} has no {label}.{group} range")
            start, end = int(bounds["start"]), int(bounds["end"])
            if end - start != expected_len or start < 0 or end > 16:
                raise ValueError(f"Invalid {label}.{group} range in {modality_path}: {bounds}")
            if group.endswith("_arm"):
                generated = [f"{side}_joint_{i}" for i in range(1, 8)]
            else:
                generated = [f"{side}_gripper_{side}_joint"]
            for index, key in zip(range(start, end), generated):
                if ordered[index] is not None:
                    raise ValueError(f"Overlapping joint ranges in {modality_path}")
                ordered[index] = key

    if any(key is None for key in ordered):
        raise ValueError(f"{modality_path} does not describe all 16 {label} dimensions")
    return ordered


def _checkpoint_camera_keys(pretrained_dir: Path, dataset_root: Path | None = None) -> list[str]:
    """Read camera modalities from checkpoint inputs, then its training dataset."""
    config = json.loads((pretrained_dir / "config.json").read_text())
    features = config.get("input_features") or {}
    cameras = [
        name.removeprefix("observation.images.")
        for name, feature in features.items()
        if name.startswith("observation.images.")
        and isinstance(feature, dict)
        and feature.get("type") == "VISUAL"
    ]
    if cameras:
        return cameras

    if dataset_root is not None:
        info_path = dataset_root / "meta" / "info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            cameras = info.get("camera_keys") or info.get("video_keys") or []
            if not cameras:
                cameras = [
                    key for key, feature in (info.get("features") or {}).items()
                    if key.startswith("observation.images.")
                    and isinstance(feature, dict)
                    and feature.get("dtype") == "video"
                ]
            if cameras:
                return [key.removeprefix("observation.images.") for key in cameras]

    # Legacy checkpoints did not persist image feature names in config or dataset info.
    return ["zed_right", "zed_left", "wrist_right", "wrist_left"]


def _camera_ros_topic(camera_key: str) -> str:
    """Map a checkpoint camera modality to its ROS image topic base name."""
    topics = {
        "world": "/camera/world/color/image_raw",
        "zed_right": "/zed/zed_node/right/color/rect/image",
        "zed_left": "/zed/zed_node/left/color/rect/image",
        "wrist_right": "/right/camera/color/image_raw",
        "right_wrist": "/right/camera/color/image_raw",
        "wrist_left": "/left/camera/color/image_raw",
        "left_wrist": "/left/camera/color/image_raw",
    }
    try:
        return topics[camera_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported checkpoint camera input: {camera_key}") from exc


def _compute_max_step_delta(
    dataset_root: Path, action_names: list, control_hz: float, dataset_fps: float,
    *, include_motion: bool = False,
) -> list | tuple[list, list]:
    """Limit per-tick steps using dataset motion and a conservative speed ceiling.

    LeRobot v3.0 datasets store every episode's frames concatenated in one shared
    parquet file per chunk (not one file per episode), so a flat np.diff across the
    whole column would fabricate a huge fake delta at each of the (total_episodes - 1)
    episode boundaries. Grouping by the row's own ``episode_index`` before diffing
    avoids that -- mirrors how model_validation/vla_eval/datasets/lerobot_custom.py
    respects episode boundaries in this same file layout, without its video decoding.
    """
    chunk_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not chunk_files:
        raise ValueError(f"No dataset parquet files found under {dataset_root / 'data'}")

    frames = pd.concat(
        [pd.read_parquet(f, columns=["action", "episode_index", "frame_index"]) for f in chunk_files],
        ignore_index=True,
    )

    deltas = []
    for _, episode in frames.groupby("episode_index", sort=False):
        actions = np.stack(episode.sort_values("frame_index")["action"].to_numpy())
        if len(actions) > 1:
            deltas.append(np.abs(np.diff(actions, axis=0)))
    if not deltas:
        raise ValueError(f"No multi-frame episodes found under {dataset_root} to compute action deltas from")

    p999 = np.percentile(np.concatenate(deltas, axis=0), 99.9, axis=0)
    # Keep a small minimum step for useful motion, but cap the resulting commanded
    # speed. Dataset-only limits can permit >3 rad/s from an unusual recording.
    floor = np.array([0.06 if "gripper" in name else 0.02 for name in action_names])
    rate_scale = (dataset_fps / control_hz) if control_hz else 1.0
    max_speed_rad_s = 0.30
    max_step = np.minimum(
        np.maximum(1.5 * p999 * rate_scale, floor),
        max_speed_rad_s / control_hz if control_hz else max_speed_rad_s,
    ).round(5).tolist()
    if include_motion:
        return max_step, p999.tolist()
    return max_step


def _detect_active_arms(action_keys: list[str], action_motion_p999: list[float]) -> list[str]:
    """Detect arms with real within-episode movement, ignoring episode reset offsets."""
    if len(action_keys) != 16 or len(action_motion_p999) != 16:
        raise ValueError("Active-arm detection requires 16 action keys and motion values")

    active = []
    for side in ("left", "right"):
        indices = [i for i, key in enumerate(action_keys) if side in key.lower()]
        if not indices:
            raise ValueError(f"Cannot detect {side} arm activity from checkpoint action keys")
        # The datasets are quantized around 0.0015 rad; 0.005 rad filters this
        # recording noise while retaining actual in-episode arm/gripper motion.
        if any(float(action_motion_p999[i]) > 0.005 for i in indices):
            active.append(side)
    if not active:
        raise ValueError(
            "Could not detect arm movement in the training dataset (99.9th-percentile "
            "within-episode action deltas are at or below 0.005 rad)"
        )
    return active


def _checkpoint_inference_metadata(checkpoint_path, control_hz=None, action_horizon=None):
    """Load bridge settings from the checkpoint's *own* training dataset.

    The dataset root and policy settings are saved by LeRobot in train_config.json.
    Bounds are calculated in ``features.action.names`` order, then the same order is
    passed to the bridge for state, actions, and clamps.  This avoids the historical
    left/right arm swap caused by assuming a fixed joint layout in the UI.

    ``control_hz``/``action_horizon`` are optional operator overrides (from the
    Inference tab); left as None they default to the dataset's recorded fps and the
    checkpoint's trained chunk_size, exactly as before this override support existed.
    """
    pretrained_dir = resolve_pretrained_model_dir(checkpoint_path)
    config_path = pretrained_dir / "train_config.json"
    if not config_path.is_file():
        raise ValueError(f"Checkpoint has no train_config.json: {pretrained_dir}")
    train_config = json.loads(config_path.read_text())
    policy_type = (train_config.get("policy") or {}).get("type")
    dataset_cfg = train_config.get("dataset") or {}
    dataset_root = _resolve_checkpoint_dataset_root(dataset_cfg)
    info_path, stats_path = dataset_root / "meta" / "info.json", dataset_root / "meta" / "stats.json"
    info, stats = json.loads(info_path.read_text()), json.loads(stats_path.read_text())
    dataset_features = info.get("features") or {}
    action_names = ((dataset_features.get("action") or {}).get("names"))
    state_names = ((dataset_features.get("observation.state") or {}).get("names"))
    if not isinstance(action_names, list) or len(action_names) != 16:
        raise ValueError("Training dataset must provide exactly 16 action names for the Daksha bridge")
    if not isinstance(state_names, list) or len(state_names) != 16:
        raise ValueError("Training dataset must provide exactly 16 observation.state names")

    modality_path = dataset_root / "meta" / "modality.json"
    modality_doc = json.loads(modality_path.read_text()) if modality_path.is_file() else {}
    robot_type = info.get("robot_type")
    bridge_state_keys = _bridge_joint_keys(
        dataset_root, state_names, modality_doc.get("state") or {}, "state", robot_type
    )
    bridge_action_keys = _bridge_joint_keys(
        dataset_root, action_names, modality_doc.get("action") or {}, "action", robot_type
    )

    try:
        action_stats = stats["action"]
        state_stats = stats["observation.state"]
        lo = [min(float(action_stats["min"][i]), float(state_stats["min"][i])) for i in range(16)]
        hi = [max(float(action_stats["max"][i]), float(state_stats["max"][i])) for i in range(16)]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Training dataset statistics are incomplete: {exc}") from exc

    # A small data-derived margin keeps normal predictions inside the learned range.
    margin = [0.01 if "gripper" in name else max(0.1 * (high - low), 0.05)
              for name, low, high in zip(bridge_action_keys, lo, hi)]

    policy_cfg = train_config.get("policy") or {}
    chunk_size = int(policy_cfg.get("chunk_size") or policy_cfg.get("n_action_steps") or 16)
    dataset_fps = float(info.get("fps") or 30)
    # "is not None", not truthiness -- 0 is a real (invalid) value here, not "unset",
    # and must reach the validation below instead of silently becoming the default.
    effective_control_hz = float(control_hz) if control_hz is not None else dataset_fps
    effective_action_horizon = int(action_horizon) if action_horizon is not None else chunk_size
    if not (1 <= effective_action_horizon <= chunk_size):
        raise ValueError(
            f"action_horizon must be between 1 and this checkpoint's chunk_size ({chunk_size}), "
            f"got {effective_action_horizon}"
        )
    if effective_control_hz <= 0:
        raise ValueError(f"control_hz must be positive, got {effective_control_hz}")
    max_step, action_motion_p999 = _compute_max_step_delta(
        dataset_root, bridge_action_keys, effective_control_hz, dataset_fps, include_motion=True
    )
    # The training dataset determines which arms this checkpoint learned to
    # control. Apply the same rule to every policy bridge so a static arm is
    # held at its measured position during inference.
    active_arms = _detect_active_arms(bridge_action_keys, action_motion_p999)

    return {
        "dataset_root": str(dataset_root),
        "lang_instruction": _dataset_instruction(dataset_root),
        "policy_type": policy_type,
        "dataset_fps": dataset_fps,
        "chunk_size": chunk_size,
        "control_hz": effective_control_hz,
        "action_horizon": effective_action_horizon,
        "state_obs_keys_csv": ",".join(bridge_state_keys),
        "action_keys_csv": ",".join(bridge_action_keys),
        "action_names": bridge_action_keys,
        "camera_keys": _checkpoint_camera_keys(pretrained_dir, dataset_root),
        "active_arms": active_arms,
        "joint_min_csv": _csv([low - pad for low, pad in zip(lo, margin)]),
        "joint_max_csv": _csv([high + pad for high, pad in zip(hi, margin)]),
        "max_step_delta_csv": _csv(max_step),
    }

@app.route('/api/inference/checkpoint_metadata', methods=['GET'])
def get_checkpoint_inference_metadata():
    """Read-only preview of the bridge settings a checkpoint would start with --
    detected policy type, dataset fps/chunk_size, and the computed safety limits --
    without starting any process. Powers the Inference tab's form before Start is hit.
    """
    checkpoint_path = request.args.get('checkpoint')
    if not checkpoint_path:
        return jsonify({"status": "error", "message": "Invalid checkpoint path: None"}), 400
    try:
        checkpoint_path = str(resolve_checkpoint_path(checkpoint_path))
    except PermissionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 403
    if not os.path.exists(checkpoint_path):
        return jsonify({"status": "error", "message": f"Invalid checkpoint path: {checkpoint_path}"}), 400

    pretrained_dir = resolve_pretrained_model_dir(checkpoint_path)
    if not (pretrained_dir / "config.json").exists():
        return jsonify({
            "status": "error",
            "message": f"No pretrained_model found under checkpoint: {checkpoint_path} "
                       f"(looked in {pretrained_dir})",
        }), 400

    try:
        control_hz = request.args.get('control_hz', type=float)
        action_horizon = request.args.get('action_horizon', type=int)
        metadata = _checkpoint_inference_metadata(checkpoint_path, control_hz, action_horizon)
        metadata["policy_type"] = metadata["policy_type"] or json.loads(
            (pretrained_dir / "config.json").read_text()
        ).get("type")
        return jsonify({"status": "success", **metadata})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Metadata error: {str(e)}"}), 500


@app.route('/api/inference/start_server', methods=['POST'])
def start_inference_server():
    """Start policy inference server."""
    global global_state

    # Cheap early-exit only -- see the atomic check-and-launch under global_state_lock
    # below, which is the real guard against two concurrent start requests both
    # passing this check and launching two server processes on the same checkpoint.
    if global_state["server_process"] and global_state["server_process"].poll() is None:
        return jsonify({"status": "error", "message": "Server already running."}), 400

    try:
        data = request.json
        policy_type = data.get('policy_type', 'generic')
        checkpoint_path = data.get('checkpoint_path')
        port = data.get('port') or find_available_port()

        if not checkpoint_path:
            return jsonify({"status": "error", "message": "Invalid checkpoint path: None"}), 400
        try:
            checkpoint_path = str(resolve_checkpoint_path(checkpoint_path))
        except PermissionError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 403
        if not os.path.exists(checkpoint_path):
            return jsonify({"status": "error", "message": f"Invalid checkpoint path: {checkpoint_path}"}), 400

        pretrained_dir = resolve_pretrained_model_dir(checkpoint_path)
        if not (pretrained_dir / "config.json").exists():
            return jsonify({
                "status": "error",
                "message": f"No pretrained_model found under checkpoint: {checkpoint_path} "
                           f"(looked in {pretrained_dir})",
            }), 400
        metadata = _checkpoint_inference_metadata(
            checkpoint_path, data.get('control_hz'), data.get('action_horizon')
        )

        saved_policy = json.loads((pretrained_dir / "config.json").read_text()).get("type")
        if saved_policy == "smolvla":
            if not _is_checkpoint_complete(pretrained_dir.parent):
                raise ValueError("SmolVLA checkpoint is incomplete; select a completed saved step")
        # The checkpoint's own config -- never the UI's policy-type filter dropdown --
        # decides which policy type actually loads. See the ACT_BRIDGE_SCRIPT /
        # DIFFUSION_BRIDGE_SCRIPT comment above for why that also decides which bridge
        # script start_bridge() below picks.
        policy_type = saved_policy or policy_type
        policy_label = {"act": "ACT", "diffusion": "Diffusion", "smolvla": "SmolVLA"}.get(
            policy_type, policy_type
        )
        metadata["policy_type"] = policy_type
        metadata["pretrained_dir"] = str(pretrained_dir)
        metadata["device"] = data.get("device", "cuda")

        print(
            f"[daksha-ui] Selected checkpoint step: {checkpoint_path}\n"
            f"[daksha-ui] Resolved model directory: {pretrained_dir}\n"
            f"[daksha-ui] Detected policy type: {policy_type}",
            flush=True,
        )

        if policy_type == "diffusion":
            # eval_daksha_diffusion.py loads this checkpoint directly in-process instead
            # of going through the generic ZMQ server (see its module docstring: the
            # server never populates DiffusionPolicy's n_obs_steps observation-history
            # queues, which breaks any checkpoint trained with n_obs_steps > 1) -- so
            # there is no separate server subprocess to start here. start_bridge() does
            # the actual checkpoint loading for Diffusion checkpoints.
            if not os.path.exists(DIFFUSION_BRIDGE_SCRIPT):
                return jsonify(
                    {"status": "error", "message": f"Bridge script not found: {DIFFUSION_BRIDGE_SCRIPT}"}
                ), 404

            clear_log(SERVER_LOG)
            log_message(SERVER_LOG, "Starting Diffusion inference (direct checkpoint load; no separate policy server).")
            log_message(SERVER_LOG, f"Checkpoint: {pretrained_dir}")
            log_message(SERVER_LOG, f"Training dataset: {metadata['dataset_root']}")
            log_message(SERVER_LOG, "Start the ROS2 bridge next -- it loads this checkpoint directly.")

            with global_state_lock:
                if global_state["server_process"] and global_state["server_process"].poll() is None:
                    return jsonify({"status": "error", "message": "Server already running."}), 400
                global_state["server_process"] = None
                global_state["server_port"] = None
                global_state["inference_metadata"] = metadata
                global_state["inference_active"] = True
                Path(PROCESS_STATE_PATH).with_name("inference_metadata.json").write_text(json.dumps(metadata))

            return jsonify({
                "status": "success",
                "message": "Diffusion checkpoint ready -- it loads directly when the bridge starts "
                           "(no separate server process).",
                "port": None,
            })

        # The ZMQ server (serve_policy_zmq.py) reads the policy type from the
        # checkpoint's own config, so one script serves ACT and SmolVLA alike.
        serve_script = f"{DAKSHA_ROOT}/serve_policy_zmq.py"
        if saved_policy == "smolvla":
            serve_script = f"{DAKSHA_ROOT}/smolvla/serve_smolvla_zmq.py"
        if not os.path.exists(serve_script):
            return jsonify({"status": "error", "message": f"Serve script not found: {serve_script}"}), 404

        clear_log(SERVER_LOG)
        log_message(SERVER_LOG, f"Starting {policy_label} inference server")
        log_message(SERVER_LOG, f"Checkpoint: {pretrained_dir}")
        log_message(SERVER_LOG, f"Training dataset: {metadata['dataset_root']}")
        log_message(SERVER_LOG, f"Port: {port}")

        python_argv, extra_env = resolve_repo_python()
        env = os.environ.copy()
        env.update(extra_env)
        # The server prints an explicit CHECKPOINT LOADED confirmation only after
        # model weights and processors are ready. Keep it unbuffered so both the UI
        # log panel and the terminal receive that confirmation immediately.
        env["PYTHONUNBUFFERED"] = "1"

        cmd = python_argv + [
            serve_script,
            f'--policy-path={pretrained_dir}',
            f'--host=0.0.0.0',
            f'--port={port}',
            f'--device={data.get("device", "cuda")}',
            f'--task-default={metadata["lang_instruction"]}',
            f'--action-horizon={metadata["action_horizon"]}',
        ]
        if saved_policy == "smolvla":
            info = json.loads((Path(metadata["dataset_root"]) / "meta/info.json").read_text())
            cmd.append(f'--robot-type={info.get("robot_type") or "leader_follower_6dof_4camera"}')
            env["PYTHONUNBUFFERED"] = "1"

        with global_state_lock:
            if ((global_state["server_process"] and global_state["server_process"].poll() is None)
                    or _managed_process_record("inference")):
                return jsonify({"status": "error", "message": "Server already running."}), 400

            with open(SERVER_LOG, "a", encoding="utf-8", buffering=1) as child_log:
                process = subprocess.Popen(
                    cmd, env=env, stdout=child_log, stderr=subprocess.STDOUT,
                    cwd=REPO_ROOT, start_new_session=True,
                )

            global_state["server_process"] = process
            global_state["server_port"] = port
            global_state["inference_metadata"] = metadata
            global_state["inference_active"] = True
            _register_managed_process("inference", process, port=port, metadata=metadata)

        threading.Thread(target=_monitor_server_logs, args=(process,), daemon=True).start()

        return jsonify({
            "status": "success",
            "message": f"Inference server started on port {port}",
            "port": port,
        })

    except ValueError as e:
        return jsonify({"status": "error", "message": f"Checkpoint metadata error: {e}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server start error: {str(e)}"}), 500

def _monitor_server_logs(process):
    """Wait for the file-logged inference server and clear its durable record."""
    returncode = process.wait()
    with global_state_lock:
        if global_state["server_process"] is process:
            global_state["inference_active"] = False
            _clear_managed_process("inference", process)
    if returncode:
        log_message(SERVER_LOG, f"Inference server exited (code {returncode})")

def _stop_bridge_process(reason: str) -> None:
    """Stop the ROS2 bridge, including one recovered after the Flask app restarted."""
    proc = global_state.get("bridge_process")
    _stop_managed_process("bridge", proc, timeout=5)
    if proc is not None:
        try:
            proc.wait(timeout=1)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass
    global_state["bridge_process"] = None
    global_state["bridge_active"] = False
    log_message(BRIDGE_LOG, reason)


@app.route('/api/inference/stop_server', methods=['POST'])
def stop_inference_server():
    """Stop the inference server (and the robot bridge, if it's running against it)."""
    global global_state

    try:
        if global_state["bridge_active"] or _managed_process_record("bridge"):
            # The bridge without a server behind it just spins on ping() and holds --
            # stopped anyway for safety/clarity rather than leaving a robot-facing
            # process attached to a server the operator just told us to kill.
            _stop_bridge_process("Bridge stopped: policy server was stopped.")

        server_proc = global_state.get("server_process")
        if server_proc is not None or _managed_process_record("inference"):
            _stop_managed_process("inference", server_proc, timeout=5)
            if server_proc is not None:
                try:
                    server_proc.wait(timeout=1)
                except (subprocess.TimeoutExpired, ChildProcessError):
                    pass

        global_state["inference_active"] = False
        global_state["server_port"] = None
        global_state["inference_metadata"] = None
        Path(PROCESS_STATE_PATH).with_name("inference_metadata.json").unlink(missing_ok=True)
        log_message(SERVER_LOG, "Server stopped by user")

        return jsonify({"status": "success", "message": "Server stopped."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Stop error: {str(e)}"}), 500

@app.route('/api/inference/logs', methods=['GET'])
def get_server_logs():
    """Stream server logs."""
    policy_type = (global_state.get("inference_metadata") or {}).get("policy_type")

    def label_server_log(message):
        if policy_type == "act":
            return message.replace("[serve_smolvla_zmq]", "[ACT policy server]")
        return message

    return Response(
        stream_log(
            SERVER_LOG,
            active_check=lambda: global_state["inference_active"],
            line_transform=label_server_log,
        ),
        mimetype='text/event-stream',
    )

# ════════════════════════════════════════════════════════════════════════════════════
# API ROUTES - Inference: ROS2 robot bridge
#
# start_inference_server()/stop_inference_server() above only manage the ZMQ policy
# server (examples/daksha/serve_policy_zmq.py / serve_smolvla_zmq.py) for ACT/SmolVLA
# checkpoints -- it loads the checkpoint and answers get_action requests, but never
# touches the robot; Diffusion checkpoints skip that server entirely (see
# start_inference_server). The routes below manage the actual ROS2 bridge --
# eval_daksha_act.py, eval_daksha_smolvla.py, or eval_daksha_diffusion.py, picked by the
# checkpoint's own policy type (ACT_BRIDGE_SCRIPT / SMOLVLA_BRIDGE_SCRIPT /
# DIFFUSION_BRIDGE_SCRIPT) -- that reads joint/camera topics,
# gets an action (from the server over ZMQ, or in-process for Diffusion), and publishes
# commands. It is a separate process (and separate Python interpreter -- see
# build_ros_bridge_cmd) so starting it never blocks on / is blocked by the server's own
# checkpoint-loading subprocess. For ACT/SmolVLA, the bridge itself waits on the
# server's ping() before it commands anything (see wait_until_ready in
# eval_daksha_act.py / eval_daksha_smolvla.py), so it is safe to start it here right after
# start_inference_server() returns, even though the checkpoint may still be loading --
# matching the same server-then-bridge order run_inference.sh uses from the command
# line.
# ════════════════════════════════════════════════════════════════════════════════════

def _monitor_bridge_logs(process):
    """Wait for the file-logged robot bridge and clear its durable record."""
    returncode = process.wait()
    with global_state_lock:
        if global_state["bridge_process"] is process:
            global_state["bridge_active"] = False
            _clear_managed_process("bridge", process)
    if returncode:
        log_message(BRIDGE_LOG, f"Bridge exited (code {returncode})")


@app.route('/api/inference/start_bridge', methods=['POST'])
def start_bridge():
    """Start the ROS2 bridge against the currently running (or about-to-be-ready) policy server."""
    global global_state

    # Cheap early-exit only -- see the atomic check-and-launch under global_state_lock
    # below, which is the real guard against two concurrent start requests.
    if ((global_state["bridge_process"] and global_state["bridge_process"].poll() is None)
            or _managed_process_record("bridge")):
        return jsonify({"status": "error", "message": "Bridge already running."}), 400

    data = request.json or {}
    metadata = global_state.get("inference_metadata")
    if not metadata:
        return jsonify({
            "status": "error",
            "message": "No checkpoint metadata is loaded. Start the policy server from a checkpoint first.",
        }), 400

    policy_type = metadata.get('policy_type')
    is_diffusion = policy_type == "diffusion"

    policy_port = None
    if not is_diffusion:
        policy_port = data.get('policy_port') or global_state.get('server_port')
        if not policy_port:
            return jsonify({
                "status": "error",
                "message": "No policy_port given and no policy server is currently running. "
                           "Start the policy server first (or pass policy_port explicitly).",
            }), 400

    if is_diffusion:
        bridge_script = DIFFUSION_BRIDGE_SCRIPT
    elif policy_type == "smolvla":
        bridge_script = SMOLVLA_BRIDGE_SCRIPT
    else:
        bridge_script = ACT_BRIDGE_SCRIPT
    if not os.path.exists(bridge_script):
        return jsonify({"status": "error", "message": f"Bridge script not found: {bridge_script}"}), 404

    try:
        if is_diffusion:
            pretrained_dir = metadata.get('pretrained_dir')
            if not pretrained_dir:
                return jsonify({
                    "status": "error",
                    "message": "No checkpoint path recorded for this Diffusion checkpoint. "
                               "Reload it from the Checkpoint dropdown and try again.",
                }), 400
            args = [
                '--policy-path', pretrained_dir,
                '--device', str(data.get('device') or metadata.get('device') or 'cuda'),
                '--control-hz', str(metadata['control_hz']),
                '--action-horizon', str(metadata['action_horizon']),
                '--lang-instruction', metadata['lang_instruction'],
                '--joint-min-csv=' + str(metadata['joint_min_csv']),
                '--joint-max-csv=' + str(metadata['joint_max_csv']),
                '--max-step-delta-csv=' + str(metadata['max_step_delta_csv']),
                '--state-obs-keys-csv', metadata['state_obs_keys_csv'],
                '--action-keys-csv', metadata['action_keys_csv'],
            ]
        else:
            args = [
                '--policy-host', str(data.get('policy_host') or '127.0.0.1'),
                '--policy-port', str(policy_port),
                '--server-wait-timeout-s', str(data.get('server_wait_timeout_s') or 120),
                '--control-hz', str(metadata['control_hz']),
                '--action-horizon', str(metadata['action_horizon']),
                '--lang-instruction', metadata['lang_instruction'],
                '--joint-min-csv=' + str(metadata['joint_min_csv']),
                '--joint-max-csv=' + str(metadata['joint_max_csv']),
                '--max-step-delta-csv=' + str(metadata['max_step_delta_csv']),
                '--state-obs-keys-csv', metadata['state_obs_keys_csv'],
                '--action-keys-csv', metadata['action_keys_csv'],
            ]
        if policy_type in ("act", "smolvla", "diffusion"):
            args.append("--active-sides-csv=" + ",".join(metadata["active_arms"]))
        if policy_type in ("act", "diffusion"):
            # Each checkpoint declares its own image feature set. Pass only those
            # camera inputs to the bridge; `world` resolves to Daksha's world view.
            for camera_key in metadata.get("camera_keys") or []:
                try:
                    camera_topic = _camera_ros_topic(camera_key)
                except ValueError as exc:
                    return jsonify({"status": "error", "message": str(exc)}), 400
                args.extend(["--camera-topic", f"{camera_key}={camera_topic}"])
            if not data.get("image_transport"):
                # Daksha's camera publishers expose CompressedImage topics by
                # default, including /camera/world/color/image_raw/compressed.
                args.extend(["--image-transport", "compressed"])

        if policy_type == "smolvla":
            # Route each checkpoint camera modality to a physical ROS camera source.
            # World is the robot's world camera; the left ZED slot is disabled for
            # checkpoints trained with only world + wrist cameras.
            camera_sources = {
                "zed_right": "zed-right-camera-key",
                "world": "zed-right-camera-key",
                "zed_left": "zed-left-camera-key",
                "wrist_right": "right-wrist-camera-key",
                "right_wrist": "right-wrist-camera-key",
                "wrist_left": "left-wrist-camera-key",
                "left_wrist": "left-wrist-camera-key",
            }
            camera_args = {
                "zed-right-camera-key": "",
                "zed-left-camera-key": "",
                "right-wrist-camera-key": "",
                "left-wrist-camera-key": "",
            }
            for camera_key in metadata.get("camera_keys") or []:
                flag = camera_sources.get(camera_key)
                if flag is None:
                    return jsonify({
                        "status": "error",
                        "message": f"Unsupported SmolVLA camera input from checkpoint: {camera_key}",
                    }), 400
                if camera_args[flag]:
                    return jsonify({
                        "status": "error",
                        "message": f"Multiple checkpoint cameras map to ROS source {flag}: {camera_key}",
                    }), 400
                camera_args[flag] = camera_key
            args.extend(f"--{flag}={key}" for flag, key in camera_args.items())

        # Optional hardware transport overrides. Joint order always comes from the
        # checkpoint's dataset and cannot be changed by the browser.
        for field, flag in (
            ('publish_joint_order', '--publish-joint-order'),
            ('image_transport', '--image-transport'),
        ):
            if data.get(field):
                args += [flag, str(data[field])]

        clear_log(BRIDGE_LOG)
        if is_diffusion:
            log_message(BRIDGE_LOG, f"Starting ROS2 bridge -> loading Diffusion checkpoint "
                                     f"{pretrained_dir} directly")
        else:
            log_message(BRIDGE_LOG, f"Starting ROS2 bridge -> policy server at "
                                     f"{data.get('policy_host') or '127.0.0.1'}:{policy_port}")

        cmd = build_ros_bridge_cmd(bridge_script, args, needs_lerobot=is_diffusion)

        with global_state_lock:
            if ((global_state["bridge_process"] and global_state["bridge_process"].poll() is None)
                    or _managed_process_record("bridge")):
                return jsonify({"status": "error", "message": "Bridge already running."}), 400

            with open(BRIDGE_LOG, "a", encoding="utf-8", buffering=1) as child_log:
                process = subprocess.Popen(
                    cmd, stdout=child_log, stderr=subprocess.STDOUT,
                    cwd=REPO_ROOT, start_new_session=True,
                )

            global_state["bridge_process"] = process
            global_state["bridge_active"] = True
            _register_managed_process("bridge", process)

        threading.Thread(target=_monitor_bridge_logs, args=(process,), daemon=True).start()

        return jsonify({"status": "success", "message": "ROS2 bridge starting."})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Bridge start error: {str(e)}"}), 500


@app.route('/api/inference/stop_bridge', methods=['POST'])
def stop_bridge():
    """Stop the ROS2 bridge."""
    if not global_state["bridge_active"] and not _managed_process_record("bridge") and not (
        global_state["bridge_process"] and global_state["bridge_process"].poll() is None
    ):
        return jsonify({"status": "error", "message": "Bridge is not running."}), 400
    try:
        _stop_bridge_process("Bridge stopped by user")
        return jsonify({"status": "success", "message": "Bridge stopped."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Stop error: {str(e)}"}), 500


@app.route('/api/inference/bridge_logs', methods=['GET'])
def get_bridge_logs():
    """Stream bridge (ROS2 node) logs."""
    return Response(
        stream_log(BRIDGE_LOG, active_check=lambda: global_state["bridge_active"]),
        mimetype='text/event-stream',
    )

def _checkpoint_dataset_root(run_path):
    """Read the dataset path saved with a checkpoint, when its config is available."""
    try:
        pretrained = resolve_pretrained_model_dir(run_path)
        train_config = json.loads((pretrained / "train_config.json").read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return None

    dataset_config = train_config.get("dataset") or {}
    if not isinstance(dataset_config, dict):
        return None
    try:
        return str(_resolve_checkpoint_dataset_root(dataset_config))
    except ValueError:
        return None


@app.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    """List available checkpoints grouped by policy and task with size details."""
    checkpoints = []
    scan_roots = []
    seen_roots = set()
    for root in (OUTPUT_ROOT, MODELS_ROOT):
        resolved_root = Path(root).resolve()
        if resolved_root.is_dir() and resolved_root not in seen_roots:
            scan_roots.append(resolved_root)
            seen_roots.add(resolved_root)

    seen_runs = set()
    for checkpoint_root in scan_roots:
        for policy_dir in sorted(os.listdir(checkpoint_root)):
            policy_path = checkpoint_root / policy_dir
            if not policy_path.is_dir():
                continue

            for task_dir in sorted(os.listdir(policy_path)):
                task_path = policy_path / task_dir
                if not task_path.is_dir():
                    continue

                for run_dir in sorted(os.listdir(task_path)):
                    run_path = task_path / run_dir
                    if not run_path.is_dir():
                        continue
                    resolved_run = run_path.resolve()
                    if resolved_run in seen_runs:
                        continue
                    seen_runs.add(resolved_run)

                    size_bytes = get_dir_size(str(run_path))
                    steps = _list_run_checkpoint_steps(run_path)
                    checkpoints.append({
                        "name": run_dir,
                        "path": str(run_path),
                        "model": policy_dir,
                        "task": task_dir,
                        "dataset_root": _checkpoint_dataset_root(run_path),
                        "size_bytes": size_bytes,
                        "size_text": format_bytes(size_bytes),
                        "created": datetime.fromtimestamp(run_path.stat().st_ctime).isoformat(),
                        # Saved step-checkpoints under this run, newest first (see
                        # CHECKPOINT_KEEP_LAST -- at most that many exist at once).
                        # "Resume" always continues from steps[0] (the `last` symlink's target).
                        "steps": steps,
                        # Inference must load the operator-selected saved step, not
                        # silently resolve this run root to whichever step is newest.
                        "checkpoint_options": _checkpoint_step_options(run_path),
                        "latest_step": steps[0] if steps else None,
                    })

    # Include runs created by the standalone SmolVLA launcher as well as UI runs.
    smolvla_runs = {Path(SMOLVLA_DEFAULT_RUN), *Path(SMOLVLA_OUTPUT_ROOT).glob("*")}
    listed_paths = {Path(item["path"]).resolve() for item in checkpoints}
    for run in sorted(smolvla_runs):
        if run.resolve() in listed_paths:
            continue
        if not run.is_dir() or not (run / "checkpoints").is_dir():
            continue
        try:
            pretrained = resolve_pretrained_model_dir(run)
            if json.loads((pretrained / "config.json").read_text()).get("type") != "smolvla":
                continue
            train_config = json.loads((pretrained / "train_config.json").read_text())
        except (OSError, ValueError):
            continue
        size_bytes = get_dir_size(str(run))
        steps = _list_run_checkpoint_steps(str(run))
        checkpoints.append({
            "name": run.name, "path": str(run), "model": "smolvla",
            "task": Path(train_config.get("dataset", {}).get("root") or run.name).name,
            "dataset_root": (
                str(Path(train_config["dataset"]["root"]).expanduser().resolve())
                if train_config.get("dataset", {}).get("root") else None
            ),
            "size_bytes": size_bytes, "size_text": format_bytes(size_bytes),
            "created": datetime.fromtimestamp(run.stat().st_ctime).isoformat(),
            "steps": steps, "latest_step": steps[0] if steps else None,
            "checkpoint_options": _checkpoint_step_options(run),
            "can_delete": False,
        })
    return jsonify(sorted(checkpoints, key=lambda x: x['created'], reverse=True))

@app.route('/api/checkpoints/delete', methods=['POST'])
def delete_checkpoint():
    """Permanently delete a checkpoint."""
    data = request.json or {}
    path = data.get('path', '').strip()

    if not path:
        return jsonify({"status": "error", "message": "No path provided."}), 400

    # A plain str.startswith(OUTPUT_ROOT) check is bypassable two ways: a sibling
    # directory whose name happens to share the OUTPUT_ROOT prefix (e.g.
    # "<OUTPUT_ROOT>_backup"), and "../" segments that pass the string check but get
    # resolved by the OS to somewhere else entirely once shutil.rmtree opens them.
    # Resolve first and check real containment, matching resolve_dataset_path().
    resolved = Path(path).expanduser().resolve()
    allowed_root = Path(OUTPUT_ROOT).resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        return jsonify({"status": "error", "message": "Invalid checkpoint path (security check failed)."}), 403
    path = str(resolved)

    if not os.path.exists(path):
        return jsonify({"status": "error", "message": f"Checkpoint not found: {path}"}), 404
    
    if not os.path.isdir(path):
        return jsonify({"status": "error", "message": f"Path is not a directory: {path}"}), 400
    
    try:
        checkpoint_name = os.path.basename(path)
        total_size = get_dir_size(path)
        size_gb = total_size / (1024**3)
        
        shutil.rmtree(path)
        log_message(TRAIN_LOG, f"Checkpoint deleted: {checkpoint_name} ({size_gb:.2f} GB)")
        
        return jsonify({
            "status": "success",
            "message": f"Checkpoint '{checkpoint_name}' permanently deleted ({size_gb:.2f} GB freed)",
        })
    except Exception as e:
        log_message(TRAIN_LOG, f"Error deleting checkpoint {path}: {str(e)}")
        return jsonify({"status": "error", "message": f"Deletion failed: {str(e)}"}), 500

# ════════════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ════════════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"status": "error", "message": "Internal server error"}), 500

def main():
    import argparse
    from werkzeug.serving import make_server

    # Re-attach the training log mirror for a run that survived a UI restart. Done here
    # (not in _recover_managed_processes, which runs at import) so the filtering helpers
    # defined later in this module are available.
    if global_state.pop("training_needs_log_reattach", False):
        record = _managed_process_record("training")
        if record:
            try:
                offset = _rebuild_training_display_log(record)
                _reattach_training_log_mirror(offset)
            except Exception as exc:
                log_message(TRAIN_PREP_LOG, f"Could not re-attach training log mirror: {exc}")

    arg_parser = argparse.ArgumentParser(description="Daksha VLA UI backend")
    arg_parser.add_argument('--host', default=os.environ.get('DAKSHA_HOST', '0.0.0.0'))
    arg_parser.add_argument('--port', type=int, default=int(os.environ.get('DAKSHA_PORT', 8104)))
    # BooleanOptionalAction (not plain store_true) so --no-debug can override an
    # inherited DAKSHA_DEBUG=true -- store_true only ever sets True when passed and
    # otherwise falls back to `default`, so with the env var set there would be no
    # way to force debug off for a single run short of unsetting the variable.
    arg_parser.add_argument(
        '--debug',
        action=argparse.BooleanOptionalAction,
        default=os.environ.get('DAKSHA_DEBUG', 'false').lower() == 'true',
    )
    argv = sys.argv
    if ROS2_AVAILABLE:
        argv = rclpy.utilities.remove_ros_args(args=argv)
    args = arg_parser.parse_args(argv[1:])

    # --debug enables Werkzeug's interactive debugger, which lets anyone who can reach
    # it execute arbitrary code on this machine. That's fine on loopback for a
    # developer's own troubleshooting session, but --host defaults to 0.0.0.0 for the
    # normal (non-debug) case of reaching this UI from elsewhere on the LAN -- so force
    # loopback-only whenever --debug is on, regardless of what --host was given, rather
    # than trusting the caller to remember to change it too.
    host = args.host
    if args.debug and host not in ('127.0.0.1', 'localhost', '::1'):
        print(f"[daksha-ui] --debug forces --host to 127.0.0.1 (was {host!r}); "
              f"the Werkzeug debugger must never be reachable from the network.")
        host = '127.0.0.1'

    # Closing app.py stops every child it started (validator UI/API, policy server, ROS
    # bridge, training) so no processes or ports are left behind. Training can be kept
    # alive across restarts by default (see shutdown_all_children).
    def _handle_shutdown_signal(signum, _frame):
        shutdown_all_children(signal.Signals(signum).name)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGHUP, _handle_shutdown_signal)  # terminal/VS Code window closed

    init_ros2()
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    server = make_server(host, args.port, app, threaded=True)

    display_host = host
    if host in ("0.0.0.0", "::", ""):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_socket:
                route_socket.connect(("192.0.2.1", 80))
                display_host = route_socket.getsockname()[0]
        except OSError:
            display_host = next(
                (
                    address.address
                    for addresses in psutil.net_if_addrs().values()
                    for address in addresses
                    if address.family == socket.AF_INET
                    and not address.address.startswith(("127.", "169.254."))
                ),
                "127.0.0.1",
            )
    print(f"Connect to UI: http://{display_host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
