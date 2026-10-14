#!/usr/bin/env python3
"""
Daksha VLA Inference Client
ZMQ client for serve_policy_zmq.py (see examples/daksha/smolvla/serve_smolvla_zmq.py
for the server-side implementation shared by every Daksha policy).

Wire protocol (msgpack over a ZMQ REQ/REP socket):
  request:  {"endpoint": "ping" | "reset" | "get_action", "data": {...}}
  response: "get_action" -> [{"action": np.ndarray}, {"horizon": int, "task": str}]
            "ping"/"reset" -> {"ok": True}  (or {"error": "..."} on failure)
  ndarrays are encoded as {"__ndarray_class__": True, "as_npy": <bytes>} via
  np.save/np.load, matching MsgSerializer in the server.
"""

import argparse
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import msgpack
import numpy as np
import zmq

# ════════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Matches DAKSHA_ARM_CONFIG["video"]["modality_keys"] in ui/app.py and
# examples/daksha/*/meta/modality.json.
DEFAULT_CAMERA_KEYS = ["zed_right", "zed_left", "wrist_right", "wrist_left"]

# ════════════════════════════════════════════════════════════════════════════════════
# MSGPACK NDARRAY ENCODING (must match MsgSerializer in serve_smolvla_zmq.py)
# ════════════════════════════════════════════════════════════════════════════════════

def _encode_custom(obj):
    if isinstance(obj, np.ndarray):
        output = io.BytesIO()
        np.save(output, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    return obj


def _decode_custom(obj):
    if not isinstance(obj, dict):
        return obj
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _pack(data: Any) -> bytes:
    return msgpack.packb(data, default=_encode_custom, use_bin_type=True)


def _unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode_custom, raw=False, strict_map_key=False)

# ════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════

@dataclass
class InferenceConfig:
    """Configuration for inference client."""
    server_host: str = "localhost"
    server_port: int = 5555
    timeout: int = 5
    task: str = "task1"
    camera_keys: List[str] = field(default_factory=lambda: list(DEFAULT_CAMERA_KEYS))
    verbose: bool = False

# ════════════════════════════════════════════════════════════════════════════════════
# INFERENCE CLIENT
# ════════════════════════════════════════════════════════════════════════════════════

class DakshaInferenceClient:
    """Client for running inference against serve_policy_zmq.py."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, config.timeout * 1000)
        self.socket.setsockopt(zmq.LINGER, 0)

        endpoint = f"tcp://{config.server_host}:{config.server_port}"
        logger.info(f"Connecting to server at {endpoint}")
        self.socket.connect(endpoint)

        self.is_ready = False
        self._handshake()

    def _request(self, endpoint: str, data: Optional[Dict] = None) -> Any:
        self.socket.send(_pack({"endpoint": endpoint, "data": data or {}}))
        return _unpack(self.socket.recv())

    def _handshake(self):
        """Verify connection with server."""
        try:
            response = self._request("ping")
            if isinstance(response, dict) and response.get("ok"):
                self.is_ready = True
                logger.info("✓ Server connection established")
            else:
                logger.error(f"❌ Server handshake failed: {response}")
        except zmq.error.Again:
            logger.error("❌ Server timeout during handshake")
            self.is_ready = False
        except Exception as e:
            logger.error(f"❌ Handshake error: {e}")
            self.is_ready = False

    def reset(self):
        """Reset policy + processor state on the server (call between episodes)."""
        try:
            return self._request("reset")
        except Exception as e:
            logger.error(f"Reset error: {e}")
            return None

    def infer(
        self,
        observation: Dict[str, np.ndarray],
        task: Optional[str] = None,
        return_raw: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Run inference on an observation.

        Args:
            observation: dict with per-camera images keyed by camera name
                (e.g. "zed_right": HxWx3 uint8) plus a "state" key holding a
                16-dim float32 array (right_arm, right_gripper, left_arm,
                left_gripper concatenated, per modality.json).
            task: language instruction override; defaults to config.task.
            return_raw: if True, return the full [payload, meta] response.

        Returns:
            action chunk as np.ndarray of shape (horizon, action_dim), or
            None on error.
        """
        if not self.is_ready:
            logger.error("Client not ready")
            return None

        try:
            request_data = self._prepare_inference_request(observation, task)

            if self.config.verbose:
                logger.info("Sending get_action request")

            response = self._request("get_action", request_data)

            if isinstance(response, dict) and "error" in response:
                logger.error(f"Server error: {response['error']}")
                return None

            payload, meta = response
            action = np.asarray(payload["action"], dtype=np.float32)

            if self.config.verbose:
                logger.info(f"Server response: horizon={meta.get('horizon')} task='{meta.get('task')}'")

            if return_raw:
                return response
            return action

        except zmq.error.Again:
            logger.error("Inference timeout")
            return None
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def _prepare_inference_request(self, observation: Dict, task: Optional[str]) -> Dict:
        """Build the {"observation": {...}, "options": {...}} payload."""
        video = {}
        for camera in self.config.camera_keys:
            if camera in observation:
                video[camera] = observation[camera]

        state = {}
        if "state" in observation:
            state["observation.state"] = np.asarray(observation["state"], dtype=np.float32)

        return {
            "observation": {"video": video, "state": state},
            "options": {"task": task or self.config.task},
        }

    def batch_infer(
        self,
        observations: List[Dict[str, np.ndarray]],
    ) -> Optional[List[np.ndarray]]:
        """Run inference on a batch of observations."""
        results = []
        for obs in observations:
            action = self.infer(obs)
            if action is not None:
                results.append(action)
            else:
                logger.warning("Batch inference skipped due to error")
                return None
        return results

    def close(self):
        """Close connection."""
        if self.socket:
            self.socket.close()
        self.context.term()
        logger.info("Client closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# ════════════════════════════════════════════════════════════════════════════════════
# DEMO / TESTING
# ════════════════════════════════════════════════════════════════════════════════════

def create_dummy_observation(config: InferenceConfig) -> Dict[str, np.ndarray]:
    """Create dummy observation for testing."""
    obs = {}
    for camera in config.camera_keys:
        obs[camera] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs["state"] = np.random.randn(16).astype(np.float32)
    return obs

def test_inference(args):
    """Test inference client with dummy data."""
    config = InferenceConfig(
        server_host=args.host,
        server_port=args.port,
        task=args.task,
        verbose=args.verbose,
    )

    logger.info(f"Testing inference against {config.server_host}:{config.server_port}")

    with DakshaInferenceClient(config) as client:
        if not client.is_ready:
            logger.error("Failed to connect to server")
            return

        logger.info("Running inference test...")

        for i in range(args.num_tests):
            obs = create_dummy_observation(config)
            logger.info(f"Test {i+1}/{args.num_tests}...")

            action = client.infer(obs)
            if action is not None:
                logger.info(f"✓ Action shape: {action.shape}, mean: {action.mean():.4f}")
            else:
                logger.error(f"✗ Inference failed")

            if i < args.num_tests - 1:
                time.sleep(0.1)

        logger.info("✓ Inference test completed")

# ════════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Daksha VLA Inference Client"
    )
    parser.add_argument("--host", default="localhost", help="Server hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=5555, help="Server port (default: 5555)")
    parser.add_argument("--task", default="task1", help="Language instruction to send (default: task1)")
    parser.add_argument("--test", action="store_true", help="Run test inference")
    parser.add_argument("--num-tests", type=int, default=5, help="Number of test inferences (default: 5)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.test:
        test_inference(args)
    else:
        config = InferenceConfig(
            server_host=args.host,
            server_port=args.port,
            task=args.task,
            verbose=args.verbose,
        )

        logger.info("Starting interactive inference client")
        logger.info("Type 'quit' to exit\n")

        with DakshaInferenceClient(config) as client:
            if not client.is_ready:
                logger.error("Failed to connect")
                return

            while True:
                try:
                    cmd = input(">> ").strip().lower()

                    if cmd == "quit":
                        break
                    elif cmd == "test":
                        obs = create_dummy_observation(config)
                        action = client.infer(obs)
                        if action is not None:
                            logger.info(f"Action: {action}")
                    elif cmd == "reset":
                        client.reset()
                        logger.info("Policy state reset")
                    elif cmd == "status":
                        logger.info(f"Connected to {args.host}:{args.port}")
                    elif cmd == "help":
                        print("Commands: test, reset, status, quit, help")
                    else:
                        print("Unknown command. Type 'help' for available commands.")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
