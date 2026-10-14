"""Model-agnostic adapter for the supplied LeRobot Diffusion checkpoint."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from vla_eval.core.base_vla import BaseVLA
from vla_eval.core.registry import register_model
from vla_eval.core.types import ActionPrediction, ModelMetadata, Observation
from vla_eval.models.diffusion.config import DiffusionCheckpointConfig
from vla_eval.models.diffusion.preprocessing import observation_history_to_diffusion_batch
from vla_eval.models.runtime import InferenceRuntime


@register_model("diffusion")
class DiffusionAdapter(BaseVLA):
    """Adapter for a LeRobot Diffusion Policy checkpoint."""

    def __init__(self, device: str | None = None, inference_profile: str = "optimized") -> None:
        self.device_override = device
        self.inference_profile = inference_profile
        self.runtime = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.cfg: DiffusionCheckpointConfig | None = None
        self._metadata: ModelMetadata | None = None
        self._loaded = False

    def load(self, checkpoint_path: str, config: dict | None = None) -> None:
        try:
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.diffusion import DiffusionPolicy
        except ImportError as exc:
            raise ImportError(
                "DiffusionAdapter requires the installed LeRobot package with Diffusion support."
            ) from exc

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        self.cfg = DiffusionCheckpointConfig.from_checkpoint(
            checkpoint,
            device_override=self.device_override,
        )
        self.cfg.validate()

        # Optional external config is only accepted as an override for documented
        # runtime values. Checkpoint architecture/interface remains authoritative.
        if config:
            requested_device = config.get("device")
            if requested_device and self.device_override is None:
                self.cfg = DiffusionCheckpointConfig.from_checkpoint(
                    checkpoint,
                    device_override=str(requested_device),
                )
                self.cfg.validate()

        policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        if getattr(policy_cfg, "type", None) != "diffusion":
            raise ValueError(
                f"LeRobot checkpoint config is not Diffusion: {getattr(policy_cfg, 'type', None)!r}"
            )

        device = torch.device(self.cfg.device)
        self.runtime = InferenceRuntime.resolve(self.inference_profile, device)
        policy_cfg.device = device
        # Compilation can cost minutes on Jetson and is not needed for these profiles.
        policy_cfg.compile_model = False
        if self.inference_profile == "fast":
            # DDIM supports fewer sampling steps using the same trained noise model.
            policy_cfg.noise_scheduler_type = "DDIM"
            policy_cfg.num_inference_steps = min(
                20, self.cfg.num_inference_steps or self.cfg.num_train_timesteps
            )

        self.policy = DiffusionPolicy.from_pretrained(
            str(checkpoint),
            config=policy_cfg,
        )
        self.policy.to(device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        self._metadata = ModelMetadata(
            model_type="diffusion",
            obs_horizon=self.cfg.n_obs_steps,
            chunk_size=self.cfg.n_action_steps,
            exec_horizon=self.cfg.n_action_steps,
            action_space="native",
            action_dim=self.cfg.action_dim,
            stochastic=True,
            num_samples=1,
            checkpoint_path=str(checkpoint),
            extra={
                "runtime": {
                    **self.runtime.metadata(),
                    "sampling": policy_cfg.noise_scheduler_type,
                    "num_inference_steps": self.policy.diffusion.num_inference_steps,
                    "checkpoint_num_inference_steps": self.cfg.num_inference_steps,
                },
                "horizon": self.cfg.horizon,
                "n_action_steps": self.cfg.n_action_steps,
                "camera_keys": list(self.cfg.camera_keys),
                "state_dim": self.cfg.state_dim,
                "leader_state_dim": self.cfg.leader_state_dim,
                "vision_backbone": self.cfg.vision_backbone,
                "normalization_mapping": dict(self.cfg.normalization_mapping),
                "noise_scheduler_type": policy_cfg.noise_scheduler_type,
                "num_train_timesteps": self.cfg.num_train_timesteps,
                "num_inference_steps": self.policy.diffusion.num_inference_steps,
                "drop_n_last_frames": self.cfg.raw.get("drop_n_last_frames"),
                "num_inference_steps_is_checkpoint_default": self.cfg.num_inference_steps is None,
            },
        )
        self._loaded = True

    def reset(self) -> None:
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.preprocessor is not None and hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if self.postprocessor is not None and hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    def predict(self, observations: list[Observation]) -> ActionPrediction:
        if not self._loaded:
            raise RuntimeError("DiffusionAdapter.load() must be called before predict().")
        assert self.cfg is not None
        assert self._metadata is not None
        assert self.policy is not None
        assert self.preprocessor is not None
        assert self.postprocessor is not None
        assert self.runtime is not None

        batch = observation_history_to_diffusion_batch(observations, self.cfg)
        device = next(self.policy.parameters()).device
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        # The processor saved beside the checkpoint owns resize/crop and the
        # training normalization statistics. No manual MIN_MAX/MEAN_STD math here.
        batch = self.preprocessor(batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        with torch.inference_mode(), self.runtime.autocast():
            actions = self.policy.predict_action_chunk(batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = time.perf_counter() - start

        if isinstance(actions, dict):
            if "action" not in actions:
                raise RuntimeError("Diffusion policy returned a dict without 'action'.")
            actions = actions["action"]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)

        actions = self.postprocessor(actions.float())
        if isinstance(actions, dict):
            if "action" not in actions:
                raise RuntimeError("Diffusion postprocessor returned no 'action'.")
            actions = actions["action"]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)

        actions = actions.detach().float().cpu().numpy()
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise RuntimeError(f"Expected batch size 1, got action shape {actions.shape}")
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions[None, :]
        elif actions.ndim != 2:
            raise RuntimeError(f"Unexpected Diffusion action shape: {actions.shape}")

        expected = (self.cfg.n_action_steps, self.cfg.action_dim)
        if actions.shape != expected:
            raise RuntimeError(
                f"Diffusion action shape mismatch: expected {expected}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("Diffusion Policy returned NaN or infinite actions.")

        return ActionPrediction(
            actions=actions,
            action_horizon=actions.shape[0],
            inference_latency_s=latency,
            metadata={
                "model_type": "diffusion",
                "checkpoint": self._metadata.checkpoint_path,
                "configured_horizon": self.cfg.horizon,
                "configured_n_action_steps": self.cfg.n_action_steps,
                "num_inference_steps": self.policy.diffusion.num_inference_steps,
                "sampling": self.policy.config.noise_scheduler_type,
            },
        )

    def get_metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise RuntimeError("DiffusionAdapter is not loaded.")
        return self._metadata
