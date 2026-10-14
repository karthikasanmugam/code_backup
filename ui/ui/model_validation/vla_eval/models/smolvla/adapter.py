"""SmolVLA adapter for the model-agnostic VLA evaluation framework."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from vla_eval.core.base_vla import BaseVLA
from vla_eval.core.registry import register_model
from vla_eval.core.types import ActionPrediction, ModelMetadata, Observation
from vla_eval.models.runtime import InferenceRuntime
from vla_eval.models.smolvla.compatibility import validate_observation_history
from vla_eval.models.smolvla.config import SmolVLACheckpointConfig
from vla_eval.models.smolvla.preprocessing import observation_to_smolvla_batch


@register_model("smolvla")
class SmolVLAAdapter(BaseVLA):
    """Thin wrapper around the official LeRobot SmolVLAPolicy API."""

    def __init__(
        self,
        device: str | None = None,
        checkpoint_config: dict | None = None,
        inference_profile: str = "optimized",
    ) -> None:
        self.device_override = device
        self.inference_profile = inference_profile
        self.runtime = None
        self.checkpoint_config_override = checkpoint_config
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.cfg: SmolVLACheckpointConfig | None = None
        self._metadata: ModelMetadata | None = None
        self._loaded = False

    def load(self, checkpoint_path: str, config: dict | None = None) -> None:
        try:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as exc:
            raise ImportError(
                "SmolVLA requires a LeRobot installation with SmolVLA support. "
                "Install the SmolVLA extra, e.g. pip install 'lerobot[smolvla]'."
            ) from exc

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.exists() or not checkpoint.is_dir():
            raise FileNotFoundError(f"SmolVLA checkpoint directory not found: {checkpoint}")

        if config is not None:
            self.cfg = SmolVLACheckpointConfig.from_dict(config)
        elif self.checkpoint_config_override is not None:
            self.cfg = SmolVLACheckpointConfig.from_dict(self.checkpoint_config_override)
        else:
            self.cfg = SmolVLACheckpointConfig.from_checkpoint(checkpoint)

        self.cfg.validate()

        policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        if getattr(policy_cfg, "type", None) != "smolvla":
            raise ValueError(
                f"LeRobot checkpoint config is not SmolVLA: {getattr(policy_cfg, 'type', None)!r}"
            )

        device_name = self.device_override or self.cfg.device
        device = torch.device(device_name)
        self.runtime = InferenceRuntime.resolve(self.inference_profile, device)
        policy_cfg.device = device
        policy_cfg.compile_model = False
        if self.inference_profile == "fast":
            policy_cfg.num_steps = min(5, policy_cfg.num_steps)

        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint), config=policy_cfg)
        self.policy.to(device)
        self.policy.eval()

        # Use the checkpoint's saved processor/statistics pipeline. This is
        # the source of truth for state/action normalization and tokenization.
        processor_overrides = {"device_processor": {"device": str(device)}}
        if (checkpoint / "tokenizer").is_dir():
            processor_overrides["tokenizer_processor"] = {"tokenizer_name": str(checkpoint / "tokenizer")}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides=processor_overrides,
        )

        self._metadata = ModelMetadata(
            model_type="smolvla",
            obs_horizon=self.cfg.n_obs_steps,
            chunk_size=self.cfg.chunk_size,
            exec_horizon=self.cfg.n_action_steps,
            action_space="native",
            action_dim=self.cfg.action_dim,
            stochastic=True,
            num_samples=1,
            checkpoint_path=str(checkpoint),
            extra={
                "runtime": {
                    **self.runtime.metadata(),
                    "num_steps": policy_cfg.num_steps,
                    "checkpoint_num_steps": self.cfg.num_steps,
                    "use_cache": policy_cfg.use_cache,
                },
                "camera_keys": self.cfg.camera_keys,
                "state_keys": self.cfg.state_keys,
                "max_state_dim": self.cfg.max_state_dim,
                "max_action_dim": self.cfg.max_action_dim,
                "resize_imgs_with_padding": self.cfg.resize_imgs_with_padding,
                "tokenizer_max_length": self.cfg.tokenizer_max_length,
                "num_steps": policy_cfg.num_steps,
                "use_cache": self.cfg.use_cache,
                "vlm_model_name": self.cfg.vlm_model_name,
                "attention_mode": self.cfg.attention_mode,
                "num_vlm_layers": self.cfg.num_vlm_layers,
                "num_expert_layers": self.cfg.num_expert_layers,
                "compile_model": policy_cfg.compile_model,
            },
        )
        self._loaded = True

    def reset(self) -> None:
        if not self._loaded or self.policy is None:
            return
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            if processor is not None and hasattr(processor, "reset"):
                processor.reset()

    def predict(self, observations: list[Observation]) -> ActionPrediction:
        if not self._loaded:
            raise RuntimeError("SmolVLAAdapter.load() must be called before predict().")
        assert self.cfg is not None
        assert self._metadata is not None
        assert self.policy is not None
        assert self.preprocessor is not None
        assert self.postprocessor is not None
        assert self.runtime is not None

        validate_observation_history(observations, self.cfg)
        observation = observations[-1]
        batch = observation_to_smolvla_batch(observation, self.cfg.camera_keys)

        device = next(self.policy.parameters()).device
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        # The preprocessor adds tokenization, normalizes state/action according
        # to checkpoint statistics, and moves inputs to the model device.
        batch = self.preprocessor(batch)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode(), self.runtime.autocast():
            actions = self.policy.predict_action_chunk(batch)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = time.perf_counter() - start

        if isinstance(actions, dict):
            if "action" not in actions:
                raise RuntimeError("SmolVLA returned a dict without the 'action' key.")
            actions = actions["action"]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)

        # This converts normalized actions back into the checkpoint's native
        # action space and moves them to CPU numpy for the common interface.
        actions = self.postprocessor(actions.float())
        if isinstance(actions, dict):
            if "action" not in actions:
                raise RuntimeError("SmolVLA postprocessor returned no 'action' key.")
            actions = actions["action"]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)

        actions = actions.detach().float().cpu().numpy()

        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise RuntimeError(f"SmolVLA adapter expects batch size 1, got {actions.shape}")
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions[None, :]
        else:
            raise RuntimeError(f"Unexpected SmolVLA action shape: {actions.shape}")

        if actions.shape[-1] != self.cfg.action_dim:
            raise RuntimeError(
                f"SmolVLA action dimension mismatch: expected {self.cfg.action_dim}, got {actions.shape[-1]}"
            )
        if actions.shape[0] < 1:
            raise RuntimeError("SmolVLA returned an empty action chunk")
        if not np.isfinite(actions).all():
            raise RuntimeError("SmolVLA returned NaN or infinite action values")

        return ActionPrediction(
            actions=actions,
            action_horizon=actions.shape[0],
            inference_latency_s=latency,
            metadata={
                "model_type": "smolvla",
                "checkpoint": self._metadata.checkpoint_path,
                "configured_chunk_size": self.cfg.chunk_size,
                "configured_n_action_steps": self.cfg.n_action_steps,
                "num_steps": self.policy.config.num_steps,
                "vlm_model_name": self.cfg.vlm_model_name,
            },
        )

    def get_metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise RuntimeError("SmolVLAAdapter is not loaded.")
        return self._metadata
