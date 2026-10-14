"""ACT adapter for the model-agnostic VLA evaluation framework."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from vla_eval.core.base_vla import BaseVLA
from vla_eval.core.registry import register_model
from vla_eval.core.types import ActionPrediction, ModelMetadata, Observation
from vla_eval.models.act.config import ACTCheckpointConfig
from vla_eval.models.act.preprocessing import observation_to_act_batch


@register_model("act")
class ACTAdapter(BaseVLA):
    """Thin wrapper around the LeRobot ACT policy."""

    def __init__(
        self,
        device: str | None = None,
        checkpoint_config: Optional[dict] = None,
    ) -> None:
        self.device_override = device
        self.checkpoint_config_override = checkpoint_config
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.cfg: ACTCheckpointConfig | None = None
        self._metadata: ModelMetadata | None = None
        self._loaded = False

    def load(
        self,
        checkpoint_path: str,
        config: Optional[dict] = None,
    ) -> None:
        try:
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies.act import ACTPolicy
            from lerobot.policies import make_pre_post_processors
        except ImportError as exc:
            raise ImportError(
                "ACTAdapter requires the installed LeRobot package with ACT support."
            ) from exc

        checkpoint = Path(
            checkpoint_path
        ).expanduser().resolve()

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"ACT checkpoint not found: {checkpoint}"
            )

        # Read and validate the config supplied with the checkpoint.
        if config is not None:
            self.cfg = ACTCheckpointConfig.from_dict(config)
        else:
            self.cfg = ACTCheckpointConfig.from_checkpoint(
                checkpoint
            )

        self.cfg.validate()

        # Let the installed LeRobot version instantiate the ACT configuration.
        policy_cfg = PreTrainedConfig.from_pretrained(
            str(checkpoint)
        )

        if getattr(policy_cfg, "type", None) != "act":
            raise ValueError(
                f"LeRobot checkpoint config is not ACT: {getattr(policy_cfg, 'type', None)!r}"
            )

        if self.device_override is not None:
            device_name = self.device_override
        else:
            device_name = self.cfg.device

        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "ACT checkpoint requests CUDA, but torch.cuda.is_available() is False. "
                "Pass device='cpu' explicitly for a CPU smoke test."
            )

        device = torch.device(device_name)
        policy_cfg.device = device

        # Load the actual trained ACT weights.
        self.policy = ACTPolicy.from_pretrained(
            str(checkpoint),
            config=policy_cfg,
        )
        self.policy.to(device)
        self.policy.eval()

        # Load the same processor pipeline saved with the checkpoint. This is
        # important because normalization/resize behavior is part of inference.
        self.preprocessor, self.postprocessor = (
            make_pre_post_processors(
                policy_cfg,
                pretrained_path=str(checkpoint),
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
        )

        self._metadata = ModelMetadata(
            model_type="act",
            obs_horizon=self.cfg.n_obs_steps,
            chunk_size=self.cfg.chunk_size,
            exec_horizon=self.cfg.n_action_steps,
            action_space="native",
            action_dim=self.cfg.action_dim,
            stochastic=False,
            num_samples=1,
            checkpoint_path=str(checkpoint),
            extra={
                "camera_keys": self.cfg.camera_keys,
                "state_keys": self.cfg.state_keys,
                "vision_backbone": self.cfg.vision_backbone,
                "dim_model": self.cfg.dim_model,
                "n_heads": self.cfg.n_heads,
                "n_encoder_layers": self.cfg.n_encoder_layers,
                "n_decoder_layers": self.cfg.n_decoder_layers,
                "use_vae": self.cfg.use_vae,
            },
        )

        self._loaded = True

    def reset(self) -> None:
        if not self._loaded:
            return

        if self.policy is not None and hasattr(
            self.policy,
            "reset",
        ):
            self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            if processor is not None and hasattr(processor, "reset"):
                processor.reset()

    def predict(
        self,
        observations: List[Observation],
    ) -> ActionPrediction:
        if not self._loaded:
            raise RuntimeError(
                "ACTAdapter.load() must be called before predict()."
            )

        assert self.cfg is not None
        assert self._metadata is not None
        assert self.policy is not None
        assert self.preprocessor is not None
        assert self.postprocessor is not None

        if len(observations) != self.cfg.n_obs_steps:
            raise ValueError(
                "ACTAdapter received an incorrect observation history length: "
                f"expected {self.cfg.n_obs_steps}, got {len(observations)}"
            )

        observation = observations[-1]

        batch = observation_to_act_batch(
            observation,
            self.cfg.camera_keys,
        )

        # Add device placement before processor/model execution.
        device = next(
            self.policy.parameters()
        ).device

        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        # The saved LeRobot processor is responsible for normalization and
        # other policy-specific input transformations.
        batch = self.preprocessor(batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        with torch.inference_mode():
            # A queued select_action fallback would reuse old predictions and
            # no longer compare a fresh prediction for this observation.
            actions = self.policy.predict_action_chunk(batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = (
            time.perf_counter()
            - start
        )

        # ACT returns [B, horizon, action_dim] for a chunk and
        # [B, action_dim] for select_action in older versions.
        if isinstance(actions, dict):
            if "action" not in actions:
                raise RuntimeError(
                    "ACT policy returned a dictionary without the 'action' key."
                )
            actions = actions["action"]

        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)

        actions = actions.detach()

        # The policy is expected to return normalized actions; postprocessor
        # restores the training-time action space.
        actions = self.postprocessor(actions)

        if isinstance(actions, dict):
            if "action" not in actions:
                raise RuntimeError(
                    "ACT postprocessor returned a dictionary without the 'action' key."
                )
            actions = actions["action"]

        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)

        actions = actions.detach().float().cpu().numpy()

        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise RuntimeError(
                    f"ACT adapter currently expects batch size 1, got {actions.shape}"
                )
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions[None, :]
        else:
            raise RuntimeError(
                f"Unexpected ACT action shape: {actions.shape}"
            )

        if actions.shape != (self.cfg.chunk_size, self.cfg.action_dim):
            raise RuntimeError(
                "ACT action chunk mismatch: "
                f"expected {(self.cfg.chunk_size, self.cfg.action_dim)}, got {actions.shape}"
            )

        if not np.isfinite(actions).all():
            raise RuntimeError(
                "ACT returned NaN or infinite action values."
            )

        return ActionPrediction(
            actions=actions,
            action_horizon=actions.shape[0],
            inference_latency_s=latency,
            metadata={
                "model_type": "act",
                "checkpoint": self._metadata.checkpoint_path,
                "configured_chunk_size": self.cfg.chunk_size,
                "configured_n_action_steps": self.cfg.n_action_steps,
            },
        )

    def get_metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise RuntimeError(
                "ACTAdapter is not loaded."
            )
        return self._metadata
