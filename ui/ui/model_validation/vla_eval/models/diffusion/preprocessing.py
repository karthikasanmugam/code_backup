"""Framework Observation -> LeRobot Diffusion Policy input batch conversion."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from vla_eval.core.types import Observation
from vla_eval.models.diffusion.compatibility import validate_observation_compatibility
from vla_eval.models.diffusion.config import DiffusionCheckpointConfig


def _image_to_chw_float32(image: np.ndarray) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape != (240, 320, 3):
        raise ValueError(f"Expected RGB HWC image (240,320,3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    # Match the training decoder before applying saved mean/std normalization.
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)


def observation_history_to_diffusion_batch(
    observations: Sequence[Observation],
    cfg: DiffusionCheckpointConfig,
) -> dict[str, Any]:
    """Build a batched temporal input with shape [B=1, T=2, ...]."""
    if len(observations) != cfg.n_obs_steps:
        raise ValueError(
            f"Expected {cfg.n_obs_steps} observations, got {len(observations)}"
        )

    for observation in observations:
        validate_observation_compatibility(observation, cfg)

    batch: dict[str, Any] = {}

    state = np.stack(
        [np.asarray(obs.state, dtype=np.float32) for obs in observations],
        axis=0,
    )
    batch["observation.state"] = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0)

    leader_states = [obs.extra.get("leader_state") for obs in observations]
    if cfg.leader_state_dim is not None:
        leader = np.stack(
            [np.asarray(value, dtype=np.float32) for value in leader_states],
            axis=0,
        )
        batch["observation.leader_state"] = torch.from_numpy(
            np.ascontiguousarray(leader)
        ).unsqueeze(0)

    for camera_key in cfg.camera_keys:
        frames = torch.stack(
            [_image_to_chw_float32(obs.images[camera_key]) for obs in observations],
            dim=0,
        )
        batch[camera_key] = frames.unsqueeze(0)

    return batch
