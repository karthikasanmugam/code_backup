"""Dataset/checkpoint compatibility checks for SmolVLA."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from vla_eval.core.types import Observation
from vla_eval.models.smolvla.config import SmolVLACheckpointConfig


def validate_observation_compatibility(
    observation: Observation,
    cfg: SmolVLACheckpointConfig,
) -> None:
    """Validate one framework Observation before sending it to SmolVLA."""
    state = np.asarray(observation.state)
    if state.shape != (cfg.state_dim,):
        raise ValueError(
            f"SmolVLA state mismatch: expected {(cfg.state_dim,)}, got {state.shape}"
        )

    leader_state = observation.extra.get("leader_state")
    if leader_state is None:
        raise ValueError(
            "SmolVLA checkpoint requires observation.leader_state, "
            "but Observation.extra does not contain 'leader_state'."
        )
    leader_state = np.asarray(leader_state)
    if leader_state.shape != (cfg.state_dim,):
        raise ValueError(
            "SmolVLA leader_state mismatch: "
            f"expected {(cfg.state_dim,)}, got {leader_state.shape}"
        )

    expected_image_shape = (240, 320, 3)
    for camera_key in cfg.camera_keys:
        if camera_key not in observation.images:
            raise KeyError(
                f"Required SmolVLA camera '{camera_key}' is missing. "
                f"Available cameras: {sorted(observation.images)}"
            )
        image = np.asarray(observation.images[camera_key])
        if image.shape != expected_image_shape:
            raise ValueError(
                f"SmolVLA camera '{camera_key}' shape mismatch: "
                f"expected HWC {expected_image_shape}, got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"SmolVLA camera '{camera_key}' must be uint8 at the framework boundary, "
                f"got {image.dtype}"
            )


def validate_observation_history(
    observations: Iterable[Observation],
    cfg: SmolVLACheckpointConfig,
) -> None:
    observations = list(observations)
    if len(observations) != cfg.n_obs_steps:
        raise ValueError(
            f"SmolVLA expects {cfg.n_obs_steps} observation(s), got {len(observations)}"
        )
    for obs in observations:
        validate_observation_compatibility(obs, cfg)
