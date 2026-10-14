"""Dataset/checkpoint compatibility checks for the Diffusion adapter."""
from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.core.types import Observation
from vla_eval.models.diffusion.config import DiffusionCheckpointConfig


def validate_observation_compatibility(
    observation: Observation,
    cfg: DiffusionCheckpointConfig,
) -> None:
    state = np.asarray(observation.state)
    if state.shape != (cfg.state_dim,):
        raise ValueError(
            f"observation.state shape mismatch: expected {(cfg.state_dim,)}, got {state.shape}"
        )

    leader = observation.extra.get("leader_state")
    if cfg.leader_state_dim is not None:
        if leader is None:
            raise ValueError("Checkpoint requires observation.leader_state")
        leader = np.asarray(leader)
        if leader.shape != (cfg.leader_state_dim,):
            raise ValueError(
                "observation.leader_state shape mismatch: "
                f"expected {(cfg.leader_state_dim,)}, got {leader.shape}"
            )

    for key in cfg.camera_keys:
        if key not in observation.images:
            raise KeyError(
                f"Required Diffusion camera '{key}' is missing. "
                f"Available={sorted(observation.images)}"
            )
        image = np.asarray(observation.images[key])
        if image.shape != (240, 320, 3):
            raise ValueError(
                f"Camera '{key}' shape mismatch: expected {(240, 320, 3)}, got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"Camera '{key}' must be uint8 in the dataset adapter, got {image.dtype}"
            )


def validate_dataset_compatibility(dataset: Any, cfg: DiffusionCheckpointConfig) -> None:
    actual_cameras = set(dataset.camera_keys)
    missing = set(cfg.camera_keys) - actual_cameras
    if missing:
        raise ValueError(f"Dataset is missing Diffusion cameras: {sorted(missing)}")

    for episode in dataset.episodes:
        frames = dataset.get_episode_frames(episode.episode_index)
        if len(frames) < cfg.n_obs_steps:
            raise ValueError(
                f"Episode {episode.episode_index} has {len(frames)} frames; "
                f"Diffusion needs at least {cfg.n_obs_steps}."
            )
