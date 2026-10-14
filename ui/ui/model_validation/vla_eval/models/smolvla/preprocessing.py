"""Framework Observation -> LeRobot SmolVLA inference batch conversion."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vla_eval.core.types import Observation


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC image, got shape={image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image at framework boundary, got {image.dtype}")

    # SmolVLA's configured VISUAL normalization is IDENTITY. The policy's
    # prepare_images() expects pixels in [0, 1] before converting them to
    # [-1, 1] for SigLIP, so unlike ACT we explicitly scale uint8 here.
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    return tensor.to(dtype=torch.float32).div_(255.0).unsqueeze(0)


def observation_to_smolvla_batch(
    observation: Observation,
    camera_keys: list[str],
) -> dict[str, Any]:
    """Build the single-step raw batch consumed by LeRobot's SmolVLA preprocessor."""
    batch: dict[str, Any] = {}

    state = np.asarray(observation.state, dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(f"Observation state must be 1D; got {state.shape}")
    batch["observation.state"] = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0)

    leader_state = observation.extra.get("leader_state")
    if leader_state is None:
        raise ValueError(
            "SmolVLA requires 'observation.leader_state' but Observation.extra has no 'leader_state'."
        )
    leader_state = np.asarray(leader_state, dtype=np.float32)
    if leader_state.ndim != 1:
        raise ValueError(f"leader_state must be 1D; got {leader_state.shape}")
    batch["observation.leader_state"] = torch.from_numpy(
        np.ascontiguousarray(leader_state)
    ).unsqueeze(0)

    for camera_key in camera_keys:
        if camera_key not in observation.images:
            raise KeyError(
                f"Required SmolVLA camera '{camera_key}' is missing from observation."
            )
        batch[camera_key] = _image_to_tensor(observation.images[camera_key])

    # SmolVLA is a VLA: its model path requires tokenized language inputs.
    # Observation.instruction is the framework's canonical task description.
    batch["task"] = observation.instruction or ""
    return batch
