"""Conversion from framework Observation objects to LeRobot ACT batches."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vla_eval.core.types import Observation


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB HWC image, got shape={image.shape}"
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image from dataset, got {image.dtype}"
        )

    image = np.ascontiguousarray(image)

    # Match LeRobot's training video decoder: uint8 -> float32 in [0, 1].
    # The checkpoint normalizer applies mean/std; it does not scale pixels.
    return (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div_(255.0)
    )

def observation_to_act_batch(
    observation: Observation,
    camera_keys: list[str],
) -> dict[str, Any]:
    """Build a one-step ACT input batch."""

    batch: dict[str, Any] = {}

    state = np.asarray(
        observation.state,
        dtype=np.float32,
    )
    if state.ndim != 1:
        raise ValueError(
            f"Observation state must be 1D; got {state.shape}"
        )
    batch["observation.state"] = torch.from_numpy(
        np.ascontiguousarray(state)
    ).unsqueeze(0)

    leader_state = observation.extra.get(
        "leader_state"
    )
    if leader_state is None:
        raise ValueError(
            "ACT checkpoint requires 'observation.leader_state', "
            "but the Observation.extra field does not contain 'leader_state'."
        )

    leader_state = np.asarray(
        leader_state,
        dtype=np.float32,
    )
    if leader_state.ndim != 1:
        raise ValueError(
            "leader_state must be a 1D vector; "
            f"got {leader_state.shape}"
        )
    batch["observation.leader_state"] = torch.from_numpy(
        np.ascontiguousarray(leader_state)
    ).unsqueeze(0)

    for camera_key in camera_keys:
        if camera_key not in observation.images:
            raise KeyError(
                f"Required ACT camera '{camera_key}' is missing from observation. "
                f"Available cameras: {sorted(observation.images)}"
            )
        batch[camera_key] = _image_to_tensor(
            observation.images[camera_key]
        )

    return batch
