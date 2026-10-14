"""
Core data contract for the VLA evaluation framework.

Every dataset adapter must produce `Sample` objects built from these types.
Every model adapter must consume `Observation` and return an `ActionPrediction`.
Nothing downstream (metrics, reporting) should ever import a dataset- or
model-specific type — only these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Observation:
    """What a VLA model sees at a single timestep (or stacked history)."""

    # camera_key -> image array, HWC, uint8, RGB
    images: Dict[str, np.ndarray]

    # proprioceptive state the policy conditions on (robot joint/EE state)
    state: np.ndarray

    # natural-language task instruction
    instruction: str

    timestamp: float
    frame_index: int
    episode_index: int

    # anything dataset-specific that isn't part of the model input contract
    # (e.g. leader_state during teleop, raw metadata). Adapters may read
    # this but the core framework never depends on its contents.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruth:
    """The recorded action / label for a timestep, used for offline eval."""

    action: np.ndarray
    task: str
    episode_index: int
    frame_index: int
    index: int  # global frame index across the whole dataset


@dataclass
class Sample:
    """One (Observation, GroundTruth) pair — the unit the dataset yields."""

    observation: Observation
    ground_truth: GroundTruth


@dataclass
class EpisodeInfo:
    """Episode-level metadata, independent of any single frame."""

    episode_index: int
    length: int  # number of frames in this episode
    task: str
    chunk_index: int = 0


@dataclass
class ActionPrediction:
    """What a model adapter returns from predict()."""

    actions: np.ndarray  # shape (horizon, action_dim) or (action_dim,)
    action_horizon: int
    timestamps: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    inference_latency_s: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalAction:
    """Model-agnostic action representation. Filled in Phase 5."""

    values: np.ndarray
    action_space: str  # e.g. "joint_position", "ee_pose_quat"
    names: Optional[list] = None


@dataclass
class ModelMetadata:
    """Static facts about a loaded model that the runner needs *before* it
    can call predict() correctly — how much observation history to feed it,
    how many future actions it predicts, whether it's stochastic, etc.

    Every BaseVLA subclass must be able to answer this truthfully after
    load() — the runner builds its observation windows and chunk-execution
    logic entirely from these values rather than hardcoding per-model
    behavior.
    """

    model_type: str  # e.g. "act", "diffusion"
    obs_horizon: int  # how many past timesteps predict() expects, oldest-first
    chunk_size: int  # how many future actions a single predict() call returns
    exec_horizon: int  # how many of those actions to actually execute before replanning (closed-loop); for offline eval, how many to score against ground truth
    action_space: str  # e.g. "joint_position", "ee_pose_quat"
    action_dim: int
    stochastic: bool = False
    num_samples: int = 1  # for stochastic models: how many samples predict() draws by default
    checkpoint_path: Optional[str] = None
    # per-dimension normalization stats used at training time, so the
    # adapter (or a shared normalization layer later) can un-normalize
    # outputs / normalize inputs consistently. Shape/keys are adapter-defined
    # (e.g. {"mean": [...], "std": [...]} or {"min": [...], "max": [...]}).
    normalization_stats: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)