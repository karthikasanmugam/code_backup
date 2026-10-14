from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


# ============================================================
# Common types
# ============================================================

Severity = Literal["HARD", "SOFT"]
Classification = Literal["REJECT", "REVIEW", "KEEP"]


# ============================================================
# Validation result
# ============================================================

@dataclass(slots=True)
class CheckResult:
    """
    Result returned by every validation function.

    A checker should return:
        list[CheckResult]

    frame_range:
        Optional inclusive/exclusive frame interval associated
        with the problem, e.g. (100, 160).
    """

    check: str
    severity: Severity
    message: str
    frame_range: tuple[int, int] | None = None

    @property
    def passed(self) -> bool:
        """True when this result represents a passing check."""
        return self.severity not in ("HARD", "SOFT")


# ============================================================
# Dataset metadata
# ============================================================

@dataclass(slots=True)
class FeatureInfo:
    """
    Description of a single feature from info.json.
    """

    name: str
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CameraInfo:
    """
    Description of one camera/video feature.
    """

    key: str
    shape: tuple[int, ...] = ()
    dtype: str | None = None

    @property
    def height(self) -> int | None:
        if len(self.shape) >= 2:
            return self.shape[-2]
        return None

    @property
    def width(self) -> int | None:
        if len(self.shape) >= 2:
            return self.shape[-1]
        return None


@dataclass(slots=True)
class DatasetInfo:
    """
    Normalized dataset information loaded from info.json.

    This is the main source of truth for dataset-defined values:
        - FPS
        - state dimensions
        - action dimensions
        - camera keys
        - feature shapes
    """

    fps: float
    state_dim: int
    action_dim: int

    features: dict[str, FeatureInfo] = field(default_factory=dict)
    cameras: dict[str, CameraInfo] = field(default_factory=dict)

    # Raw info.json content is retained so future checks can access
    # dataset-specific metadata without changing this model immediately.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def camera_keys(self) -> list[str]:
        return list(self.cameras.keys())


# ============================================================
# Episode paths and metadata
# ============================================================

@dataclass(slots=True)
class EpisodePaths:
    """
    Files associated with one episode.
    """

    episode_index: int
    parquet: Path
    videos: dict[str, Path] = field(default_factory=dict)
    video_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def video_paths(self) -> dict[str, Path]:
        return self.videos


@dataclass(slots=True)
class EpisodeMetadata:
    """
    Metadata associated with one episode.

    The exact metadata keys can vary between dataset versions,
    so the original metadata dictionary is retained.
    """

    episode_index: int
    length: int | None = None
    duration: float | None = None
    success: bool | None = None
    task_index: int | None = None

    raw: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Dataset statistics
# ============================================================

@dataclass(slots=True)
class DatasetStats:
    """
    Statistics computed during Phase 0 and reused by later checks.
    """

    # Per-dimension percentile bounds.
    action_percentiles: dict[str, np.ndarray] = field(default_factory=dict)

    # Population-level episode statistics.
    median_episode_length: float | None = None
    median_duration: float | None = None
    median_path_length: float | None = None

    # Optional task statistics.
    task_counts: dict[int, int] = field(default_factory=dict)

    # Additional statistics can be added without changing the checker API.
    extra: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Episode data
# ============================================================

@dataclass(slots=True)
class EpisodeData:
    """
    Loaded data for one episode.

    Arrays remain optional because early structural checks should
    not force expensive loading of everything.
    """

    frame_index: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    states: np.ndarray | None = None
    actions: np.ndarray | None = None

    # Parsed task indices for the episode.
    task_indices: np.ndarray | None = None

    # Optional decoded/sampled video frames:
    # camera key -> ndarray
    video_frames: dict[str, np.ndarray] = field(default_factory=dict)

    # Video timestamps / PTS:
    # camera key -> ndarray
    video_timestamps: dict[str, np.ndarray] = field(default_factory=dict)


# ============================================================
# Validation configuration
# ============================================================

@dataclass(slots=True)
class ValidationConfig:
    """
    Fully resolved validator configuration.

    The YAML loader will populate this from configs/default.yml.

    Keeping the config as a dictionary would make all checkers
    dependent on string keys. This object gives us a stable Python
    contract instead.
    """

    # Raw configuration is kept for flexibility while the project grows.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def short_circuit_on_hard(self) -> bool:
        return self.raw.get(
            "execution",
            {},
        ).get(
            "short_circuit_on_hard",
            True,
        )

    @property
    def skip_expensive_after_hard(self) -> bool:
        return self.raw.get(
            "execution",
            {},
        ).get(
            "skip_expensive_after_hard",
            True,
        )


# ============================================================
# Episode context
# ============================================================

@dataclass(slots=True)
class EpisodeContext:
    """
    Shared context passed to episode-level checks.

    This prevents individual checker functions from having large
    argument lists.

    Example:
        check_frozen_video(context)
        check_action_range(context)
        check_idle_ratio(context)
    """

    episode_index: int
    dataset_info: DatasetInfo
    paths: EpisodePaths

    metadata: EpisodeMetadata | None = None
    data: EpisodeData = field(default_factory=EpisodeData)

    stats: DatasetStats | None = None
    config: ValidationConfig | None = None


# ============================================================
# Validation summary
# ============================================================

@dataclass(slots=True)
class EpisodeValidation:
    """
    Complete validation result for one episode.
    """

    episode_index: int
    results: list[CheckResult] = field(default_factory=list)

    score: float | None = None
    classification: Classification | None = None

    @property
    def hard_failures(self) -> list[CheckResult]:
        return [
            result
            for result in self.results
            if result.severity == "HARD"
        ]

    @property
    def soft_failures(self) -> list[CheckResult]:
        return [
            result
            for result in self.results
            if result.severity == "SOFT"
        ]

    @property
    def has_hard_failure(self) -> bool:
        return bool(self.hard_failures)


@dataclass(slots=True)
class DatasetValidation:
    """
    Complete validation result for a dataset.
    """

    total_episodes: int
    episodes: list[EpisodeValidation] = field(default_factory=list)

    dataset_results: list[CheckResult] = field(default_factory=list)

    @property
    def flagged_episode_numbers(self) -> list[int]:
        """
        Unique sorted episode IDs that contain at least one failure.
        """
        return sorted(
            {
                episode.episode_index
                for episode in self.episodes
                if episode.results
            }
        )

    @property
    def kept_episode_numbers(self) -> list[int]:
        return sorted(
            {
                episode.episode_index
                for episode in self.episodes
                if episode.classification == "KEEP"
            }
        )

    @property
    def rejected_episode_numbers(self) -> list[int]:
        return sorted(
            {
                episode.episode_index
                for episode in self.episodes
                if episode.classification == "REJECT"
            }
        )

    @property
    def review_episode_numbers(self) -> list[int]:
        return sorted(
            {
                episode.episode_index
                for episode in self.episodes
                if episode.classification == "REVIEW"
            }
        )