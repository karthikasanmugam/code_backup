from __future__ import annotations

"""Transport-friendly schemas for the validator application/API layer.

These models define the contract between the validation engine, application
service, and UI. They intentionally contain no validation algorithms.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy a mapping into a plain dictionary."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping, got {type(value).__name__}")
    return dict(value)


def _copy_list(value: list[str] | tuple[str, ...] | None) -> list[str]:
    """Copy a sequence of strings into a plain list."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected a list or tuple, got {type(value).__name__}")
    return [str(item) for item in value]


# ---------------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraSchema:
    """Camera metadata surfaced to the UI."""

    key: str
    width: int | None = None
    height: int | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("Camera key must not be empty")
        for name, value in (("width", self.width), ("height", self.height)):
            if value is not None and value <= 0:
                raise ValueError(f"Camera {name} must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "width": self.width,
            "height": self.height,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class DatasetInspection:
    """Metadata discovered from one runtime-selected dataset."""

    dataset_root: str
    fps: float
    total_episodes: int
    total_frames: int
    total_tasks: int
    state_dim: int
    action_dim: int
    state_names: tuple[str, ...] = ()
    action_names: tuple[str, ...] = ()
    cameras: tuple[CameraSchema, ...] = ()
    tasks: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_root.strip():
            raise ValueError("dataset_root must not be empty")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        for name, value in (
            ("total_episodes", self.total_episodes),
            ("total_frames", self.total_frames),
            ("total_tasks", self.total_tasks),
            ("state_dim", self.state_dim),
            ("action_dim", self.action_dim),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetInspection":
        cameras = tuple(
            CameraSchema(
                key=str(item["key"]),
                width=item.get("width"),
                height=item.get("height"),
                enabled=bool(item.get("enabled", True)),
            )
            for item in payload.get("cameras", ())
        )
        return cls(
            dataset_root=str(payload["dataset_root"]),
            fps=float(payload["fps"]),
            total_episodes=int(payload.get("total_episodes", 0)),
            total_frames=int(payload.get("total_frames", 0)),
            total_tasks=int(payload.get("total_tasks", 0)),
            state_dim=int(payload.get("state_dim", 0)),
            action_dim=int(payload.get("action_dim", 0)),
            state_names=tuple(str(x) for x in payload.get("state_names", ())),
            action_names=tuple(str(x) for x in payload.get("action_names", ())),
            cameras=cameras,
            tasks=tuple(str(x) for x in payload.get("tasks", ())),
            extra=_copy_mapping(payload.get("extra")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "fps": self.fps,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": self.total_tasks,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "cameras": [camera.to_dict() for camera in self.cameras],
            "tasks": list(self.tasks),
            "extra": dict(self.extra),
        }


# ---------------------------------------------------------------------------
# Validation configuration contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationConfigRequest:
    """Configuration selected by the UI before engine resolution."""

    dataset_root: str
    profile: str = "default"
    overrides: dict[str, Any] = field(default_factory=dict)
    enabled_checks: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_root.strip():
            raise ValueError("dataset_root must not be empty")
        if not self.profile.strip():
            raise ValueError("profile must not be empty")
        for check_id, enabled in self.enabled_checks.items():
            if not str(check_id).strip():
                raise ValueError("enabled_checks contains an empty check ID")
            if not isinstance(enabled, bool):
                raise TypeError(
                    f"Enabled state for {check_id!r} must be bool"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "profile": self.profile,
            "overrides": dict(self.overrides),
            "enabled_checks": dict(self.enabled_checks),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ValidationConfigRequest":
        return cls(
            dataset_root=str(payload["dataset_root"]),
            profile=str(payload.get("profile", "default")),
            overrides=_copy_mapping(payload.get("overrides")),
            enabled_checks={
                str(key): bool(value)
                for key, value in _copy_mapping(payload.get("enabled_checks")).items()
            },
        )


@dataclass(frozen=True)
class ValidationConfigResponse:
    """Engine-ready configuration plus validation diagnostics."""

    dataset_root: str
    profile: str
    resolved: dict[str, Any]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "profile": self.profile,
            "resolved": dict(self.resolved),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "valid": self.valid,
        }


# ---------------------------------------------------------------------------
# Run lifecycle / progress
# ---------------------------------------------------------------------------

class RunState(str, Enum):
    """Stable lifecycle states exposed to the UI."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RunCreateRequest:
    """Request to create a new validation run."""

    dataset_root: str
    config: ValidationConfigRequest

    def __post_init__(self) -> None:
        if not self.dataset_root.strip():
            raise ValueError("dataset_root must not be empty")
        if self.dataset_root != self.config.dataset_root:
            raise ValueError(
                "Run dataset_root must match config.dataset_root"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "config": self.config.to_dict(),
        }


@dataclass(frozen=True)
class ProgressEvent:
    """Structured event suitable for WebSocket or SSE transport."""

    event: str
    state: RunState
    run_id: str
    completed_episodes: int = 0
    total_episodes: int = 0
    current_episode: int | None = None
    phase: str | None = None
    current_check: str | None = None
    current_camera: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.event.strip():
            raise ValueError("event must not be empty")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.completed_episodes < 0 or self.total_episodes < 0:
            raise ValueError("episode counts must be >= 0")
        if self.current_episode is not None and self.current_episode < 0:
            raise ValueError("current_episode must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "state": self.state.value,
            "run_id": self.run_id,
            "completed_episodes": self.completed_episodes,
            "total_episodes": self.total_episodes,
            "current_episode": self.current_episode,
            "phase": self.phase,
            "current_check": self.current_check,
            "current_camera": self.current_camera,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Results / evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceReference:
    """Reference to a frame, video interval, or other stored evidence."""

    kind: str
    camera: str | None = None
    frame_range: tuple[int, int] | None = None
    location: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("Evidence kind must not be empty")
        if self.frame_range is not None:
            start, end = self.frame_range
            if start < 0 or end < start:
                raise ValueError("Invalid evidence frame_range")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "camera": self.camera,
            "frame_range": list(self.frame_range) if self.frame_range else None,
            "location": self.location,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EpisodeResultSummary:
    """Compact episode-level result row for the explorer/detail views."""

    episode_index: int
    score: float | None
    status: str
    hard_count: int = 0
    soft_count: int = 0
    frame_count: int | None = None
    duration_seconds: float | None = None
    instruction: str | None = None

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be >= 0")
        for name, value in (
            ("hard_count", self.hard_count),
            ("soft_count", self.soft_count),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")
        if self.frame_count is not None and self.frame_count < 0:
            raise ValueError("frame_count must be >= 0")
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("duration_seconds must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "score": self.score,
            "status": self.status,
            "hard_count": self.hard_count,
            "soft_count": self.soft_count,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class ResultSummary:
    """Dataset-level counts for the results dashboard."""

    total_episodes: int
    pass_count: int = 0
    review_count: int = 0
    reject_count: int = 0
    hard_failures: int = 0
    soft_warnings: int = 0
    per_check: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("total_episodes", self.total_episodes),
            ("pass_count", self.pass_count),
            ("review_count", self.review_count),
            ("reject_count", self.reject_count),
            ("hard_failures", self.hard_failures),
            ("soft_warnings", self.soft_warnings),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")

        if self.pass_count + self.review_count + self.reject_count > self.total_episodes:
            raise ValueError(
                "pass/review/reject counts cannot exceed total_episodes"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "pass_count": self.pass_count,
            "review_count": self.review_count,
            "reject_count": self.reject_count,
            "hard_failures": self.hard_failures,
            "soft_warnings": self.soft_warnings,
            "per_check": dict(self.per_check),
        }
