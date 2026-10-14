from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ControlDefinition:
    """UI-facing definition for one editable validation parameter."""

    id: str
    label: str
    control_type: str
    config_path: tuple[str, ...]
    default: Any = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    unit: str | None = None
    description: str = ""
    source: str = "user"
    binding_status: str = "active"
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckDefinition:
    """Stable description of a validator check exposed to the UI."""

    id: str
    label: str
    phase: str
    severity: str
    enabled_by_default: bool = True
    description: str = ""
    controls: tuple[ControlDefinition, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Control registry
# ---------------------------------------------------------------------------
#
# This registry is intentionally descriptive. It does not run validation and
# does not duplicate the check algorithms. The backend/application layer will
# translate these definitions into resolved ValidationConfig values.
# ---------------------------------------------------------------------------

CHECK_REGISTRY: tuple[CheckDefinition, ...] = (
    CheckDefinition(
        id="structural.metadata_length",
        label="Metadata length",
        phase="structural",
        severity="HARD",
        description="Compare each episode row count with its metadata length.",
    ),
    CheckDefinition(
        id="structural.frame_index",
        label="Frame index continuity",
        phase="structural",
        severity="HARD",
        description="Require frame_index to equal a contiguous zero-based range.",
    ),
    CheckDefinition(
        id="structural.required_columns",
        label="Required columns",
        phase="structural",
        severity="HARD",
        description="Verify required structural columns are present.",
    ),
    CheckDefinition(
        id="structural.dimension_consistency",
        label="State/action dimensions",
        phase="structural",
        severity="HARD",
        description="Compare observed vector dimensions with dataset metadata.",
    ),
    CheckDefinition(
        id="structural.nan_inf",
        label="NaN / Inf",
        phase="structural",
        severity="HARD",
        description="Reject non-finite state or action values.",
    ),
    CheckDefinition(
        id="structural.episode_length",
        label="Episode length",
        phase="structural",
        severity="HARD",
        controls=(
            ControlDefinition(
                id="minimum_frames",
                label="Minimum episode length",
                control_type="integer",
                config_path=("structural", "min_episode_length"),
                default=20,
                minimum=1,
                unit="frames",
                description="Minimum number of frames allowed in an episode.",
            ),
            ControlDefinition(
                id="maximum_frames",
                label="Maximum episode length",
                control_type="integer",
                config_path=("structural", "max_episode_length"),
                default=1000,
                minimum=1,
                unit="frames",
                description="Maximum number of frames allowed in an episode.",
            ),
        ),
    ),
    CheckDefinition(
        id="video.frame_count",
        label="Video frame count",
        phase="video",
        severity="HARD",
        controls=(
            ControlDefinition(
                id="tolerance_frames",
                label="Frame-count tolerance",
                control_type="integer",
                config_path=("video", "frame_count_tolerance"),
                default=0,
                minimum=0,
                maximum=10,
                unit="frames",
                description=(
                    "Maximum allowed difference between video and parquet frame counts."
                ),
            ),
        ),
    ),
    CheckDefinition(
        id="video.resolution",
        label="Video resolution",
        phase="video",
        severity="HARD",
        description="Compare decoded video resolution with dataset-declared camera resolution.",
    ),
    CheckDefinition(
        id="video.timestamps",
        label="Video timestamps",
        phase="video",
        severity="HARD",
        controls=(
            ControlDefinition(
                id="tolerance_seconds",
                label="Timestamp tolerance",
                control_type="number",
                config_path=("video", "timestamp_tolerance_s"),
                default=None,
                minimum=0,
                unit="seconds",
                description="Maximum permitted video/parquet timestamp deviation. Use Auto to derive from FPS.",
                source="auto_or_user",
            ),
        ),
    ),
    CheckDefinition(
        id="video.image_quality",
        label="Image quality",
        phase="video",
        severity="SOFT",
        controls=(
            ControlDefinition(
                id="sample_every",
                label="Sample every",
                control_type="integer",
                config_path=("video", "image_quality", "sample_every"),
                default=15,
                minimum=1,
                unit="frames",
                description="Analyze every Nth frame for image-quality checks.",
            ),
            ControlDefinition(
                id="blur_threshold",
                label="Blur threshold",
                control_type="number",
                config_path=("video", "image_quality", "blur_threshold"),
                default=50.0,
                minimum=0,
                unit="score",
            ),
            ControlDefinition(
                id="dark_threshold",
                label="Dark threshold",
                control_type="number",
                config_path=("video", "image_quality", "dark_threshold"),
                default=10.0,
                minimum=0,
                maximum=255,
                unit="pixel value",
            ),
            ControlDefinition(
                id="bright_threshold",
                label="Bright threshold",
                control_type="number",
                config_path=("video", "image_quality", "bright_threshold"),
                default=245.0,
                minimum=0,
                maximum=255,
                unit="pixel value",
            ),
            ControlDefinition(
                id="max_bad_ratio",
                label="Maximum bad-frame ratio",
                control_type="number",
                config_path=("video", "image_quality", "max_bad_ratio"),
                default=0.05,
                minimum=0,
                maximum=1,
                step=0.01,
                unit="ratio",
            ),
        ),
    ),
    CheckDefinition(
        id="video.frozen_video",
        label="Frozen video",
        phase="video",
        severity="HARD",
        controls=(
            ControlDefinition(
                id="diff_epsilon",
                label="Frame-difference epsilon",
                control_type="number",
                config_path=("video", "frozen_video", "diff_epsilon"),
                default=0.001,
                minimum=0,
            ),
            ControlDefinition(
                id="action_epsilon",
                label="Action-difference epsilon",
                control_type="number",
                config_path=("video", "frozen_video", "action_epsilon"),
                default=0.03,
                minimum=0,
            ),
            ControlDefinition(
                id="min_run_seconds",
                label="Minimum frozen duration",
                control_type="number",
                config_path=("video", "frozen_video", "min_run_seconds"),
                default=2.0,
                minimum=0,
                unit="seconds",
            ),
        ),
    ),
    CheckDefinition(
        id="video.multiview_sync",
        label="Multiview synchronization",
        phase="video",
        severity="HARD",
        controls=(
            ControlDefinition(
                id="tolerance_frames",
                label="Sync tolerance",
                control_type="integer",
                config_path=("video", "multiview_sync", "tolerance_frames"),
                default=1,
                minimum=0,
                unit="frames",
            ),
            ControlDefinition(
                id="reference_camera",
                label="Reference camera",
                control_type="camera_select",
                config_path=("video", "multiview_sync", "reference_camera"),
                default="auto",
                source="dataset",
                description="Camera list is populated dynamically from info.json camera_keys.",
            ),
        ),
    ),
    CheckDefinition(
        id="actions.range",
        label="Action range",
        phase="actions",
        severity="HARD",
        controls=(
            ControlDefinition(
                id="max_clip_ratio",
                label="Maximum clip ratio",
                control_type="number",
                config_path=("actions", "range", "max_clip_ratio"),
                default=0.02,
                minimum=0,
                maximum=1,
                step=0.01,
                unit="ratio",
            ),
        ),
    ),
    CheckDefinition(
        id="actions.smoothness",
        label="Action smoothness",
        phase="actions",
        severity="SOFT",
        controls=(
            ControlDefinition(
                id="jerk_z_threshold",
                label="Jerk Z threshold",
                control_type="number",
                config_path=("actions", "smoothness", "jerk_z_threshold"),
                default=6.0,
                minimum=0,
                unit="score",
            ),
            ControlDefinition(
                id="exclude_dimensions",
                label="Excluded dimensions",
                control_type="feature_multi_select",
                config_path=("actions", "smoothness", "exclude_dimensions"),
                default=[],
                source="dataset",
                description="Options are populated from the dataset's actual action feature names.",
            ),
        ),
    ),
    CheckDefinition(
        id="actions.gripper",
        label="Gripper transitions",
        phase="actions",
        severity="SOFT",
        controls=(
            ControlDefinition(
                id="dimension",
                label="Gripper dimension",
                control_type="feature_select",
                config_path=("actions", "gripper", "dimension"),
                default=None,
                source="dataset",
                description="Select a detected gripper feature; the UI must not hard-code dimension indices.",
            ),
            ControlDefinition(
                id="open_close_threshold",
                label="Open/close threshold",
                control_type="number",
                config_path=("actions", "gripper", "open_close_threshold"),
                default=None,
                minimum=0,
            ),
            ControlDefinition(
                id="max_toggles",
                label="Maximum toggles",
                control_type="integer",
                config_path=("actions", "gripper", "max_toggles"),
                default=6,
                minimum=0,
            ),
            ControlDefinition(
                id="min_hold_frames",
                label="Minimum hold frames",
                control_type="integer",
                config_path=("actions", "gripper", "min_hold_frames"),
                default=5,
                minimum=0,
                unit="frames",
            ),
        ),
    ),
    CheckDefinition(
        id="actions.state_lag",
        label="State/action lag",
        phase="actions",
        severity="SOFT",
        controls=(
            ControlDefinition(
                id="max_lag_frames",
                label="Maximum lag",
                control_type="integer",
                config_path=("actions", "state_lag", "max_lag_frames"),
                default=5,
                minimum=0,
                unit="frames",
            ),
            ControlDefinition(
                id="min_correlation",
                label="Minimum correlation",
                control_type="number",
                config_path=("actions", "state_lag", "min_correlation"),
                default=0.5,
                minimum=-1,
                maximum=1,
                step=0.01,
            ),
        ),
    ),
    CheckDefinition(
        id="actions.alignment",
        label="Action alignment",
        phase="actions",
        severity="HARD",
    ),
    CheckDefinition(
        id="actions.token_budget",
        label="Token budget",
        phase="actions",
        severity="HARD",
        controls=(
            ControlDefinition(
                id="chunk_length",
                label="Chunk length",
                control_type="integer",
                config_path=("actions", "token_budget", "chunk_length"),
                default=None,
                minimum=1,
                unit="frames",
            ),
            ControlDefinition(
                id="tokens_per_dimension",
                label="Tokens per dimension",
                control_type="integer",
                config_path=("actions", "token_budget", "tokens_per_dimension"),
                default=None,
                minimum=1,
                unit="tokens",
            ),
            ControlDefinition(
                id="max_tokens",
                label="Maximum tokens",
                control_type="integer",
                config_path=("actions", "token_budget", "max_tokens"),
                default=None,
                minimum=1,
                unit="tokens",
            ),
        ),
    ),
    CheckDefinition(
        id="episode.idle_ratio",
        label="Idle ratio",
        phase="episode",
        severity="SOFT",
        controls=(
            ControlDefinition(
                id="velocity_epsilon",
                label="Velocity epsilon",
                control_type="number",
                config_path=("episode", "idle", "velocity_epsilon"),
                default=0.001,
                minimum=0,
            ),
            ControlDefinition(
                id="max_idle_ratio",
                label="Maximum idle ratio",
                control_type="number",
                config_path=("episode", "idle", "max_idle_ratio"),
                default=0.50,
                minimum=0,
                maximum=1,
                step=0.01,
                unit="ratio",
            ),
        ),
    ),
    CheckDefinition(
        id="episode.boundary_idle",
        label="Boundary idle",
        phase="episode",
        severity="SOFT",
        controls=(
            ControlDefinition(
                id="max_idle_seconds",
                label="Maximum boundary idle",
                control_type="number",
                config_path=("episode", "boundary_idle", "max_idle_seconds"),
                default=2.0,
                minimum=0,
                unit="seconds",
            ),
        ),
    ),
    CheckDefinition(
        id="episode.duration_outlier",
        label="Duration outlier",
        phase="episode",
        severity="SOFT",
        controls=(
            ControlDefinition(
                id="mad_z_threshold",
                label="MAD Z threshold",
                control_type="number",
                config_path=("episode", "duration", "mad_z_threshold"),
                default=3.5,
                minimum=0,
            ),
        ),
    ),
    CheckDefinition(
        id="episode.path_efficiency",
        label="Path efficiency",
        phase="episode",
        severity="SOFT",
        controls=(
            ControlDefinition(
                id="ratio_threshold",
                label="Maximum path ratio",
                control_type="number",
                config_path=("episode", "path_efficiency", "ratio_threshold"),
                default=2.0,
                minimum=1,
            ),
        ),
    ),
    CheckDefinition(
        id="episode.task_instruction",
        label="Task instruction",
        phase="episode",
        severity="HARD",
        controls=(),
    ),
    CheckDefinition(
        id="episode.proprio_sync",
        label="Proprioceptive synchronization",
        phase="episode",
        severity="HARD",
        controls=(
            ControlDefinition(
                id="tolerance_seconds",
                label="Sync tolerance",
                control_type="number",
                config_path=("episode", "proprio_sync", "tolerance_seconds"),
                default=None,
                minimum=0,
                unit="seconds",
                source="auto_or_user",
            ),
        ),
    ),
)


REGISTRY_BY_ID = {definition.id: definition for definition in CHECK_REGISTRY}


PHASE_ORDER = (
    "structural",
    "video",
    "actions",
    "episode",
    "dataset_quality",
)


def get_check_definition(check_id: str) -> CheckDefinition:
    """Return one check definition or raise a descriptive KeyError."""
    try:
        return REGISTRY_BY_ID[check_id]
    except KeyError as exc:
        raise KeyError(f"Unknown validation check: {check_id}") from exc


def get_checks_by_phase(phase: str) -> tuple[CheckDefinition, ...]:
    """Return checks belonging to one validation phase."""
    return tuple(
        definition
        for definition in CHECK_REGISTRY
        if definition.phase == phase
    )


def get_control(check_id: str, control_id: str) -> ControlDefinition:
    """Return a control definition from a check definition."""
    definition = get_check_definition(check_id)

    for control in definition.controls:
        if control.id == control_id:
            return control

    raise KeyError(
        f"Unknown control '{control_id}' for check '{check_id}'"
    )
