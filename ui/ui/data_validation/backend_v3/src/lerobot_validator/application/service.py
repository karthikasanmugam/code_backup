from __future__ import annotations

"""Application/service layer for the dataset validator.

This module is the bridge between the UI/API contract and the existing
validation engine. It performs orchestration, configuration resolution,
run-state management, and result shaping; it does not implement checks.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
import inspect
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

import yaml

from ..control_registry import (
    CHECK_REGISTRY,
    CheckDefinition,
    ControlDefinition,
    get_check_definition,
)
from ..dataset import load_dataset_info
from ..models import ValidationConfig
from ..validator import load_task_map, validate_dataset
from .schemas import (
    DatasetInspection,
    CameraSchema,
    EpisodeResultSummary,
    ProgressEvent,
    ResultSummary,
    RunCreateRequest,
    RunState,
    ValidationConfigRequest,
    ValidationConfigResponse,
)


_PROFILE_SUFFIXES = (".yml", ".yaml")


@dataclass
class _RunRecord:
    run_id: str
    dataset_root: Path
    profile: str
    resolved_config: dict[str, Any]
    state: RunState = RunState.QUEUED
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    progress: ProgressEvent | None = None
    validation: Any | None = None
    summary: ResultSummary | None = None
    error: str | None = None
    worker: threading.Thread | None = None
    cancel_event: threading.Event | None = None
    subscribers: list[Callable[[ProgressEvent], None]] | None = None

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.subscribers is None:
            self.subscribers = []


class ValidationService:
    """Application facade used by API/UI layers.

    Dataset paths are always supplied by the caller. The service reads
    dataset metadata dynamically and passes resolved configuration into the
    existing validator engine.
    """

    def __init__(
        self,
        *,
        config_dir: Path | str | None = None,
        runs_dir: Path | str | None = None,
        engine_validate_dataset: Callable[..., Any] = validate_dataset,
    ) -> None:
        package_root = Path(__file__).resolve().parents[3]
        self.config_dir = Path(config_dir) if config_dir is not None else package_root / "configs"
        self.runs_dir = Path(runs_dir) if runs_dir is not None else package_root / "runs"
        self._engine_validate_dataset = engine_validate_dataset
        self._runs: dict[str, _RunRecord] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Dataset inspection
    # ------------------------------------------------------------------

    def inspect_dataset(self, dataset_root: Path | str) -> DatasetInspection:
        root = self._require_dataset_root(dataset_root)
        info = load_dataset_info(root)

        raw = dict(getattr(info, "raw", {}) or {})
        total_episodes = self._int_value(
            raw,
            "total_episodes",
            default=0,
        )
        total_frames = self._int_value(
            raw,
            "total_frames",
            default=0,
        )
        total_tasks = self._int_value(
            raw,
            "total_tasks",
            default=0,
        )

        state_names = self._feature_names(info, "observation.state", "state")
        action_names = self._feature_names(info, "action", "actions")
        cameras = tuple(
            CameraSchema(
                key=key,
                width=getattr(camera, "width", None),
                height=getattr(camera, "height", None),
            )
            for key, camera in getattr(info, "cameras", {}).items()
        )

        tasks: list[str] = []
        try:
            task_map = load_task_map(root)
        except Exception:
            task_map = {}
        for key in sorted(task_map, key=str):
            value = task_map[key]
            if isinstance(value, Mapping):
                text = value.get("instruction") or value.get("task") or value.get("text")
            else:
                text = value
            if text is not None and str(text).strip():
                tasks.append(str(text).strip())

        raw_camera_keys = raw.get("camera_keys")
        camera_keys = list(raw_camera_keys) if isinstance(raw_camera_keys, list) else [camera.key for camera in cameras]

        extra = {
            "codebase_version": raw.get("codebase_version"),
            "repo_id": raw.get("repo_id"),
            "robot_type": raw.get("robot_type"),
            "use_videos": raw.get("use_videos"),
            "camera_keys": camera_keys,
            "chunks_size": raw.get("chunks_size"),
            "splits": raw.get("splits"),
        }
        extra = {key: value for key, value in extra.items() if value is not None}

        return DatasetInspection(
            dataset_root=str(root),
            fps=float(info.fps),
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_tasks=total_tasks,
            state_dim=int(getattr(info, "state_dim", 0)),
            action_dim=int(getattr(info, "action_dim", 0)),
            state_names=tuple(state_names),
            action_names=tuple(action_names),
            cameras=tuple(
                CameraSchema(
                    key=camera.key,
                    width=camera.width,
                    height=camera.height,
                )
                for camera in cameras
                if camera.key in camera_keys
            ),
            tasks=tuple(dict.fromkeys(tasks)),
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Control discovery
    # ------------------------------------------------------------------

    def get_check_definitions(
        self,
        dataset_root: Path | str | None = None,
    ) -> tuple[CheckDefinition, ...]:
        """Return registry entries, enriching dynamic selectors when metadata exists."""
        inspection = self.inspect_dataset(dataset_root) if dataset_root is not None else None
        action_options = tuple(inspection.action_names) if inspection else ()
        camera_options = tuple(camera.key for camera in inspection.cameras) if inspection else ()

        definitions: list[CheckDefinition] = []
        for definition in CHECK_REGISTRY:
            controls: list[ControlDefinition] = []
            for control in definition.controls:
                if control.control_type in {"feature_select", "feature_multi_select"}:
                    controls.append(replace(control, options=action_options))
                elif control.control_type == "camera_select":
                    controls.append(
                        replace(
                            control,
                            options=("auto", *camera_options),
                        )
                    )
                else:
                    controls.append(control)
            definitions.append(replace(definition, controls=tuple(controls)))
        return tuple(definitions)

    # ------------------------------------------------------------------
    # Configuration resolution
    # ------------------------------------------------------------------

    def resolve_config(
        self,
        request: ValidationConfigRequest,
        *,
        inspection: DatasetInspection | None = None,
    ) -> ValidationConfigResponse:
        root = self._require_dataset_root(request.dataset_root)
        inspection = inspection or self.inspect_dataset(root)

        warnings: list[str] = []
        errors: list[str] = []
        try:
            resolved = self._load_profile(request.profile)
        except FileNotFoundError as exc:
            return ValidationConfigResponse(
                dataset_root=str(root),
                profile=request.profile,
                resolved={},
                warnings=(),
                errors=(str(exc),),
            )
        except (OSError, ValueError, yaml.YAMLError) as exc:
            return ValidationConfigResponse(
                dataset_root=str(root),
                profile=request.profile,
                resolved={},
                warnings=(),
                errors=(f"Could not load profile '{request.profile}': {exc}",),
            )

        resolved = self._deep_merge(resolved, request.overrides)
        resolved.setdefault("dataset", {})
        if isinstance(resolved["dataset"], dict):
            resolved["dataset"]["data_root"] = str(root)

        self._apply_check_toggles(resolved, request.enabled_checks, warnings, errors)
        self._validate_dynamic_values(resolved, inspection, warnings, errors)
        self._validate_registry_bindings(request.enabled_checks, warnings)
        self._validate_basic_config_shape(resolved, errors)

        return ValidationConfigResponse(
            dataset_root=str(root),
            profile=request.profile,
            resolved=resolved,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(self, request: RunCreateRequest) -> str:
        response = self.resolve_config(request.config)
        if not response.valid:
            raise ValueError("; ".join(response.errors))

        run_id = uuid4().hex
        record = _RunRecord(
            run_id=run_id,
            dataset_root=Path(request.dataset_root),
            profile=request.config.profile,
            resolved_config=deepcopy(response.resolved),
            cancel_event=threading.Event(),
        )
        record.progress = ProgressEvent(
            event="run_queued",
            state=RunState.QUEUED,
            run_id=run_id,
            total_episodes=0,
            message="Validation run queued",
        )

        with self._lock:
            self._runs[run_id] = record

        self._persist_run(record)
        self._publish(record, record.progress)
        return run_id

    def start_run(self, run_id: str) -> None:
        with self._lock:
            record = self._get_run(run_id)
            if record.state not in {RunState.QUEUED}:
                raise ValueError(f"Run {run_id} is not queued")
            record.state = RunState.RUNNING
            record.started_at = time.time()
            worker = threading.Thread(
                target=self._execute_run,
                args=(run_id,),
                name=f"validator-run-{run_id[:8]}",
                daemon=True,
            )
            record.worker = worker
            record.progress = ProgressEvent(
                event="run_started",
                state=RunState.RUNNING,
                run_id=run_id,
                message="Validation run started",
            )
            progress = record.progress

        self._persist_run(record)
        self._publish(record, progress)
        worker.start()

    def run_sync(self, request: RunCreateRequest) -> str:
        """Create and execute a run in the caller's thread; useful for CLI/tests."""
        run_id = self.create_run(request)
        self._execute_run(run_id)
        return run_id

    def stop_run(self, run_id: str) -> None:
        with self._lock:
            record = self._get_run(run_id)
            if record.state not in {RunState.QUEUED, RunState.RUNNING}:
                return
            assert record.cancel_event is not None
            record.cancel_event.set()
            if record.state == RunState.QUEUED:
                record.state = RunState.CANCELLED
                record.finished_at = time.time()
                record.progress = ProgressEvent(
                    event="run_cancelled",
                    state=RunState.CANCELLED,
                    run_id=run_id,
                    message="Validation run cancelled before execution",
                )
                self._persist_run(record)
                self._publish(record, record.progress)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._get_run(run_id)
            return self._run_to_dict(record)

    def subscribe(self, run_id: str, callback: Callable[[ProgressEvent], None]) -> None:
        with self._lock:
            record = self._get_run(run_id)
            assert record.subscribers is not None
            record.subscribers.append(callback)

    def get_run_validation(self, run_id: str) -> Any | None:
        with self._lock:
            return self._get_run(run_id).validation

    def get_run_summary(self, run_id: str) -> ResultSummary | None:
        with self._lock:
            return self._get_run(run_id).summary

    # ------------------------------------------------------------------
    # Internal execution / persistence
    # ------------------------------------------------------------------

    def _execute_run(self, run_id: str) -> None:
        with self._lock:
            record = self._get_run(run_id)
            if record.state == RunState.CANCELLED:
                return
            record.state = RunState.RUNNING
            record.started_at = record.started_at or time.time()
            cancel_event = record.cancel_event

        try:
            if cancel_event is not None and cancel_event.is_set():
                raise _RunCancelled()

            dataset_info = self.inspect_dataset(record.dataset_root)
            self._update_progress(
                record,
                ProgressEvent(
                    event="dataset_started",
                    state=RunState.RUNNING,
                    run_id=run_id,
                    completed_episodes=0,
                    total_episodes=dataset_info.total_episodes,
                    phase="dataset_setup",
                    message="Dataset inspection complete",
                ),
            )

            if cancel_event is not None and cancel_event.is_set():
                raise _RunCancelled()

            engine_config = self._build_engine_config(record.resolved_config)
            progress_lock = threading.Lock()
            completed_episodes = 0

            def report_episode_progress(update: dict[str, Any]) -> None:
                nonlocal completed_episodes
                if update.get("event") != "episode_completed":
                    return
                with progress_lock:
                    completed_episodes += 1
                    completed = completed_episodes
                total = int(update.get("total_episodes", 0))
                episode_index = int(update.get("episode_index", 0))
                self._update_progress(
                    record,
                    ProgressEvent(
                        event="episode_completed",
                        state=RunState.RUNNING,
                        run_id=run_id,
                        completed_episodes=completed,
                        total_episodes=total,
                        current_episode=episode_index,
                        phase="episode_validation",
                        message=f"Episode {episode_index} completed ({completed}/{total})",
                    ),
                    persist=False,
                )

            def raise_if_cancelled() -> None:
                if cancel_event is not None and cancel_event.is_set():
                    raise _RunCancelled()

            engine_kwargs = {
                "data_root": record.dataset_root,
                "config": engine_config,
                "cancel_check": raise_if_cancelled,
            }
            try:
                parameters = inspect.signature(self._engine_validate_dataset).parameters
            except (TypeError, ValueError):
                parameters = {}
            accepts_extra_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if "progress_callback" in parameters or accepts_extra_kwargs:
                engine_kwargs["progress_callback"] = report_episode_progress
            if "cancel_check" not in parameters and not accepts_extra_kwargs:
                engine_kwargs.pop("cancel_check")
            validation = self._engine_validate_dataset(**engine_kwargs)
            summary = self._summarise_validation(validation)

            if cancel_event is not None and cancel_event.is_set():
                raise _RunCancelled()

            with self._lock:
                record.validation = validation
                record.summary = summary
                record.state = RunState.COMPLETED
                record.finished_at = time.time()
                record.progress = ProgressEvent(
                    event="dataset_finished",
                    state=RunState.COMPLETED,
                    run_id=run_id,
                    completed_episodes=summary.total_episodes,
                    total_episodes=summary.total_episodes,
                    message="Validation run completed",
                )
                progress = record.progress

            self._persist_run(record)
            self._publish(record, progress)

        except _RunCancelled:
            with self._lock:
                record.state = RunState.CANCELLED
                record.finished_at = time.time()
                record.progress = ProgressEvent(
                    event="run_cancelled",
                    state=RunState.CANCELLED,
                    run_id=run_id,
                    message="Validation run cancelled",
                )
                progress = record.progress
            self._persist_run(record)
            self._publish(record, progress)
        except Exception as exc:
            with self._lock:
                record.state = RunState.FAILED
                record.finished_at = time.time()
                record.error = str(exc)
                record.progress = ProgressEvent(
                    event="run_failed",
                    state=RunState.FAILED,
                    run_id=run_id,
                    message=str(exc),
                )
                progress = record.progress
            self._persist_run(record)
            self._publish(record, progress)

    def _update_progress(
        self,
        record: _RunRecord,
        event: ProgressEvent,
        *,
        persist: bool = True,
    ) -> None:
        with self._lock:
            record.progress = event
        if persist:
            self._persist_run(record)
        self._publish(record, event)

    def _publish(self, record: _RunRecord, event: ProgressEvent) -> None:
        callbacks = list(record.subscribers or [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                # Subscriber failures must not fail a validation run.
                continue

    def _persist_run(self, record: _RunRecord) -> None:
        run_dir = self.runs_dir / record.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(run_dir / "config.json", record.resolved_config)
        self._write_json(
            run_dir / "dataset.json",
            {"dataset_root": str(record.dataset_root)},
        )
        self._write_json(
            run_dir / "progress.json",
            record.progress.to_dict() if record.progress else {},
        )
        self._write_json(
            run_dir / "summary.json",
            record.summary.to_dict() if record.summary else {},
        )
        self._write_json(
            run_dir / "results.json",
            self._jsonable(record.validation) if record.validation is not None else {},
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _load_profile(self, profile: str) -> dict[str, Any]:
        path = Path(profile)
        if not path.is_absolute():
            filename = profile if path.suffix in _PROFILE_SUFFIXES else f"{profile}.yml"
            path = self.config_dir / filename
        if not path.exists():
            alt = path.with_suffix(".yaml") if path.suffix == ".yml" else path.with_suffix(".yml")
            if alt.exists():
                path = alt
        if not path.exists():
            raise FileNotFoundError(f"Configuration profile not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration profile must contain a mapping: {path}")
        return data

    @staticmethod
    def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(base))
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = ValidationService._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def _apply_check_toggles(
        self,
        resolved: dict[str, Any],
        toggles: Mapping[str, bool],
        warnings: list[str],
        errors: list[str],
    ) -> None:
        for check_id, enabled in toggles.items():
            try:
                definition = get_check_definition(check_id)
            except KeyError as exc:
                errors.append(str(exc))
                continue

            if not enabled:
                phase_checks = [d for d in CHECK_REGISTRY if d.phase == definition.phase]
                all_disabled = all(
                    toggles.get(other.id, other.enabled_by_default) is False
                    for other in phase_checks
                )
                if all_disabled:
                    phase = resolved.setdefault(definition.phase, {})
                    if isinstance(phase, dict):
                        phase["enabled"] = False
                else:
                    warnings.append(
                        f"Check '{check_id}' is individually disabled, but the current engine does not expose a phase-independent toggle for it."
                    )
            else:
                self._set_supported_check_enabled(resolved, definition)

    @staticmethod
    def _set_supported_check_enabled(
        resolved: dict[str, Any],
        definition: CheckDefinition,
    ) -> None:
        # Only set subsection switches that the existing engine already reads.
        if definition.id.startswith("actions."):
            subsection = definition.id.split(".", 1)[1]
            section = resolved.setdefault("actions", {})
            if isinstance(section, dict):
                target = section.setdefault(subsection, {})
                if isinstance(target, dict):
                    target.setdefault("enabled", True)
        elif definition.id == "video.image_quality":
            video = resolved.setdefault("video", {})
            if isinstance(video, dict):
                image = video.setdefault("image_quality", {})
                if isinstance(image, dict):
                    image.setdefault("enabled", True)
        elif definition.id == "video.frozen_video":
            video = resolved.setdefault("video", {})
            if isinstance(video, dict):
                frozen = video.setdefault("frozen_video", {})
                if isinstance(frozen, dict):
                    frozen.setdefault("enabled", True)
        elif definition.id == "video.multiview_sync":
            video = resolved.setdefault("video", {})
            if isinstance(video, dict):
                sync = video.setdefault("multiview_sync", {})
                if isinstance(sync, dict):
                    sync.setdefault("enabled", True)

    def _validate_dynamic_values(
        self,
        resolved: Mapping[str, Any],
        inspection: DatasetInspection,
        warnings: list[str],
        errors: list[str],
    ) -> None:
        actions = resolved.get("actions", {})
        if isinstance(actions, Mapping):
            smoothness = actions.get("smoothness", {})
            if isinstance(smoothness, Mapping):
                excluded = smoothness.get("exclude_dimensions", [])
                unknown = [name for name in excluded if str(name) not in inspection.action_names]
                if unknown:
                    errors.append(
                        "Unknown excluded action dimensions: " + ", ".join(map(str, unknown))
                    )

            gripper = actions.get("gripper", {})
            if isinstance(gripper, Mapping):
                dimension = gripper.get("dimension")
                if dimension is not None:
                    if isinstance(dimension, str) and dimension in inspection.action_names:
                        # The UI selects by the dataset feature name, while the
                        # existing engine expects an integer action index.
                        # Normalize the resolved config at the application boundary.
                        gripper["dimension"] = inspection.action_names.index(dimension)
                    else:
                        try:
                            index = int(dimension)
                        except (TypeError, ValueError):
                            errors.append(f"Unknown gripper dimension: {dimension!r}")
                        else:
                            if not 0 <= index < inspection.action_dim:
                                errors.append(
                                    f"Gripper dimension {index} is outside action dimension {inspection.action_dim}."
                                )

        video = resolved.get("video", {})
        if isinstance(video, Mapping):
            sync = video.get("multiview_sync", {})
            if isinstance(sync, Mapping):
                reference = sync.get("reference_camera", "auto")
                camera_keys = {camera.key for camera in inspection.cameras}
                if reference != "auto" and reference not in camera_keys:
                    errors.append(
                        f"Reference camera '{reference}' is not declared by the dataset."
                    )

    @staticmethod
    def _validate_registry_bindings(
        enabled_checks: Mapping[str, bool],
        warnings: list[str],
    ) -> None:
        for check_id in enabled_checks:
            definition = next((d for d in CHECK_REGISTRY if d.id == check_id), None)
            if definition is None:
                continue
            if any(control.binding_status != "active" for control in definition.controls):
                warnings.append(
                    f"Check '{check_id}' has one or more controls that are not fully wired to the engine yet."
                )

    @staticmethod
    def _validate_basic_config_shape(
        resolved: Mapping[str, Any],
        errors: list[str],
    ) -> None:
        if not isinstance(resolved, Mapping):
            errors.append("Resolved configuration must be a mapping.")
            return
        for section_name in ("dataset", "structural", "video", "actions", "episode", "dataset_quality", "scoring", "reporting"):
            value = resolved.get(section_name)
            if value is not None and not isinstance(value, Mapping):
                errors.append(f"Configuration section '{section_name}' must be a mapping.")

    # ------------------------------------------------------------------
    # Result shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_validation(validation: Any) -> ResultSummary:
        episodes = list(getattr(validation, "episodes", None) or getattr(validation, "episode_results", None) or [])
        total = len(episodes)
        pass_count = review_count = reject_count = hard_failures = soft_warnings = 0
        per_check: dict[str, int] = {}

        for episode in episodes:
            results = list(getattr(episode, "results", []) or [])
            for result in results:
                severity = str(getattr(result, "severity", "")).upper()
                check = str(getattr(result, "check", "unknown"))
                per_check[check] = per_check.get(check, 0) + 1
                if severity == "HARD":
                    hard_failures += 1
                elif severity == "SOFT":
                    soft_warnings += 1

            status = str(
                getattr(episode, "classification", None)
                or getattr(episode, "status", None)
                or "review"
            ).lower()
            if status == "pass":
                pass_count += 1
            elif status == "reject":
                reject_count += 1
            else:
                review_count += 1

        return ResultSummary(
            total_episodes=total,
            pass_count=pass_count,
            review_count=review_count,
            reject_count=reject_count,
            hard_failures=hard_failures,
            soft_warnings=soft_warnings,
            per_check=per_check,
        )

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_names(info: Any, *candidates: str) -> tuple[str, ...]:
        features = getattr(info, "features", {}) or {}
        for name in candidates:
            feature = features.get(name)
            if feature is None:
                continue
            raw = getattr(feature, "info", {}) or {}
            names = raw.get("names") if isinstance(raw, Mapping) else None
            if isinstance(names, (list, tuple)):
                return tuple(str(value) for value in names)
            if isinstance(names, Mapping):
                return tuple(str(value) for value in names.keys())
        return ()

    @staticmethod
    def _int_value(mapping: Mapping[str, Any], key: str, default: int) -> int:
        try:
            return int(mapping.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _require_dataset_root(dataset_root: Path | str) -> Path:
        root = Path(dataset_root).expanduser()
        if not str(root).strip():
            raise ValueError("dataset_root must not be empty")
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Dataset root is not a directory: {root}")
        return root.resolve()

    def _get_run(self, run_id: str) -> _RunRecord:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown validation run: {run_id}") from exc

    @staticmethod
    def _build_engine_config(resolved: Mapping[str, Any]) -> ValidationConfig:
        """Construct ValidationConfig while tolerating small constructor variations."""
        payload = dict(resolved)
        skip = bool(payload.pop("skip_expensive_after_hard", True))
        try:
            signature = inspect.signature(ValidationConfig)
            params = signature.parameters
        except (TypeError, ValueError):
            params = {}

        kwargs: dict[str, Any] = {}
        if "raw" in params:
            kwargs["raw"] = payload
        elif "config" in params:
            kwargs["config"] = payload
        elif not params:
            return ValidationConfig(payload)  # pragma: no cover
        else:
            # The current project model uses raw as the configuration payload.
            kwargs["raw"] = payload

        if "skip_expensive_after_hard" in params:
            kwargs["skip_expensive_after_hard"] = skip

        return ValidationConfig(**kwargs)

    @classmethod
    def _run_to_dict(cls, record: _RunRecord) -> dict[str, Any]:
        return {
            "run_id": record.run_id,
            "dataset_root": str(record.dataset_root),
            "profile": record.profile,
            "state": record.state.value,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "progress": record.progress.to_dict() if record.progress else None,
            "summary": record.summary.to_dict() if record.summary else None,
            "error": record.error,
            "config": deepcopy(record.resolved_config),
        }

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return {key: ValidationService._jsonable(val) for key, val in asdict(value).items()}
        if isinstance(value, Mapping):
            return {str(key): ValidationService._jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [ValidationService._jsonable(item) for item in value]
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            return ValidationService._jsonable(vars(value))
        return str(value)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(ValidationService._jsonable(value), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class _RunCancelled(Exception):
    pass


__all__ = ["ValidationService"]
