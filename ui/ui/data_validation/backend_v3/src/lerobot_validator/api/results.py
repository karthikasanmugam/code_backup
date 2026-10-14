from __future__ import annotations

"""Result and evidence HTTP routes for the validator UI.

The application service remains the source of truth for validation runs.
This module only shapes persisted/in-memory engine results for the UI.
"""

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..application.service import ValidationService


def _jsonable(value: Any) -> Any:
    """Convert engine/application values into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _episodes(validation: Any) -> list[Any]:
    if validation is None:
        return []
    return list(
        getattr(validation, "episodes", None)
        or getattr(validation, "episode_results", None)
        or []
    )


def _episode_index(episode: Any) -> int:
    return int(getattr(episode, "episode_index"))


def _result_dict(result: Any) -> dict[str, Any]:
    """Shape one engine CheckResult for the UI."""
    payload = _jsonable(result)
    if not isinstance(payload, dict):
        return {"message": str(payload)}

    frame_range = payload.get("frame_range")
    if isinstance(frame_range, tuple):
        payload["frame_range"] = list(frame_range)
    return payload


def _episode_summary(episode: Any) -> dict[str, Any]:
    results = list(getattr(episode, "results", []) or [])
    hard = sum(str(getattr(item, "severity", "")).upper() == "HARD" for item in results)
    soft = sum(str(getattr(item, "severity", "")).upper() == "SOFT" for item in results)

    return {
        "episode_index": _episode_index(episode),
        "score": getattr(episode, "score", None),
        "status": str(
            getattr(episode, "classification", None)
            or getattr(episode, "status", None)
            or "review"
        ).lower(),
        "hard_count": hard,
        "soft_count": soft,
    }


def _episode_detail(episode: Any) -> dict[str, Any]:
    payload = _episode_summary(episode)
    payload["results"] = [_result_dict(item) for item in (getattr(episode, "results", []) or [])]

    # Preserve useful episode metadata when supplied by an engine/model,
    # without assuming that every dataset has the same attributes.
    for attr, output_key in (
        ("frame_count", "frame_count"),
        ("duration_seconds", "duration_seconds"),
        ("instruction", "instruction"),
        ("task", "task"),
    ):
        value = getattr(episode, attr, None)
        if value is not None:
            payload[output_key] = _jsonable(value)

    return payload


def _episode_evidence(episode: Any) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for result in (getattr(episode, "results", []) or []):
        frame_range = getattr(result, "frame_range", None)
        if frame_range is None:
            continue
        try:
            start, end = frame_range
        except (TypeError, ValueError):
            continue
        evidence.append(
            {
                "kind": "check_result",
                "check": str(getattr(result, "check", "unknown")),
                "severity": str(getattr(result, "severity", "")),
                "frame_range": [int(start), int(end)],
                "message": str(getattr(result, "message", "")),
            }
        )
    return evidence


def _find_episode(validation: Any, episode_id: int) -> Any:
    for episode in _episodes(validation):
        if _episode_index(episode) == episode_id:
            return episode
    raise KeyError(f"Episode {episode_id} not found")


def create_router(service: ValidationService) -> APIRouter:
    """Create read-only result routes bound to one application service."""
    router = APIRouter(prefix="/api/runs", tags=["results"])

    @router.get("/{run_id}/episodes")
    def list_episodes(
        run_id: str,
        status: str | None = Query(default=None),
        hard_only: bool = Query(default=False),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        try:
            validation = service.get_run_validation(run_id)
            if validation is None:
                raise ValueError(f"Run {run_id} has no validation results")

            rows = [_episode_summary(episode) for episode in _episodes(validation)]
            if status:
                wanted = status.lower()
                rows = [row for row in rows if row["status"] == wanted]
            if hard_only:
                rows = [row for row in rows if row["hard_count"] > 0]

            total = len(rows)
            rows = rows[offset : offset + limit]
            return {
                "run_id": run_id,
                "total": total,
                "offset": offset,
                "limit": limit,
                "episodes": rows,
            }
        except Exception as exc:
            raise _http_error(exc)

    @router.get("/{run_id}/episodes/{episode_id}")
    def get_episode(run_id: str, episode_id: int) -> dict[str, Any]:
        try:
            validation = service.get_run_validation(run_id)
            if validation is None:
                raise ValueError(f"Run {run_id} has no validation results")
            return _episode_detail(_find_episode(validation, episode_id))
        except Exception as exc:
            raise _http_error(exc)

    @router.get("/{run_id}/episodes/{episode_id}/evidence")
    def get_episode_evidence(run_id: str, episode_id: int) -> dict[str, Any]:
        try:
            validation = service.get_run_validation(run_id)
            if validation is None:
                raise ValueError(f"Run {run_id} has no validation results")
            return {
                "run_id": run_id,
                "episode_index": episode_id,
                "evidence": _episode_evidence(_find_episode(validation, episode_id)),
            }
        except Exception as exc:
            raise _http_error(exc)

    return router


__all__ = ["create_router"]
