from __future__ import annotations

"""Validation-run HTTP routes.

This module is a thin transport adapter over ``ValidationService``.  Run
orchestration, cancellation, persistence, and result generation stay in the
application layer.
"""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..application.schemas import RunCreateRequest, ValidationConfigRequest
from ..application.service import ValidationService


class ValidationConfigBody(BaseModel):
    """UI configuration payload embedded in a run-create request."""

    model_config = ConfigDict(extra="forbid")

    dataset_root: str = Field(min_length=1)
    profile: str = Field(default="default", min_length=1)
    overrides: dict[str, Any] = Field(default_factory=dict)
    enabled_checks: dict[str, bool] = Field(default_factory=dict)


class RunCreateBody(BaseModel):
    """HTTP request for creating and starting a validation run."""

    model_config = ConfigDict(extra="forbid")

    dataset_root: str = Field(min_length=1)
    config: ValidationConfigBody


def _config_request(body: ValidationConfigBody) -> ValidationConfigRequest:
    return ValidationConfigRequest(
        dataset_root=body.dataset_root,
        profile=body.profile,
        overrides=body.overrides,
        enabled_checks=body.enabled_checks,
    )


def _run_request(body: RunCreateBody) -> RunCreateRequest:
    return RunCreateRequest(
        dataset_root=body.dataset_root,
        config=_config_request(body.config),
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def create_router(service: ValidationService) -> APIRouter:
    """Create run lifecycle routes bound to one application service."""

    router = APIRouter(prefix="/api/runs", tags=["validation"])

    @router.post("")
    def create_run(body: RunCreateBody) -> dict[str, str]:
        try:
            run_id = service.create_run(_run_request(body))
            service.start_run(run_id)
            return {"run_id": run_id, "state": "running"}
        except Exception as exc:
            raise _http_error(exc)

    @router.get("/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return _jsonable(service.get_run(run_id))
        except Exception as exc:
            raise _http_error(exc)

    @router.post("/{run_id}/stop")
    def stop_run(run_id: str) -> dict[str, Any]:
        try:
            service.stop_run(run_id)
            return _jsonable(service.get_run(run_id))
        except Exception as exc:
            raise _http_error(exc)

    @router.get("/{run_id}/summary")
    def get_summary(run_id: str) -> Any:
        try:
            summary = service.get_run_summary(run_id)
            return summary.to_dict() if summary is not None else None
        except Exception as exc:
            raise _http_error(exc)

    @router.get("/{run_id}/results")
    def get_results(run_id: str) -> Any:
        try:
            return _jsonable(service.get_run_validation(run_id))
        except Exception as exc:
            raise _http_error(exc)

    return router


__all__ = ["RunCreateBody", "ValidationConfigBody", "create_router"]
