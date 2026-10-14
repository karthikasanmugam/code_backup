from __future__ import annotations

"""FastAPI transport layer for the validator application service."""

from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict

from ..application.schemas import RunCreateRequest, ValidationConfigRequest
from ..application.service import ValidationService
from ..deletion import delete_episodes_in_place
from .results import create_router as create_results_router
from fastapi.middleware.cors import CORSMiddleware

class DatasetInspectBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_root: str = Field(min_length=1)


class EpisodeDeleteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_root: str = Field(min_length=1)
    episodes: list[int] = Field(min_length=1)


class ConfigResolveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_root: str = Field(min_length=1)
    profile: str = Field(default="default", min_length=1)
    overrides: dict[str, Any] = Field(default_factory=dict)
    enabled_checks: dict[str, bool] = Field(default_factory=dict)


class RunConfigBody(ConfigResolveBody):
    pass


class RunCreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_root: str = Field(min_length=1)
    config: RunConfigBody


def _config_request(body: ConfigResolveBody) -> ValidationConfigRequest:
    return ValidationConfigRequest(
        dataset_root=body.dataset_root,
        profile=body.profile,
        overrides=body.overrides,
        enabled_checks=body.enabled_checks,
    )


def _run_request(body: RunCreateBody) -> RunCreateRequest:
    config = _config_request(body.config)
    return RunCreateRequest(dataset_root=body.dataset_root, config=config)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return {key: _jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
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


def create_app(service: ValidationService | None = None) -> FastAPI:
    """Create the HTTP application with an injectable service instance."""
    validator_service = service or ValidationService()
    app = FastAPI(
        title="Modular Dataset Validator API",
        version="0.1.0",
    )
    app.state.validation_service = validator_service
    app.include_router(create_results_router(validator_service))

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/datasets/inspect")
    def inspect_dataset(body: DatasetInspectBody) -> dict[str, Any]:
        try:
            return validator_service.inspect_dataset(body.dataset_root).to_dict()
        except Exception as exc:
            raise _http_error(exc)

    @app.post("/api/datasets/delete-episodes")
    def delete_episodes(body: EpisodeDeleteBody) -> dict[str, Any]:
        """Delete one or more v3 episodes, including their shared-file segments."""
        try:
            return delete_episodes_in_place(body.dataset_root, body.episodes)
        except Exception as exc:
            raise _http_error(exc)

    @app.get("/api/checks")
    def get_checks(
        dataset_root: str | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        try:
            return [item.to_dict() if hasattr(item, "to_dict") else _jsonable(item)
                    for item in validator_service.get_check_definitions(dataset_root)]
        except Exception as exc:
            raise _http_error(exc)

    @app.post("/api/config/resolve")
    def resolve_config(body: ConfigResolveBody) -> dict[str, Any]:
        try:
            response = validator_service.resolve_config(_config_request(body))
            return response.to_dict()
        except Exception as exc:
            raise _http_error(exc)

    @app.post("/api/runs")
    def create_run(body: RunCreateBody) -> dict[str, str]:
        try:
            run_id = validator_service.create_run(_run_request(body))
            validator_service.start_run(run_id)
            return {"run_id": run_id, "state": "running"}
        except Exception as exc:
            raise _http_error(exc)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return _jsonable(validator_service.get_run(run_id))
        except Exception as exc:
            raise _http_error(exc)

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> dict[str, Any]:
        try:
            validator_service.stop_run(run_id)
            return _jsonable(validator_service.get_run(run_id))
        except Exception as exc:
            raise _http_error(exc)

    @app.get("/api/runs/{run_id}/summary")
    def get_summary(run_id: str) -> dict[str, Any] | None:
        try:
            summary = validator_service.get_run_summary(run_id)
            return summary.to_dict() if summary is not None else None
        except Exception as exc:
            raise _http_error(exc)

    @app.get("/api/runs/{run_id}/results")
    def get_results(run_id: str) -> Any:
        try:
            return _jsonable(validator_service.get_run_validation(run_id))
        except Exception as exc:
            raise _http_error(exc)

    allowed_origins = [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ]
    daksha_frontend_origin = os.environ.get("DAKSHA_VALIDATION_FRONTEND_ORIGIN")
    if daksha_frontend_origin and daksha_frontend_origin not in allowed_origins:
        allowed_origins.append(daksha_frontend_origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()

__all__ = ["app", "create_app"]
