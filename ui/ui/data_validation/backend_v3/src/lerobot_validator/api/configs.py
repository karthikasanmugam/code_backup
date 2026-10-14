from __future__ import annotations

"""Configuration-related HTTP routes for the validator API.

These routes mirror the existing public API contract exactly:

    GET  /api/checks
    POST /api/config/resolve

The application service remains responsible for configuration discovery and
resolution. This module only adapts HTTP requests to that service.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from ..application.schemas import ValidationConfigRequest
from ..application.service import ValidationService


class ConfigResolveBody(BaseModel):
    """Request body for resolving a UI-selected validation configuration."""

    model_config = ConfigDict(extra="forbid")

    dataset_root: str = Field(min_length=1)
    profile: str = Field(default="default", min_length=1)
    overrides: dict[str, Any] = Field(default_factory=dict)
    enabled_checks: dict[str, bool] = Field(default_factory=dict)


def _config_request(body: ConfigResolveBody) -> ValidationConfigRequest:
    return ValidationConfigRequest(
        dataset_root=body.dataset_root,
        profile=body.profile,
        overrides=body.overrides,
        enabled_checks=body.enabled_checks,
    )


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def create_router(service: ValidationService) -> APIRouter:
    """Create configuration routes using the existing public paths."""
    router = APIRouter(prefix="/api", tags=["config"])

    @router.get("/checks")
    def get_checks(
        dataset_root: str | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        try:
            definitions = service.get_check_definitions(dataset_root)
            return [
                definition.to_dict()
                if hasattr(definition, "to_dict")
                else {
                    "id": definition.id,
                    "label": definition.label,
                    "phase": definition.phase,
                    "severity": definition.severity,
                    "enabled_by_default": definition.enabled_by_default,
                    "description": definition.description,
                    "controls": [
                        {
                            "id": control.id,
                            "label": control.label,
                            "control_type": control.control_type,
                            "config_path": list(control.config_path),
                            "default": control.default,
                            "minimum": control.minimum,
                            "maximum": control.maximum,
                            "step": control.step,
                            "unit": control.unit,
                            "description": control.description,
                            "source": control.source,
                            "binding_status": control.binding_status,
                            "options": list(control.options),
                        }
                        for control in definition.controls
                    ],
                }
                for definition in definitions
            ]
        except Exception as exc:
            raise _http_error(exc)

    @router.post("/config/resolve")
    def resolve_config(body: ConfigResolveBody) -> dict[str, Any]:
        try:
            response = service.resolve_config(_config_request(body))
            return response.to_dict()
        except Exception as exc:
            raise _http_error(exc)

    return router


__all__ = ["ConfigResolveBody", "create_router"]
