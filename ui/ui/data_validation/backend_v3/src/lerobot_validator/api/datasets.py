from __future__ import annotations

"""Dataset-related HTTP routes for the validator API."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..application.service import ValidationService


class DatasetInspectBody(BaseModel):
    """Request body for runtime dataset inspection."""

    model_config = ConfigDict(extra="forbid")
    dataset_root: str = Field(min_length=1)


def _http_error(exc: Exception) -> HTTPException:
    """Translate service-layer exceptions into HTTP responses."""
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def create_router(service: ValidationService) -> APIRouter:
    """Create dataset routes bound to an application service instance."""
    router = APIRouter(prefix="/api/datasets", tags=["datasets"])

    @router.post("/inspect")
    def inspect_dataset(body: DatasetInspectBody) -> dict[str, Any]:
        try:
            inspection = service.inspect_dataset(body.dataset_root)
            return inspection.to_dict()
        except Exception as exc:
            raise _http_error(exc)

    return router


__all__ = ["DatasetInspectBody", "create_router"]
