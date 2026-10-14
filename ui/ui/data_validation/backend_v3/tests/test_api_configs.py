from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot_validator.api.configs import ConfigResolveBody, create_router
from lerobot_validator.application.schemas import ValidationConfigResponse


class FakeCheck:
    def __init__(self) -> None:
        self.id = "video.frame_count"
        self.label = "Video frame count"
        self.phase = "video"
        self.severity = "HARD"
        self.enabled_by_default = True
        self.description = "Checks frame counts"
        self.controls = ()


class FakeService:
    def __init__(self) -> None:
        self.dataset_roots: list[str | None] = []
        self.requests = []

    def get_check_definitions(self, dataset_root=None):
        self.dataset_roots.append(dataset_root)
        return (FakeCheck(),)

    def resolve_config(self, request):
        self.requests.append(request)
        return ValidationConfigResponse(
            dataset_root=request.dataset_root,
            profile=request.profile,
            resolved=request.overrides,
            warnings=(),
            errors=(),
        )


def _client(service: FakeService) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(service))
    return TestClient(app)


def test_get_checks_without_dataset_root() -> None:
    service = FakeService()
    response = _client(service).get("/api/checks")

    assert response.status_code == 200
    assert response.json()[0]["id"] == "video.frame_count"
    assert service.dataset_roots == [None]


def test_get_checks_for_dataset_passes_dataset_root() -> None:
    service = FakeService()
    response = _client(service).get(
        "/api/checks",
        params={"dataset_root": "/tmp/example-dataset"},
    )

    assert response.status_code == 200
    assert service.dataset_roots == ["/tmp/example-dataset"]


def test_resolve_config_builds_application_request() -> None:
    service = FakeService()
    response = _client(service).post(
        "/api/config/resolve",
        json={
            "dataset_root": "/tmp/example-dataset",
            "profile": "strict",
            "overrides": {
                "actions": {"smoothness": {"jerk_z_threshold": 9.5}}
            },
            "enabled_checks": {"actions.smoothness": True},
        },
    )

    assert response.status_code == 200
    request = service.requests[0]
    assert request.dataset_root == "/tmp/example-dataset"
    assert request.profile == "strict"
    assert request.overrides["actions"]["smoothness"]["jerk_z_threshold"] == 9.5
    assert request.enabled_checks["actions.smoothness"] is True


def test_resolve_config_rejects_missing_dataset_root() -> None:
    response = _client(FakeService()).post(
        "/api/config/resolve",
        json={"profile": "default"},
    )
    assert response.status_code == 422


def test_config_body_rejects_unknown_fields() -> None:
    try:
        ConfigResolveBody(dataset_root="/tmp/dataset", unknown=True)
    except Exception as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("Unknown field should be rejected")


def test_old_config_checks_path_is_not_registered() -> None:
    response = _client(FakeService()).get("/api/config/checks")
    assert response.status_code == 404
