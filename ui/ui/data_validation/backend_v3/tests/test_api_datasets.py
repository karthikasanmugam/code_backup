from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot_validator.api.datasets import DatasetInspectBody, create_router
from lerobot_validator.application.schemas import DatasetInspection


class FakeService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def inspect_dataset(self, dataset_root: str) -> DatasetInspection:
        self.calls.append(dataset_root)
        return DatasetInspection(
            dataset_root=str(Path(dataset_root).resolve()),
            fps=30.0,
            total_episodes=12,
            total_frames=6000,
            total_tasks=2,
            state_dim=6,
            action_dim=8,
            action_names=("joint_1", "joint_2", "gripper"),
        )


def _client(service: FakeService) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(service))
    return TestClient(app)


def test_dataset_inspect_body_forbids_unknown_fields() -> None:
    try:
        DatasetInspectBody(dataset_root="/tmp/data", extra="bad")
    except Exception:
        pass
    else:
        raise AssertionError("Unknown request fields must be rejected")


def test_inspect_dataset_returns_service_payload() -> None:
    service = FakeService()
    client = _client(service)

    response = client.post(
        "/api/datasets/inspect",
        json={"dataset_root": "/tmp/example"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fps"] == 30.0
    assert payload["total_episodes"] == 12
    assert payload["action_dim"] == 8
    assert service.calls == ["/tmp/example"]


def test_inspect_dataset_maps_missing_dataset_to_404() -> None:
    class MissingService(FakeService):
        def inspect_dataset(self, dataset_root: str) -> DatasetInspection:
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    client = _client(MissingService())
    response = client.post(
        "/api/datasets/inspect",
        json={"dataset_root": "/missing"},
    )

    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


def test_inspect_dataset_maps_invalid_request_to_422() -> None:
    client = _client(FakeService())
    response = client.post("/api/datasets/inspect", json={})

    assert response.status_code == 422


def test_inspect_dataset_maps_service_value_error_to_400() -> None:
    class InvalidService(FakeService):
        def inspect_dataset(self, dataset_root: str) -> DatasetInspection:
            raise ValueError("invalid dataset")

    client = _client(InvalidService())
    response = client.post(
        "/api/datasets/inspect",
        json={"dataset_root": "/bad"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid dataset"
