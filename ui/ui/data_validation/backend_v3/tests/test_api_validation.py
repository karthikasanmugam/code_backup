from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot_validator.api.validation import create_router
from lerobot_validator.application.schemas import ResultSummary, RunState


class FakeService:
    def __init__(self) -> None:
        self.create_requests = []
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.run_payloads = {
            "run-1": {
                "run_id": "run-1",
                "state": RunState.RUNNING,
            }
        }

    def create_run(self, request):
        self.create_requests.append(request)
        return "run-1"

    def start_run(self, run_id: str) -> None:
        self.started.append(run_id)

    def get_run(self, run_id: str):
        return self.run_payloads[run_id]

    def stop_run(self, run_id: str) -> None:
        self.stopped.append(run_id)

    def get_run_summary(self, run_id: str):
        return ResultSummary(total_episodes=2, pass_count=1, review_count=1)

    def get_run_validation(self, run_id: str):
        return {"episode_results": [], "dataset_results": []}


def _client(service: FakeService) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(service))
    return TestClient(app)


def _payload() -> dict:
    return {
        "dataset_root": "/tmp/dataset",
        "config": {
            "dataset_root": "/tmp/dataset",
            "profile": "default",
            "overrides": {"actions": {"smoothness": {"jerk_z_threshold": 8.0}}},
            "enabled_checks": {"actions.smoothness": True},
        },
    }


def test_create_run_builds_request_and_starts_service() -> None:
    service = FakeService()
    response = _client(service).post("/api/runs", json=_payload())

    assert response.status_code == 200
    assert response.json() == {"run_id": "run-1", "state": "running"}
    assert service.started == ["run-1"]
    request = service.create_requests[0]
    assert request.dataset_root == "/tmp/dataset"
    assert request.config.overrides["actions"]["smoothness"]["jerk_z_threshold"] == 8.0


def test_create_run_rejects_dataset_mismatch_before_service() -> None:
    service = FakeService()
    payload = _payload()
    payload["config"]["dataset_root"] = "/tmp/other"

    response = _client(service).post("/api/runs", json=payload)

    assert response.status_code == 400
    assert service.create_requests == []
    assert service.started == []


def test_get_run_returns_jsonable_state() -> None:
    service = FakeService()
    response = _client(service).get("/api/runs/run-1")

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-1"
    assert response.json()["state"] == "running"


def test_stop_run_delegates_to_service() -> None:
    service = FakeService()
    response = _client(service).post("/api/runs/run-1/stop")

    assert response.status_code == 200
    assert service.stopped == ["run-1"]
    assert response.json()["run_id"] == "run-1"


def test_get_summary_returns_summary_dict() -> None:
    response = _client(FakeService()).get("/api/runs/run-1/summary")

    assert response.status_code == 200
    assert response.json()["total_episodes"] == 2
    assert response.json()["pass_count"] == 1
    assert response.json()["review_count"] == 1


def test_get_results_returns_engine_payload() -> None:
    response = _client(FakeService()).get("/api/runs/run-1/results")

    assert response.status_code == 200
    assert response.json() == {"episode_results": [], "dataset_results": []}


def test_service_error_becomes_client_error() -> None:
    class ErrorService(FakeService):
        def get_run(self, run_id: str):
            raise KeyError(run_id)

    response = _client(ErrorService()).get("/api/runs/missing")

    assert response.status_code == 404
