from __future__ import annotations

from pathlib import Path
import importlib

from fastapi.testclient import TestClient

from lerobot_validator.application.schemas import DatasetInspection, ResultSummary, RunState
from lerobot_validator.api.app import create_app


class FakeService:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.run_id = "run-123"
        self.started = False
        self.stopped = False

    def inspect_dataset(self, root):
        return DatasetInspection(
            dataset_root=str(Path(root).resolve()),
            fps=25.0,
            total_episodes=3,
            total_frames=30,
            total_tasks=1,
            state_dim=2,
            action_dim=2,
            state_names=("s0", "s1"),
            action_names=("a0", "a1"),
        )

    def get_check_definitions(self, dataset_root=None):
        class D:
            def to_dict(self):
                return {"id": "video.frame_count", "controls": []}
        return (D(),)

    def resolve_config(self, request):
        class R:
            def to_dict(self):
                return {"dataset_root": request.dataset_root, "valid": True, "resolved": {}, "warnings": [], "errors": []}
        return R()

    def create_run(self, request):
        return self.run_id

    def start_run(self, run_id):
        self.started = True

    def stop_run(self, run_id):
        self.stopped = True

    def get_run(self, run_id):
        return {"run_id": run_id, "state": "cancelled" if self.stopped else "running"}

    def get_run_summary(self, run_id):
        return ResultSummary(total_episodes=3, pass_count=3)

    def get_run_validation(self, run_id):
        return {"episodes": []}


def test_health() -> None:
    client = TestClient(create_app(FakeService(Path("/tmp"))))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_inspect_endpoint_transports_dataset_metadata(tmp_path: Path) -> None:
    client = TestClient(create_app(FakeService(tmp_path)))
    root = tmp_path / "dataset"
    root.mkdir()
    response = client.post("/api/datasets/inspect", json={"dataset_root": str(root)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["fps"] == 25.0
    assert payload["action_names"] == ["a0", "a1"]


def test_checks_endpoint_supports_dynamic_dataset_query(tmp_path: Path) -> None:
    client = TestClient(create_app(FakeService(tmp_path)))
    response = client.get("/api/checks", params={"dataset_root": str(tmp_path)})
    assert response.status_code == 200
    assert response.json()[0]["id"] == "video.frame_count"


def test_config_resolve_endpoint(tmp_path: Path) -> None:
    client = TestClient(create_app(FakeService(tmp_path)))
    root = tmp_path / "dataset"
    root.mkdir()
    response = client.post(
        "/api/config/resolve",
        json={"dataset_root": str(root), "profile": "default", "overrides": {"video": {"enabled": False}}},
    )
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_create_run_starts_run(tmp_path: Path) -> None:
    service = FakeService(tmp_path)
    client = TestClient(create_app(service))
    root = tmp_path / "dataset"
    root.mkdir()
    body = {
        "dataset_root": str(root),
        "config": {"dataset_root": str(root), "profile": "default"},
    }
    response = client.post("/api/runs", json=body)
    assert response.status_code == 200
    assert response.json() == {"run_id": "run-123", "state": "running"}
    assert service.started is True


def test_run_stop_and_results_endpoints(tmp_path: Path) -> None:
    service = FakeService(tmp_path)
    client = TestClient(create_app(service))
    response = client.post("/api/runs/run-123/stop")
    assert response.status_code == 200
    assert response.json()["state"] == "cancelled"

    summary = client.get("/api/runs/run-123/summary")
    assert summary.status_code == 200
    assert summary.json()["pass_count"] == 3

    results = client.get("/api/runs/run-123/results")
    assert results.status_code == 200
    assert results.json() == {"episodes": []}


def test_missing_run_maps_to_not_found(tmp_path: Path) -> None:
    class Broken(FakeService):
        def get_run(self, run_id):
            raise KeyError(f"Unknown validation run: {run_id}")

    client = TestClient(create_app(Broken(tmp_path)))
    response = client.get("/api/runs/missing")
    assert response.status_code == 404



def test_dynamic_daksha_frontend_origin_passes_json_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    origin = "http://localhost:8124"
    monkeypatch.setenv("DAKSHA_VALIDATION_FRONTEND_ORIGIN", origin)
    client = TestClient(create_app(FakeService(tmp_path)))

    response = client.options(
        "/api/datasets/inspect",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "POST" in response.headers["access-control-allow-methods"]



def test_delete_endpoint_forwards_multiple_episode_ids_as_one_batch(
    tmp_path: Path, monkeypatch
) -> None:
    api_module = importlib.import_module("lerobot_validator.api.app")
    calls = []

    def fake_delete(root, episode_ids):
        calls.append((root, episode_ids))
        return {"deleted": sorted(set(episode_ids)), "dataset_root": root, "remaining_episodes": 2}

    monkeypatch.setattr(api_module, "delete_episodes_in_place", fake_delete)
    client = TestClient(create_app(FakeService(tmp_path)))
    response = client.post(
        "/api/datasets/delete-episodes",
        json={"dataset_root": str(tmp_path), "episodes": [3, 1, 3]},
    )

    assert response.status_code == 200
    assert response.json()["deleted"] == [1, 3]
    assert calls == [(str(tmp_path), [3, 1, 3])]
