from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot_validator.api.results import create_router


@dataclass
class FakeResult:
    check: str
    severity: str
    message: str
    frame_range: tuple[int, int] | None = None


@dataclass
class FakeEpisode:
    episode_index: int
    score: float
    classification: str
    results: tuple[FakeResult, ...]
    frame_count: int | None = None
    duration_seconds: float | None = None
    instruction: str | None = None


@dataclass
class FakeValidation:
    episodes: tuple[FakeEpisode, ...]


class FakeService:
    def __init__(self) -> None:
        self.validation = FakeValidation(
            episodes=(
                FakeEpisode(
                    episode_index=0,
                    score=0.95,
                    classification="pass",
                    frame_count=120,
                    duration_seconds=4.0,
                    instruction="pick the brown object",
                    results=(),
                ),
                FakeEpisode(
                    episode_index=1,
                    score=0.40,
                    classification="reject",
                    results=(
                        FakeResult(
                            check="video.frame_count",
                            severity="HARD",
                            message="Video has one missing frame",
                            frame_range=(10, 10),
                        ),
                        FakeResult(
                            check="actions.smoothness",
                            severity="SOFT",
                            message="High jerk detected",
                            frame_range=(20, 30),
                        ),
                    ),
                ),
            )
        )

    def get_run_validation(self, run_id: str):
        if run_id != "run-123":
            raise KeyError(f"Unknown validation run: {run_id}")
        return self.validation


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(create_router(FakeService()))
    return TestClient(app)


def test_list_episodes_returns_explorer_rows() -> None:
    response = _client().get("/api/runs/run-123/episodes")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["episodes"][1]["hard_count"] == 1
    assert payload["episodes"][1]["soft_count"] == 1


def test_list_episodes_filters_status_and_hard_only() -> None:
    response = _client().get(
        "/api/runs/run-123/episodes",
        params={"status": "reject", "hard_only": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["episodes"][0]["episode_index"] == 1


def test_list_episodes_supports_pagination() -> None:
    response = _client().get(
        "/api/runs/run-123/episodes",
        params={"offset": 1, "limit": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert len(payload["episodes"]) == 1
    assert payload["episodes"][0]["episode_index"] == 1


def test_get_episode_returns_detail_and_results() -> None:
    response = _client().get("/api/runs/run-123/episodes/1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["episode_index"] == 1
    assert payload["score"] == 0.4
    assert len(payload["results"]) == 2
    assert payload["results"][0]["frame_range"] == [10, 10]


def test_get_episode_evidence_returns_frame_references() -> None:
    response = _client().get("/api/runs/run-123/episodes/1/evidence")
    assert response.status_code == 200
    payload = response.json()
    assert payload["episode_index"] == 1
    assert len(payload["evidence"]) == 2
    assert payload["evidence"][0]["kind"] == "check_result"
    assert payload["evidence"][1]["frame_range"] == [20, 30]


def test_unknown_run_returns_404() -> None:
    response = _client().get("/api/runs/missing/episodes")
    assert response.status_code == 404


def test_missing_episode_returns_404() -> None:
    response = _client().get("/api/runs/run-123/episodes/99")
    assert response.status_code == 404
