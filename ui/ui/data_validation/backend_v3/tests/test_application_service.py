from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from lerobot_validator.application import service as service_module
from lerobot_validator.application.schemas import (
    DatasetInspection,
    RunCreateRequest,
    RunState,
    ValidationConfigRequest,
)
from lerobot_validator.application.service import ValidationService


@dataclass
class FakeResult:
    check: str
    severity: str


@dataclass
class FakeEpisode:
    classification: str
    results: list[FakeResult]


@dataclass
class FakeValidation:
    episodes: list[FakeEpisode]


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    return root


def _fake_info() -> SimpleNamespace:
    def feature(names: list[str]) -> SimpleNamespace:
        return SimpleNamespace(info={"names": names})

    cameras = {
        "cam_a": SimpleNamespace(key="cam_a", width=320, height=240),
        "cam_b": SimpleNamespace(key="cam_b", width=640, height=480),
    }
    return SimpleNamespace(
        fps=30.0,
        state_dim=3,
        action_dim=4,
        features={
            "observation.state": feature(["x", "y", "z"]),
            "action": feature(["a0", "a1", "left_gripper_joint", "a3"]),
        },
        cameras=cameras,
        raw={
            "fps": 30,
            "total_episodes": 2,
            "total_frames": 100,
            "total_tasks": 1,
            "camera_keys": ["cam_a", "cam_b"],
            "repo_id": "test/repo",
        },
    )


def _make_profile(path: Path) -> None:
    path.write_text(
        """
dataset:
  enabled: true
structural:
  min_episode_length: 20
video:
  image_quality:
    sample_every: 15
actions:
  smoothness:
    jerk_z_threshold: 6.0
""",
        encoding="utf-8",
    )


@pytest.fixture
def service_dependencies(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(service_module, "load_dataset_info", lambda root: _fake_info())
    monkeypatch.setattr(
        service_module,
        "load_task_map",
        lambda root: {0: {"instruction": "pick place"}},
    )


def test_inspect_dataset_is_metadata_driven(
    tmp_path: Path,
    service_dependencies,
) -> None:
    service = ValidationService(config_dir=tmp_path)
    inspection = service.inspect_dataset(_dataset_root(tmp_path))

    assert isinstance(inspection, DatasetInspection)
    assert inspection.fps == 30
    assert inspection.state_names == ("x", "y", "z")
    assert inspection.action_names == (
        "a0",
        "a1",
        "left_gripper_joint",
        "a3",
    )
    assert [camera.key for camera in inspection.cameras] == ["cam_a", "cam_b"]
    assert inspection.cameras[1].width == 640
    assert inspection.tasks == ("pick place",)


def test_get_check_definitions_populates_dynamic_selectors(
    tmp_path: Path,
    service_dependencies,
) -> None:
    service = ValidationService(config_dir=tmp_path)
    checks = service.get_check_definitions(_dataset_root(tmp_path))

    smoothness = next(c for c in checks if c.id == "actions.smoothness")
    excluded = next(c for c in smoothness.controls if c.id == "exclude_dimensions")
    assert excluded.options == ("a0", "a1", "left_gripper_joint", "a3")

    sync = next(c for c in checks if c.id == "video.multiview_sync")
    reference = next(c for c in sync.controls if c.id == "reference_camera")
    assert reference.options == ("auto", "cam_a", "cam_b")


def test_resolve_config_deep_merges_overrides_and_injects_dataset_root(
    tmp_path: Path,
    service_dependencies,
) -> None:
    root = _dataset_root(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _make_profile(config_dir / "default.yml")

    service = ValidationService(config_dir=config_dir)
    response = service.resolve_config(
        ValidationConfigRequest(
            dataset_root=str(root),
            overrides={"actions": {"smoothness": {"jerk_z_threshold": 9.5}}},
        )
    )

    assert response.valid
    assert response.resolved["dataset"]["data_root"] == str(root.resolve())
    assert response.resolved["actions"]["smoothness"]["jerk_z_threshold"] == 9.5
    assert response.resolved["actions"]["smoothness"] is not None


def test_resolve_config_rejects_unknown_reference_camera(
    tmp_path: Path,
    service_dependencies,
) -> None:
    root = _dataset_root(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _make_profile(config_dir / "default.yml")

    service = ValidationService(config_dir=config_dir)
    response = service.resolve_config(
        ValidationConfigRequest(
            dataset_root=str(root),
            overrides={
                "video": {
                    "multiview_sync": {
                        "reference_camera": "missing_cam",
                    }
                }
            },
        )
    )

    assert not response.valid
    assert any("Reference camera" in message for message in response.errors)


def test_run_sync_persists_resolved_config_and_summary(
    tmp_path: Path,
    service_dependencies,
) -> None:
    root = _dataset_root(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _make_profile(config_dir / "default.yml")
    runs_dir = tmp_path / "runs"

    def fake_engine(*, data_root, config):
        return FakeValidation(
            [
                FakeEpisode("PASS", []),
                FakeEpisode(
                    "REJECT",
                    [
                        FakeResult("video.frame_count", "HARD"),
                        FakeResult("actions.smoothness", "SOFT"),
                    ],
                ),
            ]
        )

    service = ValidationService(
        config_dir=config_dir,
        runs_dir=runs_dir,
        engine_validate_dataset=fake_engine,
    )
    request = RunCreateRequest(
        dataset_root=str(root),
        config=ValidationConfigRequest(dataset_root=str(root)),
    )
    run_id = service.run_sync(request)

    run = service.get_run(run_id)
    assert run["state"] == RunState.COMPLETED.value
    assert run["summary"]["total_episodes"] == 2
    assert run["summary"]["pass_count"] == 1
    assert run["summary"]["reject_count"] == 1
    assert run["summary"]["hard_failures"] == 1
    assert run["summary"]["soft_warnings"] == 1
    assert (runs_dir / run_id / "config.json").exists()
    assert (runs_dir / run_id / "results.json").exists()


def test_stop_running_run_cancels_engine_and_allows_fresh_run(
    tmp_path: Path,
    service_dependencies,
) -> None:
    root = _dataset_root(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _make_profile(config_dir / "default.yml")

    engine_started = threading.Event()
    calls = 0

    def engine_validate_dataset(*, data_root, config, cancel_check):
        nonlocal calls
        calls += 1
        if calls == 1:
            engine_started.set()
            while True:
                cancel_check()
                time.sleep(0.005)
        return FakeValidation(episodes=[])

    service = ValidationService(
        config_dir=config_dir,
        runs_dir=tmp_path / "runs",
        engine_validate_dataset=engine_validate_dataset,
    )
    request = RunCreateRequest(
        dataset_root=str(root),
        config=ValidationConfigRequest(dataset_root=str(root)),
    )

    first_run_id = service.create_run(request)
    service.start_run(first_run_id)
    assert engine_started.wait(timeout=2)
    service.stop_run(first_run_id)

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if service.get_run(first_run_id)["state"] == RunState.CANCELLED.value:
            break
        time.sleep(0.005)
    assert service.get_run(first_run_id)["state"] == RunState.CANCELLED.value

    second_run_id = service.create_run(request)
    service.start_run(second_run_id)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if service.get_run(second_run_id)["state"] == RunState.COMPLETED.value:
            break
        time.sleep(0.005)
    assert service.get_run(second_run_id)["state"] == RunState.COMPLETED.value


def test_stop_queued_run_marks_it_cancelled(
    tmp_path: Path,
    service_dependencies,
) -> None:
    root = _dataset_root(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _make_profile(config_dir / "default.yml")

    service = ValidationService(config_dir=config_dir, runs_dir=tmp_path / "runs")
    request = RunCreateRequest(
        dataset_root=str(root),
        config=ValidationConfigRequest(dataset_root=str(root)),
    )
    run_id = service.create_run(request)
    service.stop_run(run_id)

    assert service.get_run(run_id)["state"] == RunState.CANCELLED.value


def test_resolve_config_normalizes_gripper_feature_name_to_engine_index(
    tmp_path: Path,
    service_dependencies,
) -> None:
    root = _dataset_root(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    _make_profile(config_dir / "default.yml")

    service = ValidationService(config_dir=config_dir)
    response = service.resolve_config(
        ValidationConfigRequest(
            dataset_root=str(root),
            overrides={
                "actions": {
                    "gripper": {
                        "dimension": "left_gripper_joint",
                    }
                }
            },
        )
    )

    assert response.valid
    assert response.resolved["actions"]["gripper"]["dimension"] == 2


def test_default_service_paths_are_relative_to_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    service = ValidationService()
    project_root = Path(service_module.__file__).resolve().parents[3]

    assert service.config_dir == project_root / "configs"
    assert service.runs_dir == project_root / "runs"
