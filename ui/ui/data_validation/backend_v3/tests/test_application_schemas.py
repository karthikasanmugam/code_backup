from __future__ import annotations

import pytest

from lerobot_validator.application.schemas import (
    CameraSchema,
    DatasetInspection,
    EpisodeResultSummary,
    EvidenceReference,
    ProgressEvent,
    ResultSummary,
    RunCreateRequest,
    RunState,
    ValidationConfigRequest,
    ValidationConfigResponse,
)


def test_dataset_inspection_round_trip_and_dynamic_cameras() -> None:
    original = DatasetInspection(
        dataset_root="/tmp/example",
        fps=25,
        total_episodes=3,
        total_frames=120,
        total_tasks=2,
        state_dim=7,
        action_dim=7,
        state_names=("joint_1", "joint_2"),
        action_names=("joint_1", "gripper"),
        cameras=(
            CameraSchema(key="cam_a", width=640, height=480),
            CameraSchema(key="cam_b", width=320, height=240),
            CameraSchema(key="cam_c", width=160, height=120),
        ),
        tasks=("pick", "place"),
    )

    payload = original.to_dict()
    restored = DatasetInspection.from_mapping(payload)

    assert restored == original
    assert len(restored.cameras) == 3
    assert [camera.key for camera in restored.cameras] == [
        "cam_a",
        "cam_b",
        "cam_c",
    ]


def test_dataset_inspection_rejects_invalid_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        DatasetInspection(
            dataset_root="/tmp/example",
            fps=0,
            total_episodes=1,
            total_frames=10,
            total_tasks=1,
            state_dim=2,
            action_dim=2,
        )


def test_validation_config_request_round_trip() -> None:
    request = ValidationConfigRequest(
        dataset_root="/tmp/data",
        profile="strict",
        overrides={
            "video.timestamp_tolerance_s": 0.05,
            "actions.smoothness.jerk_z_threshold": 4.0,
        },
        enabled_checks={
            "video.frame_count": True,
            "video.image_quality": False,
        },
    )

    restored = ValidationConfigRequest.from_mapping(request.to_dict())
    assert restored == request


def test_validation_config_response_reports_validity() -> None:
    response = ValidationConfigResponse(
        dataset_root="/tmp/data",
        profile="default",
        resolved={"video": {"enabled": True}},
        warnings=("timestamp tolerance is auto-derived",),
    )

    assert response.valid is True
    assert response.to_dict()["valid"] is True
    assert response.to_dict()["warnings"] == [
        "timestamp tolerance is auto-derived"
    ]

    invalid = ValidationConfigResponse(
        dataset_root="/tmp/data",
        profile="default",
        resolved={},
        errors=("unknown check",),
    )
    assert invalid.valid is False
    assert invalid.to_dict()["valid"] is False


def test_run_create_requires_matching_dataset_root() -> None:
    config = ValidationConfigRequest(dataset_root="/tmp/a")

    with pytest.raises(ValueError, match="must match"):
        RunCreateRequest(
            dataset_root="/tmp/b",
            config=config,
        )


def test_progress_event_is_json_friendly() -> None:
    event = ProgressEvent(
        event="check_finished",
        state=RunState.RUNNING,
        run_id="run-123",
        completed_episodes=4,
        total_episodes=10,
        current_episode=4,
        phase="video",
        current_check="video.frame_count",
        current_camera="observation.images.cam_b",
        message="done",
    )

    payload = event.to_dict()

    assert payload["state"] == "running"
    assert payload["current_check"] == "video.frame_count"
    assert payload["current_camera"] == "observation.images.cam_b"


def test_evidence_reference_validates_frame_range() -> None:
    evidence = EvidenceReference(
        kind="video_interval",
        camera="cam_a",
        frame_range=(10, 20),
        location="runs/x/results.json",
    )

    assert evidence.to_dict()["frame_range"] == [10, 20]

    with pytest.raises(ValueError, match="frame_range"):
        EvidenceReference(kind="video_interval", frame_range=(20, 10))


def test_episode_result_summary_accepts_score_and_counts() -> None:
    result = EpisodeResultSummary(
        episode_index=2,
        score=0.82,
        status="review",
        hard_count=1,
        soft_count=3,
        frame_count=500,
        duration_seconds=20.0,
        instruction="pick the object",
    )

    assert result.to_dict()["score"] == 0.82
    assert result.to_dict()["soft_count"] == 3

    with pytest.raises(ValueError, match="between 0 and 1"):
        EpisodeResultSummary(
            episode_index=0,
            score=1.2,
            status="reject",
        )


def test_result_summary_prevents_overallocated_status_counts() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ResultSummary(
            total_episodes=2,
            pass_count=1,
            review_count=1,
            reject_count=1,
        )
