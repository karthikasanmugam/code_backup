from __future__ import annotations

import pytest

from lerobot_validator.control_registry import (
    CHECK_REGISTRY,
    PHASE_ORDER,
    get_check_definition,
    get_checks_by_phase,
    get_control,
)


def test_registry_has_stable_unique_ids():
    ids = [definition.id for definition in CHECK_REGISTRY]
    assert ids
    assert len(ids) == len(set(ids))


def test_phase_order_is_stable():
    assert PHASE_ORDER == (
        "structural",
        "video",
        "actions",
        "episode",
        "dataset_quality",
    )


def test_video_frame_count_control_maps_to_config():
    control = get_control(
        "video.frame_count",
        "tolerance_frames",
    )

    assert control.control_type == "integer"
    assert control.config_path == (
        "video",
        "frame_count_tolerance",
    )
    assert control.minimum == 0


def test_dynamic_camera_control_is_dataset_driven():
    control = get_control(
        "video.multiview_sync",
        "reference_camera",
    )

    assert control.control_type == "camera_select"
    assert control.source == "dataset"


def test_action_features_are_dataset_driven():
    control = get_control(
        "actions.smoothness",
        "exclude_dimensions",
    )

    assert control.control_type == "feature_multi_select"
    assert control.source == "dataset"


def test_gripper_dimension_is_feature_selected():
    control = get_control(
        "actions.gripper",
        "dimension",
    )

    assert control.control_type == "feature_select"
    assert control.source == "dataset"
    assert control.default is None


def test_checks_can_be_grouped_by_phase():
    video_checks = get_checks_by_phase("video")
    assert video_checks
    assert all(check.phase == "video" for check in video_checks)
    assert {check.id for check in video_checks} >= {
        "video.frame_count",
        "video.timestamps",
        "video.frozen_video",
        "video.multiview_sync",
    }


def test_unknown_check_raises():
    with pytest.raises(KeyError, match="Unknown validation check"):
        get_check_definition("does.not.exist")


def test_unknown_control_raises():
    with pytest.raises(KeyError, match="Unknown control"):
        get_control("video.frame_count", "does_not_exist")
