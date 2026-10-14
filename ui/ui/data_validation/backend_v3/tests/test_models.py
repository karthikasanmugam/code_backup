from pathlib import Path

import numpy as np

from lerobot_validator.models import (
    CameraInfo,
    CheckResult,
    Classification,
    DatasetInfo,
    DatasetStats,
    DatasetValidation,
    EpisodeContext,
    EpisodeData,
    EpisodeMetadata,
    EpisodePaths,
    EpisodeValidation,
    FeatureInfo,
    ValidationConfig,
)


def test_check_result_creation():
    result = CheckResult(
        check="frame_index",
        severity="HARD",
        message="frame_index is not contiguous",
        frame_range=(10, 20),
    )

    assert result.check == "frame_index"
    assert result.severity == "HARD"
    assert result.message == "frame_index is not contiguous"
    assert result.frame_range == (10, 20)


def test_check_result_without_frame_range():
    result = CheckResult(
        check="nan_inf",
        severity="HARD",
        message="NaN detected in action",
    )

    assert result.frame_range is None


def test_feature_info_defaults():
    feature = FeatureInfo(
        name="action",
        dtype="float32",
        shape=(7,),
    )

    assert feature.name == "action"
    assert feature.dtype == "float32"
    assert feature.shape == (7,)
    assert feature.info == {}


def test_camera_info_dimensions():
    camera = CameraInfo(
        key="observation.images.front",
        dtype="uint8",
        shape=(3, 480, 640),
    )

    assert camera.key == "observation.images.front"
    assert camera.height == 480
    assert camera.width == 640


def test_camera_info_invalid_shape_has_no_dimensions():
    camera = CameraInfo(
        key="camera",
        shape=(640,),
    )

    assert camera.height is None
    assert camera.width is None


def test_dataset_info_camera_keys():
    cameras = {
        "observation.images.front": CameraInfo(
            key="observation.images.front",
            shape=(3, 480, 640),
        ),
        "observation.images.wrist": CameraInfo(
            key="observation.images.wrist",
            shape=(3, 224, 224),
        ),
    }

    dataset_info = DatasetInfo(
        fps=30.0,
        state_dim=7,
        action_dim=7,
        cameras=cameras,
    )

    assert dataset_info.fps == 30.0
    assert dataset_info.state_dim == 7
    assert dataset_info.action_dim == 7
    assert dataset_info.camera_keys == [
        "observation.images.front",
        "observation.images.wrist",
    ]


def test_dataset_info_raw_metadata_is_preserved():
    raw = {
        "fps": 30,
        "robot_type": "custom_robot",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [7],
            }
        },
    }

    dataset_info = DatasetInfo(
        fps=30.0,
        state_dim=7,
        action_dim=7,
        raw=raw,
    )

    assert dataset_info.raw == raw
    assert dataset_info.raw["robot_type"] == "custom_robot"


def test_episode_paths():
    paths = EpisodePaths(
        episode_index=12,
        parquet=Path("/dataset/data/episode_0012.parquet"),
        videos={
            "front": Path("/dataset/videos/front/episode_0012.mp4"),
            "wrist": Path("/dataset/videos/wrist/episode_0012.mp4"),
        },
    )

    assert paths.episode_index == 12
    assert paths.parquet.name == "episode_0012.parquet"
    assert paths.video_paths["front"].suffix == ".mp4"
    assert paths.video_paths["wrist"].suffix == ".mp4"


def test_episode_metadata():
    metadata = EpisodeMetadata(
        episode_index=4,
        length=150,
        duration=5.0,
        success=True,
        task_index=2,
    )

    assert metadata.episode_index == 4
    assert metadata.length == 150
    assert metadata.duration == 5.0
    assert metadata.success is True
    assert metadata.task_index == 2


def test_episode_metadata_raw_is_preserved():
    raw = {
        "episode_index": 4,
        "length": 150,
        "duration": 5.0,
        "success": True,
        "custom_field": "example",
    }

    metadata = EpisodeMetadata(
        episode_index=4,
        raw=raw,
    )

    assert metadata.raw == raw
    assert metadata.raw["custom_field"] == "example"


def test_dataset_stats():
    stats = DatasetStats(
        action_percentiles={
            "action_0": np.array([-1.0, 1.0]),
            "action_1": np.array([-0.5, 0.5]),
        },
        median_episode_length=120.0,
        median_duration=4.0,
        median_path_length=2.5,
        task_counts={
            0: 10,
            1: 20,
        },
    )

    assert stats.median_episode_length == 120.0
    assert stats.median_duration == 4.0
    assert stats.median_path_length == 2.5
    assert stats.task_counts == {0: 10, 1: 20}

    np.testing.assert_array_equal(
        stats.action_percentiles["action_0"],
        np.array([-1.0, 1.0]),
    )


def test_episode_data():
    frame_index = np.arange(5)
    timestamps = np.array([0.0, 0.033, 0.066, 0.099, 0.132])
    states = np.zeros((5, 7), dtype=np.float32)
    actions = np.ones((5, 7), dtype=np.float32)
    task_indices = np.zeros(5, dtype=np.int64)

    data = EpisodeData(
        frame_index=frame_index,
        timestamps=timestamps,
        states=states,
        actions=actions,
        task_indices=task_indices,
    )

    assert data.frame_index.shape == (5,)
    assert data.timestamps.shape == (5,)
    assert data.states.shape == (5, 7)
    assert data.actions.shape == (5, 7)
    assert data.task_indices.shape == (5,)


def test_episode_data_video_defaults():
    data = EpisodeData()

    assert data.video_frames == {}
    assert data.video_timestamps == {}


def test_validation_config_defaults():
    config = ValidationConfig()

    assert config.short_circuit_on_hard is True
    assert config.skip_expensive_after_hard is True


def test_validation_config_custom_values():
    config = ValidationConfig(
        raw={
            "execution": {
                "short_circuit_on_hard": False,
                "skip_expensive_after_hard": False,
            }
        }
    )

    assert config.short_circuit_on_hard is False
    assert config.skip_expensive_after_hard is False


def test_episode_context():
    dataset_info = DatasetInfo(
        fps=30.0,
        state_dim=7,
        action_dim=7,
    )

    paths = EpisodePaths(
        episode_index=3,
        parquet=Path("/tmp/episode_0003.parquet"),
    )

    context = EpisodeContext(
        episode_index=3,
        dataset_info=dataset_info,
        paths=paths,
    )

    assert context.episode_index == 3
    assert context.dataset_info.fps == 30.0
    assert context.paths.episode_index == 3
    assert isinstance(context.data, EpisodeData)


def test_episode_validation_failure_properties():
    results = [
        CheckResult(
            check="frame_index",
            severity="HARD",
            message="Invalid frame index",
        ),
        CheckResult(
            check="image_quality",
            severity="SOFT",
            message="Too many blurry frames",
        ),
    ]

    validation = EpisodeValidation(
        episode_index=5,
        results=results,
    )

    assert validation.has_hard_failure is True
    assert len(validation.hard_failures) == 1
    assert len(validation.soft_failures) == 1
    assert validation.hard_failures[0].check == "frame_index"
    assert validation.soft_failures[0].check == "image_quality"


def test_episode_validation_without_failures():
    validation = EpisodeValidation(
        episode_index=5,
        results=[],
    )

    assert validation.has_hard_failure is False
    assert validation.hard_failures == []
    assert validation.soft_failures == []


def test_dataset_validation_flagged_episode_numbers():
    episode_1 = EpisodeValidation(
        episode_index=3,
        results=[
            CheckResult(
                check="frame_index",
                severity="HARD",
                message="bad frame index",
            )
        ],
    )

    episode_2 = EpisodeValidation(
        episode_index=1,
        results=[],
    )

    episode_3 = EpisodeValidation(
        episode_index=8,
        results=[
            CheckResult(
                check="nan_inf",
                severity="HARD",
                message="NaN detected",
            )
        ],
    )

    dataset = DatasetValidation(
        total_episodes=3,
        episodes=[episode_1, episode_2, episode_3],
    )

    assert dataset.flagged_episode_numbers == [3, 8]


def test_dataset_validation_classifications():
    episodes = [
        EpisodeValidation(
            episode_index=5,
            classification="KEEP",
        ),
        EpisodeValidation(
            episode_index=2,
            classification="REJECT",
        ),
        EpisodeValidation(
            episode_index=9,
            classification="REVIEW",
        ),
    ]

    dataset = DatasetValidation(
        total_episodes=3,
        episodes=episodes,
    )

    assert dataset.kept_episode_numbers == [5]
    assert dataset.rejected_episode_numbers == [2]
    assert dataset.review_episode_numbers == [9]


def test_dataset_validation_returns_sorted_episode_numbers():
    episodes = [
        EpisodeValidation(episode_index=10),
        EpisodeValidation(episode_index=2),
        EpisodeValidation(episode_index=7),
    ]

    for episode in episodes:
        episode.results.append(
            CheckResult(
                check="test",
                severity="SOFT",
                message="test failure",
            )
        )

    dataset = DatasetValidation(
        total_episodes=3,
        episodes=episodes,
    )

    assert dataset.flagged_episode_numbers == [2, 7, 10]