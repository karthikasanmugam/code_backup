from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import lerobot_validator.video as video_module
from lerobot_validator.video import (
    VideoInfo,
    check_camera_coverage,
    check_frozen_video,
    check_image_quality,
    check_multiview_sync,
    check_resolution_consistency,
    check_video_decodable,
    check_video_exists,
    check_video_frame_count,
    check_video_timestamps,
    find_frozen_intervals,
    sample_video_frames,
    scan_video_segment,
)


# ============================================================
# Dummy assets
# ============================================================

def create_dummy_file(
    path: Path,
    size: int = 10,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        b"x" * size
    )

    return path


def make_frame(
    value: int,
    height: int = 32,
    width: int = 32,
) -> np.ndarray:
    return np.full(
        (height, width, 3),
        value,
        dtype=np.uint8,
    )


class FakeVideoCapture:
    """
    Small cv2.VideoCapture replacement for deterministic tests.
    """

    def __init__(
        self,
        frames: list[np.ndarray] | None = None,
        fps: float = 30.0,
    ):
        self.frames = frames or []
        self.fps = fps
        self.position = 0
        self.opened = True

    def isOpened(self):
        return self.opened

    def release(self):
        self.opened = False

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)

        if property_id == cv2.CAP_PROP_FPS:
            return self.fps

        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            if not self.frames:
                return 0
            return self.frames[0].shape[1]

        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            if not self.frames:
                return 0
            return self.frames[0].shape[0]

        if property_id == cv2.CAP_PROP_POS_MSEC:
            return (
                self.position / self.fps
            ) * 1000.0

        return 0

    def set(
        self,
        property_id,
        value,
    ):
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)

    def read(self):
        if self.position >= len(self.frames):
            return False, None

        frame = self.frames[self.position]

        self.position += 1

        return True, frame.copy()


def patch_capture(
    monkeypatch,
    frames: list[np.ndarray],
    fps: float = 30.0,
):
    def factory(_path):
        return FakeVideoCapture(
            frames=frames,
            fps=fps,
        )

    monkeypatch.setattr(
        video_module.cv2,
        "VideoCapture",
        factory,
    )


# ============================================================
# check_video_exists
# ============================================================

def test_video_exists_valid(tmp_path):
    front = create_dummy_file(
        tmp_path / "front.mp4"
    )

    wrist = create_dummy_file(
        tmp_path / "wrist.mp4"
    )

    results = check_video_exists(
        {
            "front": front,
            "wrist": wrist,
        }
    )

    assert results == []


def test_video_exists_missing(tmp_path):
    results = check_video_exists(
        {
            "front": tmp_path / "missing.mp4",
        }
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"
    assert results[0].check == "video_exists"


def test_video_exists_empty(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.touch()

    results = check_video_exists(
        {
            "front": empty,
        }
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"
    assert "empty" in results[0].message.lower()


# ============================================================
# Decoder
# ============================================================

def test_video_decodable_valid(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "video.mp4"
    )

    frames = [
        make_frame(0),
        make_frame(50),
        make_frame(100),
    ]

    patch_capture(
        monkeypatch,
        frames,
    )

    results = check_video_decodable(path)

    assert results == []


def test_video_decodable_failure(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "broken.mp4"
    )

    class BrokenCapture:
        def isOpened(self):
            return True

        def release(self):
            pass

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 0
            return 0

        def read(self):
            return False, None

        def set(self, *_args):
            pass

    monkeypatch.setattr(
        video_module.cv2,
        "VideoCapture",
        lambda _path: BrokenCapture(),
    )

    results = check_video_decodable(path)

    assert len(results) == 1
    assert results[0].severity == "HARD"


# ============================================================
# Frame count
# ============================================================

def test_video_frame_count_matches(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "video.mp4"
    )

    frames = [
        make_frame(0)
        for _ in range(30)
    ]

    patch_capture(
        monkeypatch,
        frames,
    )

    results = check_video_frame_count(
        video_path=path,
        n_rows=30,
    )

    assert results == []


def test_video_frame_count_mismatch(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "video.mp4"
    )

    frames = [
        make_frame(0)
        for _ in range(27)
    ]

    patch_capture(
        monkeypatch,
        frames,
    )

    results = check_video_frame_count(
        video_path=path,
        n_rows=30,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"
    assert "27" in results[0].message
    assert "30" in results[0].message


def test_video_frame_count_tolerance(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "video.mp4"
    )

    frames = [
        make_frame(0)
        for _ in range(29)
    ]

    patch_capture(
        monkeypatch,
        frames,
    )

    results = check_video_frame_count(
        video_path=path,
        n_rows=30,
        tol=1,
    )

    assert results == []


# ============================================================
# Resolution
# ============================================================

def test_resolution_matches(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "video.mp4"
    )

    frames = [
        make_frame(
            value=0,
            height=480,
            width=640,
        )
    ]

    patch_capture(
        monkeypatch,
        frames,
    )

    results = check_resolution_consistency(
        video_path=path,
        expected_hw=(480, 640),
    )

    assert results == []


def test_resolution_mismatch(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "video.mp4"
    )

    frames = [
        make_frame(
            value=0,
            height=224,
            width=224,
        )
    ]

    patch_capture(
        monkeypatch,
        frames,
    )

    results = check_resolution_consistency(
        video_path=path,
        expected_hw=(480, 640),
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"


# ============================================================
# Sampling
# ============================================================

def test_sample_video_frames(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "video.mp4"
    )

    frames = [
        make_frame(i * 10)
        for i in range(10)
    ]

    patch_capture(
        monkeypatch,
        frames,
    )

    sampled, indices = sample_video_frames(
        video_path=path,
        sample_every=2,
    )

    assert sampled.shape[0] == 5

    np.testing.assert_array_equal(
        indices,
        np.array([0, 2, 4, 6, 8]),
    )


def test_scan_video_segment_collects_frames_and_samples_in_one_pass(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(tmp_path / "video.mp4")
    source_frames = [make_frame(i * 10) for i in range(6)]
    patch_capture(monkeypatch, source_frames)

    timestamps, sampled, indices, expected, complete = scan_video_segment(
        path,
        sample_every=2,
    )

    assert len(timestamps) == 6
    assert expected == 6
    assert complete is True
    assert sampled.shape[0] == 3
    np.testing.assert_array_equal(indices, np.array([0, 2, 4]))


def test_sample_video_frames_max_limit(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "video.mp4"
    )

    frames = [
        make_frame(i)
        for i in range(10)
    ]

    patch_capture(
        monkeypatch,
        frames,
    )

    sampled, indices = sample_video_frames(
        video_path=path,
        sample_every=1,
        max_frames=3,
    )

    assert sampled.shape[0] == 3

    np.testing.assert_array_equal(
        indices,
        np.array([0, 1, 2]),
    )


# ============================================================
# Timestamp checks
# ============================================================

def test_video_timestamps_match(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "video.mp4"
    )

    frames = [
        make_frame(i)
        for i in range(5)
    ]

    patch_capture(
        monkeypatch,
        frames,
        fps=30.0,
    )

    parquet_ts = np.arange(5) / 30.0

    results = check_video_timestamps(
        video_path=path,
        parquet_ts=parquet_ts,
        tol_s=0.01,
    )

    assert results == []


def test_video_timestamps_mismatch(
    tmp_path,
    monkeypatch,
):
    path = create_dummy_file(
        tmp_path / "video.mp4"
    )

    frames = [
        make_frame(i)
        for i in range(5)
    ]

    patch_capture(
        monkeypatch,
        frames,
        fps=30.0,
    )

    parquet_ts = (
        np.arange(5) / 30.0
        + np.array([0, 0, 0.2, 0.2, 0.2])
    )

    results = check_video_timestamps(
        video_path=path,
        parquet_ts=parquet_ts,
        tol_s=0.01,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"


# ============================================================
# Image quality
# ============================================================

def test_image_quality_valid():
    rng = np.random.default_rng(123)

    frames = rng.integers(
        20,
        230,
        size=(10, 64, 64, 3),
        dtype=np.uint8,
    )

    results = check_image_quality(
        frames=frames,
        blur_thresh=1.0,
        dark_thresh=5.0,
        bright_thresh=250.0,
    )

    assert results == []


def test_image_quality_dark_frames():
    frames = np.zeros(
        (10, 64, 64, 3),
        dtype=np.uint8,
    )

    results = check_image_quality(
        frames=frames,
        blur_thresh=0.0,
        dark_thresh=10.0,
        bright_thresh=245.0,
        max_bad_ratio=0.05,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"


def test_image_quality_bright_frames():
    frames = np.full(
        (10, 64, 64, 3),
        255,
        dtype=np.uint8,
    )

    results = check_image_quality(
        frames=frames,
        blur_thresh=0.0,
        dark_thresh=0.0,
        bright_thresh=245.0,
        max_bad_ratio=0.05,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"


# ============================================================
# Frozen video
# ============================================================

def test_find_frozen_intervals_detects_freeze():
    frames = np.stack(
        [
            make_frame(0),
            make_frame(0),
            make_frame(0),
            make_frame(0),
            make_frame(0),
            make_frame(0),
        ]
    )

    actions = np.ones(
        (6, 2),
        dtype=np.float32,
    )

    intervals = find_frozen_intervals(
        frames=frames,
        actions=actions,
        diff_eps=1e-3,
        act_eps=0.03,
        fps=2.0,
        min_run_s=2.0,
    )

    assert len(intervals) >= 1


def test_find_frozen_intervals_no_freeze():
    frames = np.stack(
        [
            make_frame(0),
            make_frame(50),
            make_frame(100),
            make_frame(150),
            make_frame(200),
        ]
    )

    actions = np.ones(
        (5, 2),
        dtype=np.float32,
    )

    intervals = find_frozen_intervals(
        frames=frames,
        actions=actions,
        diff_eps=1e-3,
        act_eps=0.03,
        fps=30.0,
        min_run_s=2.0,
    )

    assert intervals == []


def test_frozen_video_returns_hard_failure():
    frames = np.stack(
        [
            make_frame(50)
            for _ in range(10)
        ]
    )

    actions = np.ones(
        (10, 3),
        dtype=np.float32,
    )

    results = check_frozen_video(
        frames=frames,
        actions=actions,
        fps=2.0,
        min_run_s=2.0,
    )

    assert len(results) >= 1
    assert all(
        result.severity == "HARD"
        for result in results
    )

    assert all(
        result.frame_range is not None
        for result in results
    )


# ============================================================
# Multi-view synchronization
# ============================================================

def test_multiview_sync_matches(
    tmp_path,
    monkeypatch,
):
    front = create_dummy_file(
        tmp_path / "front.mp4"
    )

    wrist = create_dummy_file(
        tmp_path / "wrist.mp4"
    )

    def fake_info(path):
        return VideoInfo(
            path=path,
            frame_count=100,
            fps=30.0,
            width=640,
            height=480,
            duration=100 / 30,
        )

    monkeypatch.setattr(
        video_module,
        "get_video_info",
        fake_info,
    )

    results = check_multiview_sync(
        {
            "front": front,
            "wrist": wrist,
        },
        tol_frames=1,
    )

    assert results == []


def test_multiview_sync_detects_mismatch(
    tmp_path,
    monkeypatch,
):
    front = create_dummy_file(
        tmp_path / "front.mp4"
    )

    wrist = create_dummy_file(
        tmp_path / "wrist.mp4"
    )

    def fake_info(path):
        count = (
            100
            if path == front
            else 96
        )

        return VideoInfo(
            path=path,
            frame_count=count,
            fps=30.0,
            width=640,
            height=480,
            duration=count / 30,
        )

    monkeypatch.setattr(
        video_module,
        "get_video_info",
        fake_info,
    )

    results = check_multiview_sync(
        {
            "front": front,
            "wrist": wrist,
        },
        tol_frames=1,
    )

    assert len(results) == 1
    assert results[0].severity == "HARD"


# ============================================================
# Camera coverage
# ============================================================

def test_camera_coverage_without_detector_is_skipped():
    frames = np.zeros(
        (10, 32, 32, 3),
        dtype=np.uint8,
    )

    results = check_camera_coverage(
        frames=frames,
        detector=None,
    )

    assert results == []


def test_camera_coverage_detector_passes():
    frames = np.zeros(
        (10, 32, 32, 3),
        dtype=np.uint8,
    )

    def detector(_frame):
        return True

    results = check_camera_coverage(
        frames=frames,
        detector=detector,
        sample_every=2,
    )

    assert results == []


def test_camera_coverage_detector_flags():
    frames = np.zeros(
        (10, 32, 32, 3),
        dtype=np.uint8,
    )

    def detector(_frame):
        return False

    results = check_camera_coverage(
        frames=frames,
        detector=detector,
        sample_every=2,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"


def test_camera_coverage_dict_detector():
    frames = np.zeros(
        (5, 32, 32, 3),
        dtype=np.uint8,
    )

    def detector(_frame):
        return {
            "target": True,
            "gripper": True,
        }

    results = check_camera_coverage(
        frames=frames,
        detector=detector,
    )

    assert results == []