from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .models import CheckResult


# ============================================================
# Video data model
# ============================================================

@dataclass(slots=True)
class VideoInfo:
    """Basic metadata extracted from one video."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    duration: float

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (height, width)."""
        return self.height, self.width


# ============================================================
# Video I/O helpers
# ============================================================

def open_video(video_path: Path) -> cv2.VideoCapture:
    """
    Open a video using OpenCV.

    The caller is responsible for releasing the capture.
    """
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    return capture


def get_video_info(video_path: Path) -> VideoInfo:
    """
    Read basic video metadata.

    This function does not decode the whole video.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video does not exist: {video_path}"
        )

    if video_path.stat().st_size == 0:
        raise ValueError(
            f"Video is empty: {video_path}"
        )

    capture = open_video(video_path)

    try:
        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )
        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if fps > 0 and frame_count > 0:
            duration = float(frame_count / fps)
        else:
            duration = 0.0

        return VideoInfo(
            path=video_path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration=duration,
        )
    finally:
        capture.release()


def read_first_frame(video_path: Path) -> np.ndarray:
    """Read and return the first decoded frame."""
    capture = open_video(video_path)

    try:
        success, frame = capture.read()

        if not success or frame is None:
            raise ValueError(
                f"Could not decode first frame: {video_path}"
            )

        return frame
    finally:
        capture.release()


def read_last_frame(video_path: Path) -> np.ndarray:
    """Seek to and return the last decoded frame."""
    capture = open_video(video_path)

    try:
        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if frame_count <= 0:
            raise ValueError(
                f"Video contains no frames: {video_path}"
            )

        # Try exact last-frame seek first.
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_count - 1,
        )

        success, frame = capture.read()

        if success and frame is not None:
            return frame

        # Some codecs do not seek perfectly. Fall back to reading
        # sequentially and retain the final successful frame.
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        last_frame = None

        while True:
            success, current = capture.read()

            if not success:
                break

            last_frame = current

        if last_frame is None:
            raise ValueError(
                f"Could not decode last frame: {video_path}"
            )

        return last_frame
    finally:
        capture.release()


def sample_video_frames(
    video_path: Path,
    sample_every: int = 15,
    max_frames: int | None = None,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample frames without decoding the entire video into memory.

    Returns:
        frames:
            ndarray of shape (N, H, W, C)
        indices:
            original video frame indices
    """
    if sample_every <= 0:
        raise ValueError(
            f"sample_every must be > 0, got {sample_every}"
        )

    capture = open_video(video_path)

    frames: list[np.ndarray] = []
    indices: list[int] = []

    try:
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        start_frame = max(0, int(round((start_time_s or 0.0) * video_fps))) if video_fps > 0 else 0
        end_frame = int(round(end_time_s * video_fps)) if end_time_s is not None and video_fps > 0 else None
        if start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = 0

        while end_frame is None or start_frame + frame_index < end_frame:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % sample_every == 0:
                frames.append(frame)
                indices.append(frame_index)

                if (
                    max_frames is not None
                    and len(frames) >= max_frames
                ):
                    break

            frame_index += 1
    finally:
        capture.release()

    if not frames:
        return (
            np.empty((0,), dtype=np.uint8),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(frames),
        np.asarray(indices, dtype=np.int64),
    )


def scan_video_segment(
    video_path: Path,
    *,
    time_range: tuple[float, float] | None = None,
    sample_every: int | None = None,
    max_sampled_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None, bool]:
    """Decode a video/episode segment once for all runner-level checks.

    Returns timestamps, sampled frames, local sample indices, expected frame
    count when known, and whether the full expected segment decoded.
    """
    if sample_every is not None and sample_every <= 0:
        raise ValueError(f"sample_every must be > 0, got {sample_every}")

    capture = open_video(video_path)
    timestamps: list[float] = []
    sampled_frames: list[np.ndarray] = []
    sampled_indices: list[int] = []

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = 0
        end_frame = total_frames if total_frames > 0 else None
        if time_range is not None and fps > 0:
            start_frame = max(0, int(round(time_range[0] * fps)))
            end_frame = max(start_frame, int(round(time_range[1] * fps)))
            if total_frames > 0:
                end_frame = min(end_frame, total_frames)
        if start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        local_index = 0
        while end_frame is None or start_frame + local_index < end_frame:
            success, frame = capture.read()
            if not success or frame is None:
                break

            timestamps.append(
                float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            )
            if (
                sample_every is not None
                and local_index % sample_every == 0
                and (
                    max_sampled_frames is None
                    or len(sampled_frames) < max_sampled_frames
                )
            ):
                sampled_frames.append(frame)
                sampled_indices.append(local_index)
            local_index += 1

        expected_frames = (
            max(0, end_frame - start_frame)
            if end_frame is not None
            else None
        )
        decoded_all = expected_frames is None or local_index == expected_frames
    finally:
        capture.release()

    timestamps_array = np.asarray(timestamps, dtype=np.float64)
    frames_array = (
        np.stack(sampled_frames)
        if sampled_frames
        else np.empty((0,), dtype=np.uint8)
    )
    indices_array = np.asarray(sampled_indices, dtype=np.int64)
    return timestamps_array, frames_array, indices_array, expected_frames, decoded_all


def read_all_frames(
    video_path: Path,
) -> np.ndarray:
    """
    Decode all frames.

    Intended for tests/small videos. Production pixel-level checks
    should generally use sample_video_frames().
    """
    capture = open_video(video_path)
    frames: list[np.ndarray] = []

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        return np.empty((0,), dtype=np.uint8)

    return np.stack(frames)


# ============================================================
# Phase 2 - Video existence
# ============================================================

def check_video_exists(
    video_paths: dict[str, Path],
) -> list[CheckResult]:
    """
    Verify every expected camera video exists and is non-zero.
    """
    results: list[CheckResult] = []

    for camera_key, path in video_paths.items():
        path = Path(path)

        if not path.exists():
            results.append(
                CheckResult(
                    check="video_exists",
                    severity="HARD",
                    message=(
                        f"Camera '{camera_key}' video is missing: "
                        f"{path}"
                    ),
                )
            )
            continue

        if not path.is_file():
            results.append(
                CheckResult(
                    check="video_exists",
                    severity="HARD",
                    message=(
                        f"Camera '{camera_key}' path is not a file: "
                        f"{path}"
                    ),
                )
            )
            continue

        if path.stat().st_size == 0:
            results.append(
                CheckResult(
                    check="video_exists",
                    severity="HARD",
                    message=(
                        f"Camera '{camera_key}' video is empty: "
                        f"{path}"
                    ),
                )
            )

    return results


# ============================================================
# Video decoding
# ============================================================

def check_video_decodable(
    video_path: Path,
    time_range: tuple[float, float] | None = None,
) -> list[CheckResult]:
    """
    Verify the first and last frames can be decoded.
    """
    results: list[CheckResult] = []

    try:
        if time_range is None:
            read_first_frame(video_path)
        else:
            capture = open_video(video_path)
            try:
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_range[0]) * 1000.0)
                success, frame = capture.read()
                if not success or frame is None:
                    raise ValueError("Could not decode episode segment first frame")
            finally:
                capture.release()
    except Exception as exc:
        results.append(
            CheckResult(
                check="video_decodable",
                severity="HARD",
                message=(
                    f"First frame cannot be decoded: {exc}"
                ),
            )
        )
        return results

    try:
        if time_range is None:
            read_last_frame(video_path)
        else:
            capture = open_video(video_path)
            try:
                capture.set(cv2.CAP_PROP_POS_MSEC, max(time_range[0], time_range[1] - 0.05) * 1000.0)
                success, frame = capture.read()
                if not success or frame is None:
                    raise ValueError("Could not decode episode segment last frame")
            finally:
                capture.release()
    except Exception as exc:
        results.append(
            CheckResult(
                check="video_decodable",
                severity="HARD",
                message=(
                    f"Last frame cannot be decoded: {exc}"
                ),
            )
        )

    return results


# ============================================================
# Frame count
# ============================================================

def check_video_frame_count(
    video_path: Path,
    n_rows: int,
    tol: int = 0,
    time_range: tuple[float, float] | None = None,
    observed_frames: int | None = None,
) -> list[CheckResult]:
    """
    Verify decoded/container frame count matches parquet rows.

    For normal files, metadata frame count is used first.
    A stricter decoder-based count is performed only when needed.
    """
    if tol < 0:
        raise ValueError(
            f"tol must be >= 0, got {tol}"
        )

    try:
        if observed_frames is not None:
            frame_count = int(observed_frames)
        elif time_range is None:
            info = get_video_info(video_path)
            frame_count = info.frame_count
        else:
            frame_count = len(extract_video_timestamps(video_path, *time_range))
    except Exception as exc:
        return [
            CheckResult(
                check="video_frame_count",
                severity="HARD",
                message=f"Could not inspect video: {exc}",
            )
        ]

    difference = abs(frame_count - n_rows)

    if difference <= tol:
        return []

    return [
        CheckResult(
            check="video_frame_count",
            severity="HARD",
            message=(
                f"Video segment has {frame_count} frames but parquet "
                f"has {n_rows} rows; difference={difference}."
            ),
        )
    ]


# ============================================================
# Resolution
# ============================================================

def check_resolution_consistency(
    video_path: Path,
    expected_hw: tuple[int, int],
) -> list[CheckResult]:
    """
    Verify video resolution matches expected (height, width).
    """
    if len(expected_hw) != 2:
        raise ValueError(
            "expected_hw must be (height, width)"
        )

    expected_height, expected_width = map(
        int,
        expected_hw,
    )

    try:
        info = get_video_info(video_path)
    except Exception as exc:
        return [
            CheckResult(
                check="resolution_consistency",
                severity="HARD",
                message=f"Could not inspect video: {exc}",
            )
        ]

    if (
        info.height == expected_height
        and info.width == expected_width
    ):
        return []

    return [
        CheckResult(
            check="resolution_consistency",
            severity="HARD",
            message=(
                f"Video resolution is "
                f"{info.height}x{info.width}; expected "
                f"{expected_height}x{expected_width}."
            ),
        )
    ]


# ============================================================
# Video timestamps
# ============================================================

def extract_video_timestamps(
    video_path: Path,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> np.ndarray:
    """
    Extract per-frame presentation timestamps using OpenCV.

    CAP_PROP_POS_MSEC is backend-dependent, so this helper is kept
    separate from the checker.
    """
    capture = open_video(video_path)
    timestamps: list[float] = []

    try:
        if start_time_s is not None:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_time_s) * 1000.0)
        while True:
            success, _ = capture.read()

            if not success:
                break

            timestamp_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            video_fps = float(capture.get(cv2.CAP_PROP_FPS))
            half_frame = 0.5 / video_fps if video_fps > 0 else 0.0
            if end_time_s is not None and timestamp_s >= end_time_s - half_frame:
                break
            if start_time_s is None or timestamp_s >= start_time_s - half_frame:
                timestamps.append(timestamp_s)
    finally:
        capture.release()

    return np.asarray(
        timestamps,
        dtype=np.float64,
    )


def check_video_timestamps(
    video_path: Path,
    parquet_ts: np.ndarray,
    tol_s: float,
    time_range: tuple[float, float] | None = None,
    observed_timestamps: np.ndarray | None = None,
) -> list[CheckResult]:
    """
    Verify video PTS aligns with parquet timestamps.
    """
    if tol_s < 0:
        raise ValueError(
            f"tol_s must be >= 0, got {tol_s}"
        )

    parquet_ts = np.asarray(
        parquet_ts,
        dtype=np.float64,
    )

    try:
        video_ts = (
            np.asarray(observed_timestamps, dtype=np.float64)
            if observed_timestamps is not None
            else extract_video_timestamps(
                video_path,
                *(time_range or (None, None)),
            )
        )
    except Exception as exc:
        return [
            CheckResult(
                check="video_timestamps",
                severity="HARD",
                message=(
                    f"Could not extract video timestamps: {exc}"
                ),
            )
        ]

    if len(video_ts) != len(parquet_ts):
        return [
            CheckResult(
                check="video_timestamps",
                severity="HARD",
                message=(
                    f"Video has {len(video_ts)} timestamps while "
                    f"parquet has {len(parquet_ts)} timestamps."
                ),
            )
        ]

    if len(video_ts) == 0:
        return []

    # Normalize both streams to their respective first timestamp.
    video_relative = video_ts - video_ts[0]
    parquet_relative = parquet_ts - parquet_ts[0]

    difference = np.abs(
        video_relative - parquet_relative
    )

    bad = np.flatnonzero(
        difference > tol_s
    )

    if len(bad) == 0:
        return []

    return [
        CheckResult(
            check="video_timestamps",
            severity="HARD",
            message=(
                f"{len(bad)} video timestamps differ from parquet "
                f"timestamps by more than {tol_s:.6f}s."
            ),
            frame_range=(
                int(bad[0]),
                int(bad[-1]) + 1,
            ),
        )
    ]


# ============================================================
# Image quality
# ============================================================

def _frame_brightness(frame: np.ndarray) -> float:
    """Return mean grayscale brightness in [0, 255]."""
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

    return float(np.mean(gray))


def _laplacian_variance(frame: np.ndarray) -> float:
    """Return Laplacian variance used as a blur indicator."""
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def check_image_quality(
    frames: np.ndarray,
    blur_thresh: float = 50.0,
    dark_thresh: float = 10.0,
    bright_thresh: float = 245.0,
    max_bad_ratio: float = 0.05,
) -> list[CheckResult]:
    """
    Check sampled frames for blur, excessive darkness and excessive
    brightness.

    This function expects already-sampled frames.
    """
    if not 0.0 <= max_bad_ratio <= 1.0:
        raise ValueError(
            "max_bad_ratio must be between 0 and 1"
        )

    frames = np.asarray(frames)

    if frames.size == 0:
        return []

    bad_indices: list[int] = []

    for index, frame in enumerate(frames):
        laplacian = _laplacian_variance(frame)
        brightness = _frame_brightness(frame)

        is_bad = (
            laplacian < blur_thresh
            or brightness < dark_thresh
            or brightness > bright_thresh
        )

        if is_bad:
            bad_indices.append(index)

    bad_ratio = len(bad_indices) / len(frames)

    if bad_ratio <= max_bad_ratio:
        return []

    return [
        CheckResult(
            check="image_quality",
            severity="SOFT",
            message=(
                f"{len(bad_indices)}/{len(frames)} sampled frames "
                f"({bad_ratio:.2%}) failed image-quality thresholds."
            ),
        )
    ]


# ============================================================
# Frozen video
# ============================================================

def _frame_difference(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> float:
    """
    Mean absolute pixel difference normalized to approximately [0, 1].
    """
    a = frame_a.astype(np.float32) / 255.0
    b = frame_b.astype(np.float32) / 255.0

    if a.shape != b.shape:
        return float("inf")

    return float(
        np.mean(
            np.abs(a - b)
        )
    )


def find_frozen_intervals(
    frames: np.ndarray,
    actions: np.ndarray,
    diff_eps: float,
    act_eps: float,
    fps: float,
    min_run_s: float,
) -> list[tuple[int, int]]:
    """
    Find frame ranges where video is effectively frozen while
    actions are non-zero.
    """
    if fps <= 0:
        raise ValueError(
            f"fps must be > 0, got {fps}"
        )

    if min_run_s < 0:
        raise ValueError(
            f"min_run_s must be >= 0, got {min_run_s}"
        )

    frames = np.asarray(frames)
    actions = np.asarray(actions)

    if len(frames) < 2 or len(actions) < 2:
        return []

    n = min(
        len(frames) - 1,
        len(actions) - 1,
    )

    frozen = np.zeros(
        n,
        dtype=bool,
    )

    for i in range(n):
        diff = _frame_difference(
            frames[i],
            frames[i + 1],
        )

        action_magnitude = float(
            np.linalg.norm(
                actions[i + 1]
            )
        )

        frozen[i] = (
            diff <= diff_eps
            and action_magnitude >= act_eps
        )

    min_run_frames = max(
        1,
        int(np.ceil(min_run_s * fps)),
    )

    intervals: list[tuple[int, int]] = []

    run_start: int | None = None

    for i, is_frozen in enumerate(frozen):
        if is_frozen and run_start is None:
            run_start = i

        if not is_frozen and run_start is not None:
            run_end = i + 1

            if run_end - run_start >= min_run_frames:
                intervals.append(
                    (run_start, run_end)
                )

            run_start = None

    if run_start is not None:
        run_end = len(frozen) + 1

        if run_end - run_start >= min_run_frames:
            intervals.append(
                (run_start, run_end)
            )

    return intervals


def check_frozen_video(
    frames: np.ndarray,
    actions: np.ndarray,
    diff_eps: float = 1e-3,
    act_eps: float = 0.03,
    min_run_s: float = 2.0,
    fps: float = 30.0,
) -> list[CheckResult]:
    """
    Flag long periods where frames barely change but actions are active.
    """
    intervals = find_frozen_intervals(
        frames=frames,
        actions=actions,
        diff_eps=diff_eps,
        act_eps=act_eps,
        fps=fps,
        min_run_s=min_run_s,
    )

    return [
        CheckResult(
            check="frozen_video",
            severity="HARD",
            message=(
                f"Video appears frozen while actions are active "
                f"for approximately {end - start} frames."
            ),
            frame_range=(start, end),
        )
        for start, end in intervals
    ]


# ============================================================
# Multi-view synchronization
# ============================================================

def check_multiview_sync(
    video_paths: dict[str, Path],
    tol_frames: int = 1,
    reference_camera: str | None = None,
) -> list[CheckResult]:
    """
    Verify camera frame counts agree within tolerance.

    Timestamp-level cross-camera checking can be added once the
    exact timestamp representation of the custom dataset is known.
    """
    if tol_frames < 0:
        raise ValueError(
            f"tol_frames must be >= 0, got {tol_frames}"
        )

    if reference_camera is not None and reference_camera not in video_paths:
        raise ValueError(
            f"reference_camera must be one of {sorted(video_paths)}, "
            f"got {reference_camera!r}"
        )

    if len(video_paths) < 2:
        return []

    infos: dict[str, VideoInfo] = {}
    results: list[CheckResult] = []

    for camera_key, path in video_paths.items():
        try:
            infos[camera_key] = get_video_info(path)
        except Exception as exc:
            results.append(
                CheckResult(
                    check="multiview_sync",
                    severity="HARD",
                    message=(
                        f"Could not inspect camera '{camera_key}': "
                        f"{exc}"
                    ),
                )
            )

    if results:
        return results

    counts = {
        camera: info.frame_count
        for camera, info in infos.items()
    }

    minimum = min(counts.values())
    maximum = max(counts.values())

    if maximum - minimum <= tol_frames:
        return []

    return [
        CheckResult(
            check="multiview_sync",
            severity="HARD",
            message=(
                "Camera frame counts are not synchronized: "
                + ", ".join(
                    f"{camera}={count}"
                    for camera, count in counts.items()
                )
                + f"; tolerance={tol_frames} frame(s)."
            ),
        )
    ]


# ============================================================
# Camera coverage
# ============================================================

def check_camera_coverage(
    frames: np.ndarray,
    detector: Callable[[np.ndarray], Any] | None = None,
    sample_every: int = 15,
) -> list[CheckResult]:
    """
    Optional target/gripper visibility check.

    detector is intentionally generic. It may be:
        - an object detector
        - a VLM wrapper
        - a custom callable

    When detector is None, the check is skipped.
    """
    if sample_every <= 0:
        raise ValueError(
            f"sample_every must be > 0, got {sample_every}"
        )

    if detector is None:
        return []

    frames = np.asarray(frames)

    if frames.size == 0:
        return []

    sampled = frames[::sample_every]

    missing_indices: list[int] = []

    for local_index, frame in enumerate(sampled):
        try:
            result = detector(frame)
        except Exception as exc:
            return [
                CheckResult(
                    check="camera_coverage",
                    severity="SOFT",
                    message=(
                        f"Camera coverage detector failed: {exc}"
                    ),
                )
            ]

        # Supported simple detector contracts:
        #   True  -> visible
        #   False -> missing
        #   dict with "target" and "gripper" booleans
        visible = False

        if isinstance(result, bool):
            visible = result

        elif isinstance(result, dict):
            visible = bool(
                result.get("target", False)
                and result.get("gripper", False)
            )

        if not visible:
            missing_indices.append(
                local_index * sample_every
            )

    if not missing_indices:
        return []

    return [
        CheckResult(
            check="camera_coverage",
            severity="SOFT",
            message=(
                f"Target/gripper visibility failed on "
                f"{len(missing_indices)} sampled frame(s)."
            ),
        )
    ]


# ============================================================
# Convenience runner
# ============================================================

def run_video_checks(
    video_paths: dict[str, Path],
    n_rows: int,
    expected_resolutions: dict[str, tuple[int, int]] | None = None,
    parquet_timestamps: np.ndarray | None = None,
    video_ranges: dict[str, tuple[float, float]] | None = None,
    fps: float = 30.0,
    sample_every: int = 15,
    image_quality_enabled: bool = True,
    frozen_video_enabled: bool = True,
    multiview_sync_enabled: bool = True,
    frame_count_tolerance: int = 0,
    timestamp_tolerance_s: float | None = None,
    multiview_tolerance_frames: int = 1,
    reference_camera: str = "auto",
    camera_coverage_detector: Callable[[np.ndarray], Any] | None = None,
    max_sampled_frames: int | None = None,
    sampled_video_out: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[CheckResult]:
    """
    Run Phase 2 checks for all supplied cameras.

    Expensive pixel checks are sampled.
    """
    results: list[CheckResult] = []

    # --------------------------------------------------------
    # Cheap file checks
    # --------------------------------------------------------

    results.extend(
        check_video_exists(video_paths)
    )

    # Do not continue with video decoding for missing files.
    existing_paths = {
        camera: path
        for camera, path in video_paths.items()
        if Path(path).exists()
        and Path(path).is_file()
        and Path(path).stat().st_size > 0
    }

    # --------------------------------------------------------
    # Per-camera checks
    # --------------------------------------------------------

    sampled_frames: dict[str, np.ndarray] = {}

    video_ranges = video_ranges or {}
    for camera_key, video_path in existing_paths.items():
        time_range = video_ranges.get(camera_key)
        needs_samples = (
            image_quality_enabled
            or frozen_video_enabled
            or camera_coverage_detector is not None
        )
        try:
            video_ts, frames, frame_indices, expected_frames, decoded_all = scan_video_segment(
                video_path,
                time_range=time_range,
                sample_every=sample_every if needs_samples else None,
                max_sampled_frames=max_sampled_frames,
            )
        except Exception as exc:
            results.append(
                CheckResult(
                    check="video_decodable",
                    severity="HARD",
                    message=f"Could not decode camera '{camera_key}': {exc}",
                )
            )
            continue

        if not decoded_all:
            results.append(
                CheckResult(
                    check="video_decodable",
                    severity="HARD",
                    message=(
                        f"Camera '{camera_key}' segment decoded {len(video_ts)} "
                        f"of {expected_frames} expected frames."
                    ),
                )
            )

        results.extend(
            check_video_frame_count(
                video_path=video_path,
                n_rows=n_rows,
                tol=frame_count_tolerance,
                time_range=time_range,
                observed_frames=len(video_ts),
            )
        )

        if expected_resolutions and camera_key in expected_resolutions:
            results.extend(
                check_resolution_consistency(
                    video_path=video_path,
                    expected_hw=expected_resolutions[camera_key],
                )
            )

        if parquet_timestamps is not None:
            results.extend(
                check_video_timestamps(
                    video_path=video_path,
                    parquet_ts=parquet_timestamps,
                    tol_s=(
                        timestamp_tolerance_s
                        if timestamp_tolerance_s is not None
                        else 0.5 / fps
                    ),
                    time_range=time_range,
                    observed_timestamps=video_ts,
                )
            )

        # Share the sampled segment across image-quality and frozen checks.
        if needs_samples:
            sampled_frames[camera_key] = frames
            if sampled_video_out is not None:
                sampled_video_out[camera_key] = (frames, frame_indices)

    # --------------------------------------------------------
    # Image quality
    # --------------------------------------------------------

    if image_quality_enabled:
        for camera_key, frames in sampled_frames.items():
            camera_results = check_image_quality(
                frames=frames,
            )

            for result in camera_results:
                result.message = (
                    f"[{camera_key}] {result.message}"
                )

            results.extend(camera_results)

    # --------------------------------------------------------
    # Frozen-video check
    # --------------------------------------------------------
    #
    # Action data is intentionally not passed into this runner yet.
    # The episode pipeline will call check_frozen_video() once the
    # sampled video frames are aligned with the action stream.
    #
    # Keeping that alignment outside this function avoids silently
    # assuming a particular custom dataset representation.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Multi-view synchronization
    # --------------------------------------------------------

    if multiview_sync_enabled:
        results.extend(
            check_multiview_sync(
                video_paths=existing_paths,
                tol_frames=multiview_tolerance_frames,
                reference_camera=(
                    None
                    if reference_camera == "auto"
                    else reference_camera
                ),
            )
        )

    # --------------------------------------------------------
    # Optional camera coverage
    # --------------------------------------------------------

    if camera_coverage_detector is not None:
        for camera_key, frames in sampled_frames.items():
            camera_results = check_camera_coverage(
                frames=frames,
                detector=camera_coverage_detector,
                sample_every=1,
            )

            for result in camera_results:
                result.message = (
                    f"[{camera_key}] {result.message}"
                )

            results.extend(camera_results)

    return results