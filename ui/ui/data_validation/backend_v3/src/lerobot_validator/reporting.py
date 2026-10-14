from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    CheckResult,
    DatasetValidation,
    EpisodeValidation,
)


# ============================================================
# Helpers
# ============================================================

DIVIDER = "=" * 85
SUB_DIVIDER = "-" * 85


def _format_frame_range(
    frame_range: tuple[int, int] | None,
) -> str:
    if frame_range is None:
        return ""

    return f" | Frames: {frame_range[0]}-{frame_range[1]}"


def _result_to_dict(
    result: CheckResult,
) -> dict[str, Any]:
    return {
        "check": result.check,
        "severity": result.severity,
        "message": result.message,
        "frame_range": (
            list(result.frame_range)
            if result.frame_range is not None
            else None
        ),
    }


def _episode_to_dict(
    episode: EpisodeValidation,
    include_passed_checks: bool = False,
) -> dict[str, Any]:
    return {
        "episode_index": episode.episode_index,
        "score": episode.score,
        "classification": episode.classification,
        "results": [
            _result_to_dict(result)
            for result in episode.results
        ],
    }


def _ensure_output_dir(
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return output_dir


# ============================================================
# Existing-style flagged report
# ============================================================

def build_flagged_log(
    validation: DatasetValidation,
    data_root: Path,
    fps: float | None = None,
    stuck_thresh: float | None = None,
) -> str:
    """
    Build the human-readable audit log.

    The format intentionally follows the user's existing validator.
    """
    flagged_episodes = [
        episode
        for episode in validation.episodes
        if episode.results
    ]

    flagged_numbers = sorted(
        episode.episode_index
        for episode in flagged_episodes
    )

    lines: list[str] = []

    lines.append(DIVIDER)
    lines.append("FLAGGED EPISODES VALIDATION AUDIT")
    lines.append(
        f"Generated on   : "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(
        f"Data Root      : {Path(data_root)}"
    )

    if stuck_thresh is not None:
        lines.append(
            f"Stuck Thresh   : >= {stuck_thresh} seconds"
        )

    if fps is not None:
        lines.append(
            f"Nominal FPS    : {fps} Hz"
        )

    lines.append(DIVIDER)
    lines.append("")
    lines.append(
        f"Total Episodes Processed : "
        f"{validation.total_episodes}"
    )
    lines.append(
        f"Total Flagged Episodes   : "
        f"{len(flagged_episodes)}"
    )
    lines.append(
        f"Flagged Episode Numbers  : "
        f"{flagged_numbers if flagged_numbers else 'None'}"
    )
    lines.append("")
    lines.append(SUB_DIVIDER)
    lines.append("DETAILED FLAGGED EPISODE LOG")
    lines.append(SUB_DIVIDER)
    lines.append("")

    if not flagged_episodes:
        lines.append(
            "All episodes passed validation. "
            "No errors or flagged quality issues found."
        )
        lines.append("")
    else:
        for episode in flagged_episodes:
            lines.append(
                f"[EPISODE {episode.episode_index}]"
            )

            lines.append(
                f"  Classification : "
                f"{episode.classification or 'UNCLASSIFIED'}"
            )

            if episode.score is not None:
                lines.append(
                    f"  Score          : "
                    f"{episode.score:.4f}"
                )

            lines.append("  Reasons:")

            for index, result in enumerate(
                episode.results,
                start=1,
            ):
                frame_info = _format_frame_range(
                    result.frame_range
                )

                lines.append(
                    f"    ({index}) "
                    f"[{result.severity}] "
                    f"{result.check}: "
                    f"{result.message}"
                    f"{frame_info}"
                )

            lines.append("")

    lines.append(DIVIDER)
    lines.append("END OF FLAGGED REPORT")
    lines.append(DIVIDER)

    return "\n".join(lines) + "\n"


def write_flagged_log(
    validation: DatasetValidation,
    log_path: Path,
    data_root: Path,
    fps: float | None = None,
    stuck_thresh: float | None = None,
) -> Path:
    """
    Write the existing-style flagged audit log.
    """
    log_path = Path(log_path)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = build_flagged_log(
        validation=validation,
        data_root=data_root,
        fps=fps,
        stuck_thresh=stuck_thresh,
    )

    log_path.write_text(
        content,
        encoding="utf-8",
    )

    return log_path


# ============================================================
# Episode ID files
# ============================================================

def write_episode_ids(
    episode_indices: list[int],
    path: Path,
) -> Path:
    """
    Write one episode ID per line.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for episode_index in sorted(
            set(episode_indices)
        ):
            file.write(
                f"{episode_index}\n"
            )

    return path


def write_flagged_ids(
    validation: DatasetValidation,
    path: Path,
) -> Path:
    """Write flagged episode IDs."""
    return write_episode_ids(
        validation.flagged_episode_numbers,
        path,
    )


def write_kept_ids(
    validation: DatasetValidation,
    path: Path,
) -> Path:
    """Write KEEP episode IDs."""
    return write_episode_ids(
        validation.kept_episode_numbers,
        path,
    )


# ============================================================
# CSV report
# ============================================================

def write_scores_csv(
    validation: DatasetValidation,
    path: Path,
) -> Path:
    """
    Write one row per episode.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "episode_index",
        "score",
        "classification",
        "hard_failure_count",
        "soft_failure_count",
        "checks",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for episode in sorted(
            validation.episodes,
            key=lambda item: item.episode_index,
        ):
            writer.writerow(
                {
                    "episode_index": episode.episode_index,
                    "score": (
                        f"{episode.score:.6f}"
                        if episode.score is not None
                        else ""
                    ),
                    "classification": (
                        episode.classification or ""
                    ),
                    "hard_failure_count": len(
                        episode.hard_failures
                    ),
                    "soft_failure_count": len(
                        episode.soft_failures
                    ),
                    "checks": ";".join(
                        result.check
                        for result in episode.results
                    ),
                }
            )

    return path


# ============================================================
# JSON report
# ============================================================

def write_results_json(
    validation: DatasetValidation,
    path: Path,
    data_root: Path | None = None,
    include_passed_checks: bool = False,
) -> Path:
    """
    Write machine-readable validation results.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data: dict[str, Any] = {
        "generated_on": datetime.now().isoformat(
            timespec="seconds"
        ),
        "data_root": (
            str(Path(data_root))
            if data_root is not None
            else None
        ),
        "total_episodes": validation.total_episodes,
        "flagged_episode_numbers": (
            validation.flagged_episode_numbers
        ),
        "kept_episode_numbers": (
            validation.kept_episode_numbers
        ),
        "rejected_episode_numbers": (
            validation.rejected_episode_numbers
        ),
        "review_episode_numbers": (
            validation.review_episode_numbers
        ),
        "episodes": [
            _episode_to_dict(
                episode,
                include_passed_checks=include_passed_checks,
            )
            for episode in sorted(
                validation.episodes,
                key=lambda item: item.episode_index,
            )
        ],
        "dataset_results": [
            _result_to_dict(result)
            for result in validation.dataset_results
        ],
    }

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


# ============================================================
# Report bundle
# ============================================================

def write_reports(
    validation: DatasetValidation,
    output_dir: Path,
    data_root: Path,
    fps: float | None = None,
    stuck_thresh: float | None = None,
) -> dict[str, Path]:
    """
    Write the complete Phase 6 report bundle.
    """
    output_dir = _ensure_output_dir(
        output_dir
    )

    paths = {
        "flagged_log": (
            output_dir / "flagged_episodes.log"
        ),
        "flagged_ids": (
            output_dir / "flagged_episodes_ids.txt"
        ),
        "kept_ids": (
            output_dir / "kept_episodes_ids.txt"
        ),
        "scores_csv": (
            output_dir / "episode_scores.csv"
        ),
        "results_json": (
            output_dir / "validation_results.json"
        ),
    }

    write_flagged_log(
        validation=validation,
        log_path=paths["flagged_log"],
        data_root=data_root,
        fps=fps,
        stuck_thresh=stuck_thresh,
    )

    write_flagged_ids(
        validation,
        paths["flagged_ids"],
    )

    write_kept_ids(
        validation,
        paths["kept_ids"],
    )

    write_scores_csv(
        validation,
        paths["scores_csv"],
    )

    write_results_json(
        validation,
        paths["results_json"],
        data_root=data_root,
    )

    return paths