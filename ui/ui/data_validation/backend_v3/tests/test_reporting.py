from __future__ import annotations

import csv
import json
from pathlib import Path

from lerobot_validator.models import (
    CheckResult,
    DatasetValidation,
    EpisodeValidation,
)
from lerobot_validator.reporting import (
    build_flagged_log,
    write_episode_ids,
    write_flagged_ids,
    write_kept_ids,
    write_reports,
    write_results_json,
    write_scores_csv,
)


# ============================================================
# Fixtures
# ============================================================

def make_validation() -> DatasetValidation:
    episode_3 = EpisodeValidation(
        episode_index=3,
        results=[
            CheckResult(
                check="frame_index",
                severity="HARD",
                message="frame index is invalid",
                frame_range=(10, 11),
            ),
            CheckResult(
                check="image_quality",
                severity="SOFT",
                message="too many dark frames",
            ),
        ],
        score=0.5,
        classification="REJECT",
    )

    episode_1 = EpisodeValidation(
        episode_index=1,
        results=[],
        score=1.0,
        classification="KEEP",
    )

    episode_7 = EpisodeValidation(
        episode_index=7,
        results=[
            CheckResult(
                check="idle_ratio",
                severity="SOFT",
                message="idle ratio too high",
            )
        ],
        score=0.8,
        classification="REVIEW",
    )

    return DatasetValidation(
        total_episodes=3,
        episodes=[
            episode_3,
            episode_1,
            episode_7,
        ],
    )


# ============================================================
# Log
# ============================================================

def test_build_flagged_log():
    validation = make_validation()

    content = build_flagged_log(
        validation=validation,
        data_root=Path("/dummy/dataset"),
        fps=30.0,
        stuck_thresh=2.0,
    )

    assert "FLAGGED EPISODES VALIDATION AUDIT" in content
    assert "Total Episodes Processed : 3" in content
    assert "Total Flagged Episodes   : 2" in content
    assert "Flagged Episode Numbers  : [3, 7]" in content

    assert "[EPISODE 3]" in content
    assert "[EPISODE 7]" in content

    assert "[HARD] frame_index" in content
    assert "[SOFT] image_quality" in content
    assert "Frames: 10-11" in content

    assert "END OF FLAGGED REPORT" in content


def test_build_flagged_log_no_flagged_episodes():
    validation = DatasetValidation(
        total_episodes=2,
        episodes=[
            EpisodeValidation(
                episode_index=0,
                results=[],
                score=1.0,
                classification="KEEP",
            ),
            EpisodeValidation(
                episode_index=1,
                results=[],
                score=1.0,
                classification="KEEP",
            ),
        ],
    )

    content = build_flagged_log(
        validation,
        Path("/dummy"),
    )

    assert "Total Flagged Episodes   : 0" in content
    assert "Flagged Episode Numbers  : None" in content
    assert "All episodes passed validation" in content


# ============================================================
# ID files
# ============================================================

def test_write_episode_ids(tmp_path):
    path = tmp_path / "ids.txt"

    result = write_episode_ids(
        [10, 2, 10, 5],
        path,
    )

    assert result == path

    assert path.read_text(
        encoding="utf-8"
    ) == (
        "2\n"
        "5\n"
        "10\n"
    )


def test_write_flagged_ids(tmp_path):
    validation = make_validation()

    path = tmp_path / "flagged.txt"

    write_flagged_ids(
        validation,
        path,
    )

    assert path.read_text(
        encoding="utf-8"
    ) == (
        "3\n"
        "7\n"
    )


def test_write_kept_ids(tmp_path):
    validation = make_validation()

    path = tmp_path / "kept.txt"

    write_kept_ids(
        validation,
        path,
    )

    assert path.read_text(
        encoding="utf-8"
    ) == "1\n"


# ============================================================
# CSV
# ============================================================

def test_write_scores_csv(tmp_path):
    validation = make_validation()

    path = tmp_path / "scores.csv"

    write_scores_csv(
        validation,
        path,
    )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 3

    # Output should be sorted by episode index.
    assert rows[0]["episode_index"] == "1"
    assert rows[1]["episode_index"] == "3"
    assert rows[2]["episode_index"] == "7"

    assert rows[0]["classification"] == "KEEP"
    assert rows[1]["classification"] == "REJECT"
    assert rows[2]["classification"] == "REVIEW"

    assert rows[1]["hard_failure_count"] == "1"
    assert rows[1]["soft_failure_count"] == "1"

    assert (
        rows[1]["checks"]
        == "frame_index;image_quality"
    )


# ============================================================
# JSON
# ============================================================

def test_write_results_json(tmp_path):
    validation = make_validation()

    path = tmp_path / "results.json"

    write_results_json(
        validation=validation,
        path=path,
        data_root=Path("/dummy/dataset"),
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert data["data_root"] == "/dummy/dataset"
    assert data["total_episodes"] == 3

    assert data["flagged_episode_numbers"] == [
        3,
        7,
    ]

    assert data["kept_episode_numbers"] == [1]
    assert data["rejected_episode_numbers"] == [3]
    assert data["review_episode_numbers"] == [7]

    assert len(data["episodes"]) == 3

    episode_3 = next(
        episode
        for episode in data["episodes"]
        if episode["episode_index"] == 3
    )

    assert episode_3["classification"] == "REJECT"
    assert episode_3["score"] == 0.5

    assert (
        episode_3["results"][0]["frame_range"]
        == [10, 11]
    )


# ============================================================
# Full report bundle
# ============================================================

def test_write_reports(tmp_path):
    validation = make_validation()

    output_dir = tmp_path / "reports"

    paths = write_reports(
        validation=validation,
        output_dir=output_dir,
        data_root=tmp_path / "dataset",
        fps=30.0,
        stuck_thresh=2.0,
    )

    expected_names = {
        "flagged_log",
        "flagged_ids",
        "kept_ids",
        "scores_csv",
        "results_json",
    }

    assert set(paths.keys()) == expected_names

    for path in paths.values():
        assert path.exists()
        assert path.is_file()

    assert (
        paths["flagged_log"].name
        == "flagged_episodes.log"
    )

    assert (
        paths["flagged_ids"].name
        == "flagged_episodes_ids.txt"
    )

    assert (
        paths["kept_ids"].name
        == "kept_episodes_ids.txt"
    )

    assert (
        paths["scores_csv"].name
        == "episode_scores.csv"
    )

    assert (
        paths["results_json"].name
        == "validation_results.json"
    )