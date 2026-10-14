from __future__ import annotations

import pytest

from lerobot_validator.models import CheckResult
from lerobot_validator.scoring import (
    classify_episode,
    score_and_classify_episode,
    score_episode,
    score_episode_by_check,
)


def hard_result(
    check: str = "frame_index",
) -> CheckResult:
    return CheckResult(
        check=check,
        severity="HARD",
        message="hard failure",
    )


def soft_result(
    check: str = "image_quality",
) -> CheckResult:
    return CheckResult(
        check=check,
        severity="SOFT",
        message="soft failure",
    )


# ============================================================
# score_episode
# ============================================================

def test_score_episode_no_failures():
    assert score_episode([]) == pytest.approx(1.0)


def test_score_episode_ignores_hard_failures():
    results = [
        hard_result(),
    ]

    assert score_episode(results) == pytest.approx(1.0)


def test_score_episode_soft_failure():
    results = [
        soft_result(),
    ]

    assert score_episode(results) == pytest.approx(0.0)


def test_score_episode_multiple_soft_failures():
    results = [
        soft_result("image_quality"),
        soft_result("action_smoothness"),
    ]

    score = score_episode(
        results,
        weights={
            "image_quality": 1.0,
            "action_smoothness": 1.0,
        },
    )

    assert score == pytest.approx(0.0)


def test_score_episode_zero_weights():
    results = [
        soft_result("image_quality"),
    ]

    score = score_episode(
        results,
        weights={
            "image_quality": 0.0,
        },
    )

    assert score == pytest.approx(1.0)


def test_score_episode_rejects_negative_weight():
    with pytest.raises(ValueError):
        score_episode(
            [],
            weights={
                "image_quality": -1.0,
            },
        )


# ============================================================
# score_episode_by_check
# ============================================================

def test_weighted_score_no_failures():
    score = score_episode_by_check(
        results=[],
        weights={
            "image_quality": 1.0,
            "action_smoothness": 1.0,
        },
    )

    assert score == pytest.approx(1.0)


def test_weighted_score_one_of_two_checks_fails():
    results = [
        soft_result("image_quality"),
    ]

    score = score_episode_by_check(
        results=results,
        weights={
            "image_quality": 1.0,
            "action_smoothness": 1.0,
        },
    )

    assert score == pytest.approx(0.5)


def test_weighted_score_respects_weights():
    results = [
        soft_result("action_smoothness"),
    ]

    score = score_episode_by_check(
        results=results,
        weights={
            "image_quality": 1.0,
            "action_smoothness": 3.0,
        },
    )

    assert score == pytest.approx(0.25)


def test_weighted_score_unknown_check():
    results = [
        soft_result("new_future_check"),
    ]

    score = score_episode_by_check(
        results=results,
        weights={
            "image_quality": 1.0,
        },
    )

    # Unknown checks do not penalize a configured scoring budget.
    assert score == pytest.approx(1.0)


def test_weighted_score_zero_weights():
    score = score_episode_by_check(
        results=[
            soft_result("image_quality"),
        ],
        weights={
            "image_quality": 0.0,
        },
    )

    assert score == pytest.approx(1.0)


# ============================================================
# classify_episode
# ============================================================

def test_classify_clean_episode():
    classification = classify_episode(
        results=[],
        score=1.0,
        min_score=0.70,
    )

    assert classification == "KEEP"


def test_classify_low_score():
    classification = classify_episode(
        results=[],
        score=0.50,
        min_score=0.70,
    )

    assert classification == "REVIEW"


def test_classify_score_at_threshold():
    classification = classify_episode(
        results=[],
        score=0.70,
        min_score=0.70,
    )

    assert classification == "KEEP"


def test_classify_hard_failure():
    classification = classify_episode(
        results=[
            hard_result(),
        ],
        score=1.0,
        min_score=0.70,
    )

    assert classification == "REJECT"


def test_classify_hard_failure_can_be_disabled():
    classification = classify_episode(
        results=[
            hard_result(),
        ],
        score=1.0,
        min_score=0.70,
        reject_on_hard=False,
    )

    assert classification == "KEEP"


def test_classify_hard_failure_and_low_score():
    classification = classify_episode(
        results=[
            hard_result(),
        ],
        score=0.2,
        min_score=0.70,
    )

    assert classification == "REJECT"


def test_classify_invalid_score():
    with pytest.raises(ValueError):
        classify_episode(
            results=[],
            score=1.1,
        )


def test_classify_invalid_min_score():
    with pytest.raises(ValueError):
        classify_episode(
            results=[],
            score=0.5,
            min_score=1.5,
        )


# ============================================================
# Combined operation
# ============================================================

def test_score_and_classify_keep():
    score, classification = score_and_classify_episode(
        results=[],
        weights={
            "image_quality": 1.0,
            "action_smoothness": 1.0,
        },
        min_score=0.70,
    )

    assert score == pytest.approx(1.0)
    assert classification == "KEEP"


def test_score_and_classify_review():
    results = [
        soft_result("image_quality"),
    ]

    score, classification = score_and_classify_episode(
        results=results,
        weights={
            "image_quality": 1.0,
            "action_smoothness": 1.0,
        },
        min_score=0.70,
    )

    assert score == pytest.approx(0.5)
    assert classification == "REVIEW"


def test_score_and_classify_reject():
    results = [
        hard_result(),
    ]

    score, classification = score_and_classify_episode(
        results=results,
        weights={
            "image_quality": 1.0,
        },
    )

    assert score == pytest.approx(1.0)
    assert classification == "REJECT"