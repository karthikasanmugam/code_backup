from __future__ import annotations

from collections.abc import Mapping

from .models import CheckResult, Classification


# ============================================================
# Score
# ============================================================

def score_episode(
    results: list[CheckResult],
    weights: Mapping[str, float] | None = None,
) -> float:
    """
    Convert SOFT validation failures into a quality score in [0, 1].

    Rules:
        - No SOFT failures -> 1.0
        - HARD failures do not directly reduce the score.
          Classification handles HARD failures separately.
        - Each SOFT failure contributes a weighted penalty.
        - Multiple failures of the same check are counted separately.

    Unknown checks receive a default weight of 1.0.
    """
    if weights is None:
        weights = {}

    for check_name, weight in weights.items():
        if weight < 0:
            raise ValueError(
                f"Weight for '{check_name}' must be >= 0, "
                f"got {weight}"
            )

    soft_results = [
        result
        for result in results
        if result.severity == "SOFT"
    ]

    if not soft_results:
        return 1.0

    total_penalty = 0.0
    total_weight = 0.0

    for result in soft_results:
        weight = float(
            weights.get(
                result.check,
                1.0,
            )
        )

        total_penalty += weight
        total_weight += weight

    if total_weight <= 0:
        return 1.0

    # Normalize penalty by the total weight of the encountered
    # SOFT checks so the result remains between 0 and 1.
    score = 1.0 - (
        total_penalty / total_weight
    )

    return max(
        0.0,
        min(1.0, score),
    )


# ============================================================
# More useful weighted score
# ============================================================

def score_episode_by_check(
    results: list[CheckResult],
    weights: Mapping[str, float] | None = None,
) -> float:
    """
    Score SOFT failures relative to configured check weights.

    Unlike score_episode(), this considers the configured weights
    as the complete scoring budget.

    Example:

        weights = {
            "image_quality": 1.0,
            "action_smoothness": 2.0,
            "idle_ratio": 1.0,
        }

    One failed action_smoothness check produces a larger penalty
    than one failed image_quality check.
    """
    if weights is None:
        weights = {}

    for check_name, weight in weights.items():
        if weight < 0:
            raise ValueError(
                f"Weight for '{check_name}' must be >= 0, "
                f"got {weight}"
            )

    positive_weights = {
        name: float(weight)
        for name, weight in weights.items()
        if weight > 0
    }

    if not positive_weights:
        return 1.0

    failed_checks = {
        result.check
        for result in results
        if result.severity == "SOFT"
    }

    penalty = sum(
        weight
        for check, weight in positive_weights.items()
        if check in failed_checks
    )

    total_weight = sum(
        positive_weights.values()
    )

    return max(
        0.0,
        min(
            1.0,
            1.0 - penalty / total_weight,
        ),
    )


# ============================================================
# Classification
# ============================================================

def classify_episode(
    results: list[CheckResult],
    score: float,
    min_score: float = 0.70,
    reject_on_hard: bool = True,
) -> Classification:
    """
    Classify an episode.

    Rules:
        HARD failure + reject_on_hard=True -> REJECT
        score < min_score                  -> REVIEW
        otherwise                          -> KEEP
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"score must be between 0 and 1, got {score}"
        )

    if not 0.0 <= min_score <= 1.0:
        raise ValueError(
            f"min_score must be between 0 and 1, got {min_score}"
        )

    has_hard = any(
        result.severity == "HARD"
        for result in results
    )

    if has_hard and reject_on_hard:
        return "REJECT"

    if score < min_score:
        return "REVIEW"

    return "KEEP"


# ============================================================
# Combined helper
# ============================================================

def score_and_classify_episode(
    results: list[CheckResult],
    weights: Mapping[str, float] | None = None,
    min_score: float = 0.70,
    reject_on_hard: bool = True,
) -> tuple[float, Classification]:
    """
    Convenience function used by the validation pipeline.
    """
    score = score_episode_by_check(
        results=results,
        weights=weights,
    )

    classification = classify_episode(
        results=results,
        score=score,
        min_score=min_score,
        reject_on_hard=reject_on_hard,
    )

    return score, classification