from __future__ import annotations

from typing import Iterable

import numpy as np

from .models import CheckResult


# ============================================================
# Helpers
# ============================================================

def _as_2d_float(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    """Convert embeddings to a 2D float array."""
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 1D or 2D, got shape {array.shape}"
        )

    return array


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    a = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(
            f"Embedding shapes do not match: "
            f"{a.shape} vs {b.shape}"
        )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if (
        norm_a <= np.finfo(np.float64).eps
        or norm_b <= np.finfo(np.float64).eps
    ):
        # Both zero vectors can reasonably be considered identical.
        if (
            norm_a <= np.finfo(np.float64).eps
            and norm_b <= np.finfo(np.float64).eps
        ):
            return 1.0

        return 0.0

    return float(
        np.dot(a, b)
        / (norm_a * norm_b)
    )


# ============================================================
# Duplicate episodes
# ============================================================

def find_duplicate_pairs(
    embeddings: np.ndarray,
    sim_thresh: float = 0.98,
) -> list[tuple[int, int, float]]:
    """
    Find pairs of embeddings with cosine similarity >= sim_thresh.

    Returns:
        [(index_a, index_b, similarity), ...]
    """
    if not -1.0 <= sim_thresh <= 1.0:
        raise ValueError(
            "sim_thresh must be between -1 and 1"
        )

    embeddings = _as_2d_float(
        embeddings,
        "embeddings",
    )

    n = len(embeddings)

    if n < 2:
        return []

    pairs: list[tuple[int, int, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            similarity = cosine_similarity(
                embeddings[i],
                embeddings[j],
            )

            if similarity >= sim_thresh:
                pairs.append(
                    (i, j, similarity)
                )

    return pairs


def detect_duplicate_episodes(
    embeddings: np.ndarray,
    sim_thresh: float = 0.98,
    episode_indices: Iterable[int] | None = None,
) -> list[CheckResult]:
    """
    Detect near-duplicate episodes from precomputed embeddings.

    The embedding itself can be constructed upstream from:
        first-frame embedding
        +
        downsampled action trace embedding
    """
    embeddings = _as_2d_float(
        embeddings,
        "embeddings",
    )

    if episode_indices is None:
        episode_indices = range(
            len(embeddings)
        )
    else:
        episode_indices = list(
            episode_indices
        )

    episode_indices = list(
        episode_indices
    )

    if len(episode_indices) != len(embeddings):
        raise ValueError(
            "episode_indices length must match embeddings"
        )

    pairs = find_duplicate_pairs(
        embeddings,
        sim_thresh,
    )

    results: list[CheckResult] = []

    for i, j, similarity in pairs:
        results.append(
            CheckResult(
                check="duplicate_episodes",
                severity="SOFT",
                message=(
                    f"Episodes {episode_indices[i]} and "
                    f"{episode_indices[j]} have cosine similarity "
                    f"{similarity:.5f}, above threshold "
                    f"{sim_thresh:.5f}."
                ),
            )
        )

    return results


# ============================================================
# Task balance
# ============================================================

def check_task_balance(
    task_counts: dict[int, int],
    min_per_task: int,
) -> list[CheckResult]:
    """
    Flag tasks represented by fewer than min_per_task episodes.
    """
    if min_per_task < 0:
        raise ValueError(
            "min_per_task must be >= 0"
        )

    if not task_counts:
        return []

    underrepresented = {
        task: count
        for task, count in task_counts.items()
        if count < min_per_task
    }

    if not underrepresented:
        return []

    details = ", ".join(
        f"task {task}: {count}"
        for task, count in sorted(
            underrepresented.items()
        )
    )

    return [
        CheckResult(
            check="task_balance",
            severity="SOFT",
            message=(
                f"{len(underrepresented)} task(s) have fewer than "
                f"{min_per_task} episode(s): {details}."
            ),
        )
    ]


# ============================================================
# Diversity contribution
# ============================================================

def compute_novelty(
    ep_embedding: np.ndarray,
    retained_embeddings: np.ndarray,
) -> float:
    """
    Compute novelty as:

        1 - max(cosine_similarity(episode, retained_episode))

    Higher values mean the episode is less similar to the retained set.
    """
    ep_embedding = np.asarray(
        ep_embedding,
        dtype=np.float64,
    ).reshape(-1)

    retained_embeddings = _as_2d_float(
        retained_embeddings,
        "retained_embeddings",
    )

    if len(retained_embeddings) == 0:
        return 1.0

    similarities = np.asarray(
        [
            cosine_similarity(
                ep_embedding,
                retained,
            )
            for retained in retained_embeddings
        ]
    )

    return float(
        1.0 - np.max(similarities)
    )


def check_diversity_contribution(
    ep_embedding: np.ndarray,
    retained_embeddings: np.ndarray,
    min_novelty: float,
) -> list[CheckResult]:
    """
    Check whether an episode contributes sufficient novelty to
    an already-retained dataset.
    """
    if not 0.0 <= min_novelty <= 2.0:
        raise ValueError(
            "min_novelty must be between 0 and 2"
        )

    novelty = compute_novelty(
        ep_embedding,
        retained_embeddings,
    )

    if novelty >= min_novelty:
        return []

    return [
        CheckResult(
            check="diversity_contribution",
            severity="SOFT",
            message=(
                f"Episode novelty is {novelty:.5f}; "
                f"minimum required novelty is {min_novelty:.5f}."
            ),
        )
    ]


# ============================================================
# Convenience runner
# ============================================================

def run_dataset_quality_checks(
    embeddings: np.ndarray | None = None,
    episode_indices: Iterable[int] | None = None,
    task_counts: dict[int, int] | None = None,
    retained_embeddings: np.ndarray | None = None,
    duplicate_similarity_threshold: float = 0.98,
    min_per_task: int | None = None,
    min_novelty: float | None = None,
) -> list[CheckResult]:
    """
    Run enabled Phase 5 dataset-level checks.

    All inputs are optional so the pipeline can enable checks
    incrementally as the required representations become available.
    """
    results: list[CheckResult] = []

    if embeddings is not None:
        results.extend(
            detect_duplicate_episodes(
                embeddings=embeddings,
                sim_thresh=duplicate_similarity_threshold,
                episode_indices=episode_indices,
            )
        )

    if (
        task_counts is not None
        and min_per_task is not None
    ):
        results.extend(
            check_task_balance(
                task_counts=task_counts,
                min_per_task=min_per_task,
            )
        )

    if (
        embeddings is not None
        and retained_embeddings is not None
        and min_novelty is not None
    ):
        for index, embedding in enumerate(
            embeddings
        ):
            results.extend(
                check_diversity_contribution(
                    ep_embedding=embedding,
                    retained_embeddings=retained_embeddings,
                    min_novelty=min_novelty,
                )
            )

    return results