from __future__ import annotations

import numpy as np
import pytest

from lerobot_validator.dataset_quality import (
    check_diversity_contribution,
    check_task_balance,
    compute_novelty,
    cosine_similarity,
    detect_duplicate_episodes,
    find_duplicate_pairs,
    run_dataset_quality_checks,
)


# ============================================================
# Cosine similarity
# ============================================================

def test_cosine_similarity_identical_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])

    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])

    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])

    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_shape_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity(
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )


def test_cosine_similarity_zero_vectors():
    zero_a = np.zeros(3)
    zero_b = np.zeros(3)

    assert cosine_similarity(
        zero_a,
        zero_b,
    ) == pytest.approx(1.0)


# ============================================================
# Duplicate detection
# ============================================================

def test_find_duplicate_pairs_identical_embeddings():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    pairs = find_duplicate_pairs(
        embeddings,
        sim_thresh=0.98,
    )

    assert len(pairs) == 1

    i, j, similarity = pairs[0]

    assert (i, j) == (0, 1)
    assert similarity == pytest.approx(1.0)


def test_find_duplicate_pairs_no_duplicates():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    pairs = find_duplicate_pairs(
        embeddings,
        sim_thresh=0.98,
    )

    assert pairs == []


def test_duplicate_threshold():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.1],
        ]
    )

    pairs = find_duplicate_pairs(
        embeddings,
        sim_thresh=0.90,
    )

    assert len(pairs) == 1


def test_duplicate_episode_results():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    results = detect_duplicate_episodes(
        embeddings=embeddings,
        sim_thresh=0.98,
        episode_indices=[10, 20, 30],
    )

    assert len(results) == 1
    assert results[0].check == "duplicate_episodes"
    assert results[0].severity == "SOFT"
    assert "10" in results[0].message
    assert "20" in results[0].message


def test_duplicate_episode_indices_length_mismatch():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    with pytest.raises(ValueError):
        detect_duplicate_episodes(
            embeddings,
            episode_indices=[1],
        )


# ============================================================
# Task balance
# ============================================================

def test_task_balance_valid():
    counts = {
        0: 10,
        1: 12,
        2: 20,
    }

    results = check_task_balance(
        task_counts=counts,
        min_per_task=5,
    )

    assert results == []


def test_task_balance_flags_underrepresented_tasks():
    counts = {
        0: 10,
        1: 2,
        2: 20,
    }

    results = check_task_balance(
        task_counts=counts,
        min_per_task=5,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"
    assert results[0].check == "task_balance"
    assert "task 1" in results[0].message


def test_task_balance_empty_dataset():
    results = check_task_balance(
        task_counts={},
        min_per_task=5,
    )

    assert results == []


# ============================================================
# Diversity
# ============================================================

def test_novelty_with_empty_retained_set():
    embedding = np.array([1.0, 0.0])

    novelty = compute_novelty(
        embedding,
        np.empty((0, 2)),
    )

    assert novelty == pytest.approx(1.0)


def test_novelty_identical_embedding():
    embedding = np.array([1.0, 0.0])

    retained = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    novelty = compute_novelty(
        embedding,
        retained,
    )

    assert novelty == pytest.approx(0.0)


def test_novelty_orthogonal_embedding():
    embedding = np.array([0.0, 1.0])

    retained = np.array(
        [
            [1.0, 0.0],
        ]
    )

    novelty = compute_novelty(
        embedding,
        retained,
    )

    assert novelty == pytest.approx(1.0)


def test_diversity_contribution_passes():
    embedding = np.array([0.0, 1.0])

    retained = np.array(
        [
            [1.0, 0.0],
        ]
    )

    results = check_diversity_contribution(
        ep_embedding=embedding,
        retained_embeddings=retained,
        min_novelty=0.5,
    )

    assert results == []


def test_diversity_contribution_flags():
    embedding = np.array([1.0, 0.0])

    retained = np.array(
        [
            [1.0, 0.0],
        ]
    )

    results = check_diversity_contribution(
        ep_embedding=embedding,
        retained_embeddings=retained,
        min_novelty=0.2,
    )

    assert len(results) == 1
    assert results[0].severity == "SOFT"
    assert results[0].check == "diversity_contribution"


# ============================================================
# Phase 5 runner
# ============================================================

def test_run_dataset_quality_checks():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    results = run_dataset_quality_checks(
        embeddings=embeddings,
        episode_indices=[1, 2, 3],
        task_counts={
            0: 10,
            1: 2,
        },
        retained_embeddings=np.array(
            [
                [1.0, 0.0],
            ]
        ),
        duplicate_similarity_threshold=0.98,
        min_per_task=5,
        min_novelty=0.2,
    )

    checks = [
        result.check
        for result in results
    ]

    assert "duplicate_episodes" in checks
    assert "task_balance" in checks
    assert "diversity_contribution" in checks