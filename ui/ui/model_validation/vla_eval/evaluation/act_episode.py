"""
Offline ACT episode evaluation.

For every selected dataset timestep:

    observation
        ↓
    ACT
        ↓
    [chunk_size x action_dim] action chunk   (e.g. 16 x 16 or 32 x 16 for ACT)
        ↓
    first action
        ↓
    compare against actual action

The first action is used for the initial visualization because it gives
one predicted 16-D vector per dataset frame.
"""

from __future__ import annotations

import dataclasses
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

# Importing ACT registers the adapter.
import vla_eval.models.act  # noqa: F401
import vla_eval.models.diffusion  # noqa: F401
import vla_eval.models.smolvla  # noqa: F401
from vla_eval.core.base_vla import BaseVLA
from vla_eval.core.types import Sample
from vla_eval.datasets.lerobot_custom import (
    LeRobotCustomDataset,
)
from vla_eval.models.runtime import load_evaluation_model, prediction_seed
from vla_eval.evaluation.joints import resolve_action_order, resolve_state_order


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate per-joint and aggregate metrics.

    Shapes:

        actual    = [N, 16]
        predicted = [N, 16]
    """

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.ndim != 2 or actual.shape != predicted.shape or 0 in actual.shape:
        raise ValueError("Actual and predicted actions must have matching nonempty [frames, joints] shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted actions must contain only finite values")
    error = predicted - actual

    abs_error = np.abs(error)

    squared_error = error ** 2

    mae = np.mean(
        abs_error,
        axis=0,
    )

    rmse = np.sqrt(
        np.mean(
            squared_error,
            axis=0,
        )
    )

    max_abs_error = np.max(
        abs_error,
        axis=0,
    )

    return {
        "mae": mae.astype(float).tolist(),
        "rmse": rmse.astype(float).tolist(),
        "max_abs_error": max_abs_error.astype(float).tolist(),

        "mean_mae": float(
            np.mean(mae)
        ),

        "mean_rmse": float(
            np.mean(rmse)
        ),

        "max_joint_mae": float(
            np.max(mae)
        ),

        "max_joint_rmse": float(
            np.max(rmse)
        ),
    }


def evaluate_act_episode(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    episode_index: int,
    device: str = "cuda",
    stride: int = 1,
    max_frames: int | None = None,
    progress_callback=None,
    model_type: str = "act",
    inference_profile: str = "checkpoint",
    model: BaseVLA | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Run per-frame validation, optionally reusing a model loaded by the job owner.
    The first action is freshly predicted at every selected frame in all profiles.

    Args:
        dataset_path:
            Converted LeRobot v3.0 dataset.

        checkpoint_path:
            ACT pretrained_model directory.

        episode_index:
            Episode to evaluate.

        device:
            cuda or cpu.

        stride:
            Evaluate every Nth frame.

        max_frames:
            Optional maximum number of frames.

        progress_callback:
            callback(done, total).

    Returns:
        JSON-serializable evaluation result.
    """

    if stride < 1:
        raise ValueError(
            "stride must be >= 1"
        )
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")

    dataset = LeRobotCustomDataset(
        str(dataset_path)
    )

    try:

        episodes = {
            ep.episode_index: ep
            for ep in dataset.episodes
        }

        if episode_index not in episodes:
            raise ValueError(
                f"Episode {episode_index} not found. "
                f"Available episodes: "
                f"{sorted(episodes)}"
            )

        episode = episodes[
            episode_index
        ]

        global_frames = (
            dataset.get_episode_frames(
                episode_index
            )
        )

        selected_frames = global_frames[
            ::stride
        ]

        if max_frames is not None:
            selected_frames = (
                selected_frames[
                    :max_frames
                ]
            )

        if not selected_frames:
            raise ValueError(
                "No frames selected."
            )

        # -------------------------------------------------------------
        # Create model
        # -------------------------------------------------------------

        if model is None:
            model = load_evaluation_model(model_type, str(checkpoint_path), device, inference_profile)
        elif (model.get_metadata().model_type != model_type or
              Path(model.get_metadata().checkpoint_path).resolve() != Path(checkpoint_path).resolve()):
            raise ValueError("The reusable model does not match the requested checkpoint/model type")

        model.reset()

        # -------------------------------------------------------------
        # Storage
        # -------------------------------------------------------------

        actual_values = []

        predicted_values = []

        observed_state_values = []

        frame_indices = []

        global_indices = []

        latencies = []

        task = ""

        # -------------------------------------------------------------
        # Inference
        # -------------------------------------------------------------

        metadata = model.get_metadata()
        joint_names, joint_names_source, action_permutation = resolve_action_order(
            getattr(dataset, "info", {}), checkpoint_path, metadata.action_dim
        )
        # Same class of ordering mismatch, but on the *input* side: if the evaluation
        # dataset's observation.state columns aren't in the checkpoint's training
        # order, the model is fed a scrambled robot state and its predictions are
        # wrong from the start, not just mislabeled on the chart.
        _, _, state_permutation = resolve_state_order(
            getattr(dataset, "info", {}), checkpoint_path, metadata.action_dim
        )

        def fetch(index: int) -> Sample:
            sample = dataset[index]
            if state_permutation is None:
                return sample
            observation = dataclasses.replace(
                sample.observation, state=sample.observation.state[state_permutation]
            )
            leader_state = observation.extra.get("leader_state")
            if leader_state is not None:
                observation = dataclasses.replace(
                    observation, extra={**observation.extra, "leader_state": leader_state[state_permutation]}
                )
            return dataclasses.replace(sample, observation=observation)

        total = len(
            selected_frames
        )

        frame_positions = {
            global_index: position
            for position, global_index in enumerate(global_frames)
        }

        # Frames within an obs_horizon window overlap across consecutive steps
        # (e.g. step i needs [i-1, i], step i+1 needs [i, i+1]). This cache lets
        # an already-decoded frame be reused instead of re-fetched, and -- just
        # as importantly -- lets every fetch happen in ascending index order.
        # The underlying video reader (lerobot_custom.LeRobotCustomDataset)
        # only does a cheap sequential cap.read() when frames are requested in
        # increasing order; fetching a lower index after a higher one forces an
        # expensive backward seek. Always fetching missing indices ascending,
        # and evicting anything below the current window, keeps every access
        # monotonically forward for the whole episode.
        sample_cache: dict[int, Any] = {}
        evaluation_start = time.perf_counter()

        for completed, global_index in enumerate(
            selected_frames,
            start=1,
        ):

            if metadata.obs_horizon == 1:
                sample = fetch(global_index)
                observations = [sample.observation]
            else:
                frame_position = frame_positions[global_index]
                history_start = max(0, frame_position - metadata.obs_horizon + 1)
                history_indices = global_frames[history_start : frame_position + 1]

                for index in history_indices:
                    if index not in sample_cache:
                        sample_cache[index] = fetch(index)

                sample = sample_cache[global_index]
                observations = [sample_cache[index].observation for index in history_indices]
                if len(observations) < metadata.obs_horizon:
                    observations = [observations[0]] * (metadata.obs_horizon - len(observations)) + observations

                for cached_index in [idx for idx in sample_cache if idx < history_indices[0]]:
                    del sample_cache[cached_index]

            # The same frame gets the same noise regardless of stride/job scope.
            seeded = prediction_seed((seed + global_index) % 2**32, device) if metadata.stochastic else nullcontext()
            with seeded:
                prediction = model.predict(observations)

            actions = np.asarray(
                prediction.actions,
                dtype=np.float32,
            )

            # Every adapter returns a two-dimensional action chunk, shaped
            # [metadata.chunk_size, metadata.action_dim] (chunk_size varies by
            # model/checkpoint: 16 or 32 for ACT, 32 for Diffusion, 50 for SmolVLA).
            #
            if actions.shape != (metadata.chunk_size, metadata.action_dim):
                raise RuntimeError(
                    f"Unexpected {model_type} output shape: {actions.shape}; "
                    f"expected [horizon, joints] = {(metadata.chunk_size, metadata.action_dim)}"
                )

            if actions.shape[1] != metadata.action_dim:
                raise RuntimeError(
                    f"Unexpected {model_type} action dimension: "
                    f"{actions.shape}"
                )

            # ---------------------------------------------------------
            # Compare the first action in the returned chunk with the label.
            # ---------------------------------------------------------

            predicted = actions[0]

            actual = np.asarray(
                sample.ground_truth.action,
                dtype=np.float32,
            )
            observed_state = np.asarray(sample.observation.state, dtype=np.float32)
            if action_permutation is not None:
                actual = actual[action_permutation]

            if actual.shape != (metadata.action_dim,):
                raise RuntimeError(
                    "Unexpected ground-truth action shape: "
                    f"{actual.shape}"
                )
            if not np.isfinite(actual).all():
                raise RuntimeError(f"Ground-truth action contains NaN/Inf at dataset index {global_index}")
            if observed_state.shape != (metadata.action_dim,):
                raise RuntimeError(
                    f"Unexpected observation state shape: {observed_state.shape}; "
                    f"expected {(metadata.action_dim,)}"
                )
            if not np.isfinite(observed_state).all():
                raise RuntimeError(f"Observation state contains NaN/Inf at dataset index {global_index}")
            if (
                sample.ground_truth.episode_index != sample.observation.episode_index
                or sample.ground_truth.frame_index != sample.observation.frame_index
            ):
                raise RuntimeError("Observation and ground-truth action refer to different episode/frame indices")

            if not np.isfinite(
                predicted
            ).all():

                raise RuntimeError(
                    f"{model_type} produced NaN/Inf values at "
                    f"dataset index {global_index}"
                )

            actual_values.append(
                actual
            )

            predicted_values.append(
                predicted
            )

            observed_state_values.append(observed_state)

            frame_indices.append(
                int(
                    sample.observation.frame_index
                )
            )

            global_indices.append(
                int(global_index)
            )

            latencies.append(
                float(
                    prediction.inference_latency_s
                    or 0.0
                )
            )

            if not task:
                task = str(
                    sample.observation.instruction
                )

            if progress_callback:
                progress_callback(
                    completed,
                    total,
                )

        evaluation_wall_s = time.perf_counter() - evaluation_start
        actual_array = np.stack(
            actual_values,
            axis=0,
        )

        predicted_array = np.stack(
            predicted_values,
            axis=0,
        )
        observed_state_array = np.stack(observed_state_values, axis=0)

        metrics = calculate_metrics(
            actual_array,
            predicted_array,
        )

        total_latency = max(
            float(np.sum(latencies)),
            1e-9,
        )

        return {
            "dataset": str(
                Path(
                    dataset_path
                ).resolve()
            ),

            "checkpoint": str(
                Path(
                    checkpoint_path
                ).resolve()
            ),

            "episode_index": int(
                episode_index
            ),

            "episode_length": int(
                episode.length
            ),

            "evaluated_frames": int(
                len(selected_frames)
            ),

            "stride": int(
                stride
            ),

            "action_dim": metadata.action_dim,

            "joint_names": joint_names,
            "joint_names_source": joint_names_source,
            "action_units": "dataset native units",
            "comparison": "predicted_action[t, joint], recorded_action[t, joint], and observation.state[t, joint]",
            "episode_indices": [int(episode_index)] * len(selected_frames),
            "seed": seed,

            "prediction_alignment":
                f"first_action_of_{metadata.chunk_size}_step_chunk",

            "model_type": model_type,
            "runtime": metadata.extra.get("runtime", {"profile": "checkpoint", "device": device}),
            "inference_latencies_s": latencies,
            "evaluation_wall_s": evaluation_wall_s,
            "evaluated_frames_per_second": len(selected_frames) / max(evaluation_wall_s, 1e-9),

            "frame_indices":
                frame_indices,

            "global_indices":
                global_indices,

            "actual":
                actual_array.tolist(),

            "predicted":
                predicted_array.tolist(),

            "observed_state": observed_state_array.tolist(),

            "metrics":
                metrics,

            "latency": {
                "mean_s": float(
                    np.mean(latencies)
                ),

                "p50_s": float(
                    np.percentile(
                        latencies,
                        50,
                    )
                ),

                "p95_s": float(
                    np.percentile(
                        latencies,
                        95,
                    )
                ),

                "p99_s": float(
                    np.percentile(
                        latencies,
                        99,
                    )
                ),

                "actions_per_second":
                    float(
                        len(selected_frames)
                        / total_latency
                    ),
            },

            "task": task,
        }

    finally:

        dataset.close()
