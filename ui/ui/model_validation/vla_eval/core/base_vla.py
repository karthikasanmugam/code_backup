"""
Common VLA model interface. Every adapter (ACT, Diffusion Policy, ...) wraps
its checkpoint-loading, preprocessing, and inference behind this contract so
the runner, metrics engine, and reporting never need to know which model
produced a prediction.

Design choices baked in here (see project design-notes discussion):
  - predict() takes a LIST of Observations (an observation-history window),
    not a single Observation. ACT typically only needs the latest one
    (obs_horizon=1); Diffusion Policy typically needs the last 2
    (obs_horizon=2). Rather than special-casing this in the runner, the
    adapter declares how much history it needs via get_metadata().obs_horizon
    and the runner builds the window accordingly.
  - get_metadata() must be answerable truthfully after load() — the runner
    relies on it for windowing, chunk execution, and stochastic sampling
    instead of hardcoding per-model-family behavior.
  - reset() exists because both ACT (temporal ensembling buffer) and
    Diffusion Policy (action queue) carry internal state across an episode
    that must be cleared between episodes/rollouts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from vla_eval.core.types import ActionPrediction, ModelMetadata, Observation


class BaseVLA(ABC):
    @abstractmethod
    def load(self, checkpoint_path: str, config: Optional[dict] = None) -> None:
        """Load weights, normalization stats, processors/tokenizers, etc.

        Must populate whatever internal state get_metadata() reports on.
        Called once per evaluation run per checkpoint.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear any per-episode internal state (temporal ensembling buffer,
        action queue, KV cache, etc). Called at the start of every episode.
        """

    @abstractmethod
    def predict(self, observations: List[Observation]) -> ActionPrediction:
        """Run inference.

        Args:
            observations: history window, oldest-first, most-recent-last.
                Length must equal get_metadata().obs_horizon — the runner
                guarantees this.

        Returns:
            ActionPrediction with `actions` of shape
            (chunk_size, action_dim) — even if the model only predicts a
            single action, wrap it as chunk_size=1 for a uniform contract.
        """

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Must be valid after load(). See ModelMetadata docstring."""

    # ------------------------------------------------------------------
    # Optional convenience — not abstract, adapters can override if useful
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Adapters may override; default assumes get_metadata() raising
        or returning None means not loaded. Kept simple on purpose."""
        try:
            return self.get_metadata() is not None
        except Exception:
            return False