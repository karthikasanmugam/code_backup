"""
Common dataset interface. A model/metrics/report never talks to a dataset
adapter directly beyond this contract — that's what makes the same runner
work whether the underlying data is this custom LeRobot-style format, HDF5,
RLDS, or anything else added later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from vla_eval.core.types import EpisodeInfo, Sample


class BaseDataset(ABC):
    """Indexable over the *entire* dataset (all episodes concatenated)."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of frames across all episodes."""

    @abstractmethod
    def __getitem__(self, index: int) -> Sample:
        """Global frame index -> Sample (Observation + GroundTruth)."""

    @property
    @abstractmethod
    def episodes(self) -> List[EpisodeInfo]:
        """Metadata for every episode in the dataset."""

    @abstractmethod
    def get_episode_frames(self, episode_index: int) -> List[int]:
        """Global frame indices belonging to one episode, in order.

        This is what the closed-loop / rollout-by-episode runner (Phase 8)
        will iterate over — it needs frames grouped and ordered per episode,
        not just a flat range.
        """

    def num_episodes(self) -> int:
        return len(self.episodes)
