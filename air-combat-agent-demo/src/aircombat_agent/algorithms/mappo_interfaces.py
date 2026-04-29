"""MAPPO-style engineering interfaces.

This module does not claim to be a full MAPPO implementation. Instead, it gives a
clean, runnable scaffold for the most error-prone engineering part of MAPPO
projects: data collection, shape alignment, masks, and done propagation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    share_obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    masks: np.ndarray
    info: Dict


@dataclass
class MultiAgentRolloutBuffer:
    """A minimal rollout buffer for multi-agent on-policy algorithms."""

    transitions: List[Transition] = field(default_factory=list)

    def add(self, transition: Transition) -> None:
        self._validate(transition)
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    def as_batch(self) -> Dict[str, np.ndarray]:
        if not self.transitions:
            raise ValueError("Cannot build batch from an empty buffer.")
        return {
            "obs": np.stack([t.obs for t in self.transitions]),
            "share_obs": np.stack([t.share_obs for t in self.transitions]),
            "actions": np.stack([t.actions for t in self.transitions]),
            "rewards": np.stack([t.rewards for t in self.transitions]),
            "dones": np.stack([t.dones for t in self.transitions]),
            "masks": np.stack([t.masks for t in self.transitions]),
        }

    @staticmethod
    def _validate(t: Transition) -> None:
        assert t.obs.ndim == 2, f"obs should be [n_agents, obs_dim], got {t.obs.shape}"
        assert t.share_obs.ndim == 2, f"share_obs should be [n_agents, share_dim], got {t.share_obs.shape}"
        assert t.actions.ndim == 1, f"actions should be [n_agents], got {t.actions.shape}"
        assert t.rewards.ndim == 1, f"rewards should be [n_agents], got {t.rewards.shape}"
        assert t.dones.ndim == 1, f"dones should be [n_agents], got {t.dones.shape}"
        assert t.masks.ndim == 2, f"masks should be [n_agents, 1], got {t.masks.shape}"
        n = t.obs.shape[0]
        assert t.share_obs.shape[0] == n
        assert t.actions.shape[0] == n
        assert t.rewards.shape[0] == n
        assert t.dones.shape[0] == n
        assert t.masks.shape[0] == n


def make_done_vector(done: bool, n_agents: int) -> np.ndarray:
    return np.full((n_agents,), bool(done), dtype=bool)


def random_policy_actions(n_agents: int, n_actions: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(low=0, high=n_actions, size=(n_agents,), dtype=int)
