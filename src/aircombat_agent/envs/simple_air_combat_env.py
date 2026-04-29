"""A lightweight 2D multi-agent air-combat environment.

The environment is intentionally compact. It is not a high-fidelity flight model;
it is a runnable scaffold that exposes the core engineering interfaces used in
multi-agent reinforcement learning: observations, shared observations, actions,
rewards, masks, and done flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from aircombat_agent.utils.geometry import angle_between, clamp, heading_to_vector, norm, unit, wrap_pi


@dataclass
class AircraftState:
    pos: np.ndarray
    heading: float
    speed: float
    alive: bool = True


class SimpleAirCombatEnv:
    """Two friendly agents versus one target aircraft.

    Action space per friendly agent:
        0: keep heading/speed
        1: turn left
        2: turn right
        3: accelerate
        4: decelerate
    """

    n_agents = 2
    n_actions = 5

    def __init__(self, seed: int | None = None, max_steps: int = 300):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.world_size = 100.0
        self.dt = 1.0
        self.max_turn = np.deg2rad(6.0)
        self.min_speed = 0.6
        self.max_speed = 2.4
        self.capture_range = 5.0
        self.boundary = 120.0
        self.step_count = 0
        self.friendlies: list[AircraftState] = []
        self.enemy: AircraftState | None = None

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.step_count = 0
        self.friendlies = [
            AircraftState(np.array([-55.0, -18.0]), heading=0.05, speed=1.8),
            AircraftState(np.array([-55.0, 18.0]), heading=-0.05, speed=1.8),
        ]
        self.enemy = AircraftState(np.array([30.0, 0.0]), heading=np.pi, speed=1.2)
        return self._obs(), self._share_obs(), self._masks()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Dict]:
        self.step_count += 1
        actions = np.asarray(actions, dtype=int).reshape(self.n_agents)

        for i, action in enumerate(actions):
            self._apply_action(self.friendlies[i], int(action))

        self._enemy_policy()
        self._integrate()

        rewards, reward_info = self._reward()
        done = self._done()
        info = {
            "step": self.step_count,
            "reward_terms": reward_info,
            "distances": [norm(f.pos - self.enemy.pos) for f in self.friendlies],
            "capture": min(norm(f.pos - self.enemy.pos) for f in self.friendlies) < self.capture_range,
        }
        return self._obs(), self._share_obs(), rewards, done, {**info, "masks": self._masks()}

    def _apply_action(self, aircraft: AircraftState, action: int) -> None:
        if not aircraft.alive:
            return
        if action == 1:
            aircraft.heading += self.max_turn
        elif action == 2:
            aircraft.heading -= self.max_turn
        elif action == 3:
            aircraft.speed += 0.08
        elif action == 4:
            aircraft.speed -= 0.08
        aircraft.speed = clamp(aircraft.speed, self.min_speed, self.max_speed)
        aircraft.heading = wrap_pi(aircraft.heading)

    def _enemy_policy(self) -> None:
        assert self.enemy is not None
        # The target slowly flies toward the negative x direction with slight evasive oscillation.
        self.enemy.heading = wrap_pi(np.pi + 0.25 * np.sin(self.step_count / 25.0))

    def _integrate(self) -> None:
        assert self.enemy is not None
        for aircraft in [*self.friendlies, self.enemy]:
            if aircraft.alive:
                aircraft.pos = aircraft.pos + heading_to_vector(aircraft.heading) * aircraft.speed * self.dt
                if np.any(np.abs(aircraft.pos) > self.boundary):
                    aircraft.alive = False

    def _obs(self) -> np.ndarray:
        assert self.enemy is not None
        obs = []
        for i, own in enumerate(self.friendlies):
            ally = self.friendlies[1 - i]
            own_v = heading_to_vector(own.heading) * own.speed
            ally_v = heading_to_vector(ally.heading) * ally.speed
            enemy_v = heading_to_vector(self.enemy.heading) * self.enemy.speed
            rel_enemy = self.enemy.pos - own.pos
            rel_ally = ally.pos - own.pos
            obs_i = np.concatenate([
                own.pos / self.world_size,
                own_v / self.max_speed,
                rel_enemy / self.world_size,
                enemy_v / self.max_speed,
                rel_ally / self.world_size,
                ally_v / self.max_speed,
                np.array([own.speed / self.max_speed, float(own.alive), norm(rel_enemy) / self.world_size]),
            ])
            obs.append(obs_i)
        return np.asarray(obs, dtype=np.float32)

    def _share_obs(self) -> np.ndarray:
        assert self.enemy is not None
        values = []
        for f in self.friendlies:
            values.extend([f.pos[0], f.pos[1], f.heading, f.speed, float(f.alive)])
        values.extend([self.enemy.pos[0], self.enemy.pos[1], self.enemy.heading, self.enemy.speed, float(self.enemy.alive)])
        share = np.asarray(values, dtype=np.float32) / np.asarray([self.world_size, self.world_size, np.pi, self.max_speed, 1.0] * 3, dtype=np.float32)
        return np.tile(share, (self.n_agents, 1))

    def _masks(self) -> np.ndarray:
        return np.asarray([[1.0 if f.alive else 0.0] for f in self.friendlies], dtype=np.float32)

    def _reward(self) -> Tuple[np.ndarray, Dict[str, float]]:
        assert self.enemy is not None
        distances = np.array([norm(f.pos - self.enemy.pos) for f in self.friendlies])
        approach = -0.01 * distances.mean()

        # Encourage both aircraft to keep a useful lateral separation instead of collapsing into one route.
        friendly_sep = norm(self.friendlies[0].pos - self.friendlies[1].pos)
        separation = -abs(friendly_sep - 28.0) / 100.0

        # Encourage forming a wide angle around the target.
        v0 = self.friendlies[0].pos - self.enemy.pos
        v1 = self.friendlies[1].pos - self.enemy.pos
        pincer_angle = angle_between(v0, v1)
        angle_reward = -abs(pincer_angle - np.deg2rad(80.0)) / np.pi

        capture = 10.0 if distances.min() < self.capture_range else 0.0
        boundary_penalty = -5.0 if any(not f.alive for f in self.friendlies) else 0.0
        total = approach + separation + angle_reward + capture + boundary_penalty
        rewards = np.asarray([total, total], dtype=np.float32)
        terms = {
            "approach": float(approach),
            "separation": float(separation),
            "pincer_angle": float(angle_reward),
            "capture": float(capture),
            "boundary_penalty": float(boundary_penalty),
            "total": float(total),
        }
        return rewards, terms

    def _done(self) -> bool:
        assert self.enemy is not None
        distances = [norm(f.pos - self.enemy.pos) for f in self.friendlies]
        captured = min(distances) < self.capture_range
        invalid = not any(f.alive for f in self.friendlies)
        timeout = self.step_count >= self.max_steps
        return bool(captured or invalid or timeout)
