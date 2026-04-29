"""Rule-based two-aircraft pincer tactic baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aircombat_agent.envs.simple_air_combat_env import SimpleAirCombatEnv
from aircombat_agent.utils.geometry import angle_of, norm, rotate, unit, wrap_pi


@dataclass
class PincerConfig:
    lateral_offset: float = 22.0
    forward_offset: float = 18.0
    heading_tolerance: float = 0.08
    close_range: float = 18.0


class RuleBasedPincerAgent:
    """Converts pincer-tactic geometry into discrete actions."""

    def __init__(self, config: PincerConfig | None = None):
        self.config = config or PincerConfig()

    def act(self, env: SimpleAirCombatEnv) -> np.ndarray:
        assert env.enemy is not None
        enemy = env.enemy
        enemy_forward = unit(np.array([np.cos(enemy.heading), np.sin(enemy.heading)]))
        enemy_left = rotate(enemy_forward, np.pi / 2.0)

        actions = []
        for i, friendly in enumerate(env.friendlies):
            side = -1.0 if i == 0 else 1.0
            # Dynamic convergence point: behind and laterally offset from target.
            convergence = (
                enemy.pos
                - enemy_forward * self.config.forward_offset
                + side * enemy_left * self.config.lateral_offset
            )
            to_point = convergence - friendly.pos
            desired_heading = angle_of(to_point)
            heading_error = wrap_pi(desired_heading - friendly.heading)

            if abs(heading_error) > self.config.heading_tolerance:
                action = 1 if heading_error > 0 else 2
            else:
                dist_to_enemy = norm(enemy.pos - friendly.pos)
                action = 4 if dist_to_enemy < self.config.close_range else 3
            actions.append(action)
        return np.asarray(actions, dtype=int)
