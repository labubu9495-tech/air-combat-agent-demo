from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aircombat_agent.algorithms.mappo_interfaces import (
    MultiAgentRolloutBuffer,
    Transition,
    make_done_vector,
    random_policy_actions,
)
from aircombat_agent.envs.simple_air_combat_env import SimpleAirCombatEnv


def main() -> None:
    rng = np.random.default_rng(42)
    env = SimpleAirCombatEnv(seed=42, max_steps=20)
    obs, share_obs, masks = env.reset()
    buffer = MultiAgentRolloutBuffer()

    done = False
    while not done:
        actions = random_policy_actions(env.n_agents, env.n_actions, rng)
        next_obs, next_share_obs, rewards, done, info = env.step(actions)
        buffer.add(
            Transition(
                obs=obs,
                share_obs=share_obs,
                actions=actions,
                rewards=rewards,
                dones=make_done_vector(done, env.n_agents),
                masks=info["masks"],
                info=info,
            )
        )
        obs, share_obs = next_obs, next_share_obs

    batch = buffer.as_batch()
    print("MAPPO-style rollout batch shapes:")
    for k, v in batch.items():
        print(f"  {k:10s}: {v.shape}")
    print("Last reward terms:", buffer.transitions[-1].info["reward_terms"])


if __name__ == "__main__":
    main()
