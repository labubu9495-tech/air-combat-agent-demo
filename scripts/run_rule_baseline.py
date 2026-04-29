from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aircombat_agent.envs.simple_air_combat_env import SimpleAirCombatEnv
from aircombat_agent.tactics.rule_based_pincer import RuleBasedPincerAgent


def main() -> None:
    env = SimpleAirCombatEnv(seed=7, max_steps=260)
    agent = RuleBasedPincerAgent()
    env.reset()

    friendly_tracks = [[], []]
    enemy_track = []
    total_reward = 0.0

    done = False
    while not done:
        for i, f in enumerate(env.friendlies):
            friendly_tracks[i].append(f.pos.copy())
        enemy_track.append(env.enemy.pos.copy())

        actions = agent.act(env)
        _, _, rewards, done, info = env.step(actions)
        total_reward += float(np.mean(rewards))

    os.makedirs(PROJECT_ROOT / "outputs", exist_ok=True)
    plt.figure(figsize=(7, 5))
    for i, track in enumerate(friendly_tracks):
        arr = np.asarray(track)
        plt.plot(arr[:, 0], arr[:, 1], label=f"friendly-{i}")
        plt.scatter(arr[0, 0], arr[0, 1], marker="o")
        plt.scatter(arr[-1, 0], arr[-1, 1], marker="x")
    enemy_arr = np.asarray(enemy_track)
    plt.plot(enemy_arr[:, 0], enemy_arr[:, 1], label="target")
    plt.scatter(enemy_arr[0, 0], enemy_arr[0, 1], marker="o")
    plt.scatter(enemy_arr[-1, 0], enemy_arr[-1, 1], marker="x")
    plt.axis("equal")
    plt.xlabel("x / km")
    plt.ylabel("y / km")
    plt.title("Rule-based pincer tactic demo")
    plt.legend()
    out = PROJECT_ROOT / "outputs" / "rule_based_pincer.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)

    print(f"Episode finished at step={info['step']}, capture={info['capture']}, total_reward={total_reward:.2f}")
    print(f"Trajectory figure saved to: {out}")


if __name__ == "__main__":
    main()
