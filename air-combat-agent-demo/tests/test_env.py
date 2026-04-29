import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aircombat_agent.envs.simple_air_combat_env import SimpleAirCombatEnv
from aircombat_agent.tactics.rule_based_pincer import RuleBasedPincerAgent
from aircombat_agent.algorithms.mappo_interfaces import MultiAgentRolloutBuffer, Transition, make_done_vector


def test_env_shapes():
    env = SimpleAirCombatEnv(seed=1)
    obs, share_obs, masks = env.reset()
    assert obs.shape[0] == env.n_agents
    assert share_obs.shape[0] == env.n_agents
    assert masks.shape == (env.n_agents, 1)


def test_rule_agent_actions():
    env = SimpleAirCombatEnv(seed=1)
    env.reset()
    agent = RuleBasedPincerAgent()
    actions = agent.act(env)
    assert actions.shape == (env.n_agents,)
    assert np.all(actions >= 0)
    assert np.all(actions < env.n_actions)


def test_buffer_add():
    env = SimpleAirCombatEnv(seed=1)
    obs, share_obs, masks = env.reset()
    actions = np.array([0, 1])
    _, _, rewards, done, info = env.step(actions)
    buffer = MultiAgentRolloutBuffer()
    buffer.add(Transition(obs, share_obs, actions, rewards, make_done_vector(done, env.n_agents), info["masks"], info))
    batch = buffer.as_batch()
    assert batch["obs"].shape[0] == 1
