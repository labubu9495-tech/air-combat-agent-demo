# Air Combat Agent Demo

A compact, runnable demonstration project for an **AI Agent-assisted multi-agent air-combat decision framework**.

This repository shows how a traditional rule-based two-aircraft pincer tactic can be converted into a structured multi-agent decision project. It includes:

- a lightweight 2D air-combat simulation environment;
- a rule-based pincer tactic baseline;
- reward decomposition for cooperative decision learning;
- buffer, mask, and done-signal interfaces for MAPPO-style training;
- runnable training/evaluation scripts;
- CI configuration for GitHub.

> This is a simplified research/demo scaffold. It is designed for project presentation, GitHub portfolio display, and further extension into real MAPPO/PPO frameworks.

## Project Motivation

Traditional rule-based air-combat tactics rely on manually designed logic, such as fixed convergence points, fixed approach angles, and fixed switching conditions. These rules are interpretable but often lack adaptability in dynamic confrontation scenarios.

This project demonstrates an upgrade path from **expert-rule tactics** to an **Agent-assisted multi-agent decision framework**:

1. Decompose the rule-based pincer tactic into tactical modules.
2. Convert tactical logic into state, action, reward, and termination designs.
3. Build a reusable environment interface for multi-agent reinforcement learning.
4. Use AI Agent support for logic abstraction, code structure, debugging, and iterative optimization.

## Repository Structure

```text
air-combat-agent-demo/
├── src/aircombat_agent/
│   ├── envs/simple_air_combat_env.py     # Lightweight multi-agent combat environment
│   ├── tactics/rule_based_pincer.py      # Rule-based cooperative pincer tactic
│   ├── algorithms/mappo_interfaces.py    # MAPPO-style buffer/mask/done scaffold
│   └── utils/geometry.py                 # Geometry helpers
├── scripts/
│   ├── run_rule_baseline.py              # Run and visualize rule-based tactic
│   └── inspect_mappo_flow.py             # Inspect obs/action/reward/mask/done flow
├── tests/test_env.py                     # Minimal tests
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/<your-name>/air-combat-agent-demo.git
cd air-combat-agent-demo
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the rule-based pincer baseline:

```bash
python scripts/run_rule_baseline.py
```

Inspect the MAPPO-style data flow:

```bash
python scripts/inspect_mappo_flow.py
```

Run tests:

```bash
pytest -q
```

## Core Design

### 1. Tactical prior

The original pincer tactic is represented as a cooperative rule module:

- select target;
- assign two friendly agents to left/right approach sides;
- calculate dynamic convergence points around the target;
- control heading and speed toward the convergence points;
- switch from convergence to attack when geometry conditions are satisfied.

### 2. Multi-agent learning interface

The environment exposes MAPPO-style data fields:

- `obs`: local observation of each friendly aircraft;
- `share_obs`: centralized training observation;
- `actions`: per-agent action commands;
- `rewards`: decomposed cooperative rewards;
- `dones`: episode termination flags;
- `masks`: valid-agent masks for inactive/dead agents.

### 3. Reward decomposition

The reward contains several interpretable terms:

- target approach reward;
- cooperative separation reward;
- pincer-angle reward;
- survival/validity reward;
- capture reward;
- boundary penalty.

This makes the project suitable for converting a rule-based tactic into a trainable multi-agent policy.

## Suggested GitHub Description

> AI Agent-assisted multi-agent air-combat decision framework: from rule-based pincer tactics to MAPPO-style training interfaces.

## Suggested Tags

`multi-agent-reinforcement-learning`, `air-combat`, `mappo`, `ppo`, `rule-based-tactics`, `ai-agent`, `simulation`
