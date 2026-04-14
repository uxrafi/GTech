# Reinforcement Learning Report

## Overview
Experiments on two MDPs (Blackjack-v1 and CartPole-v1) using:
- **Model-based**: Value Iteration (VI), Policy Iteration (PI)
- **Model-free**: SARSA, Q-Learning (tabular)

## Directory Structure
```
rl/
├── src/
│   ├── config.py            # Hyperparameters and seeds
│   ├── discretize.py        # CartPole discretization utilities
│   ├── vi_pi.py             # Value Iteration & Policy Iteration
│   ├── sarsa.py             # Tabular SARSA
│   ├── qlearning.py         # Tabular Q-Learning (+ Double Q option)
│   ├── hyperparameter_search.py  # Staged search (random + successive halving)
│   ├── evaluate.py          # Policy evaluation helpers
│   ├── plots.py             # All figure generation
│   └── run_all.py           # Main entry point — runs everything
├── data/
│   └── (wine.csv, adult.csv — not used by RL experiments)
├── output/
│   └── figures/             # All generated PNGs saved here
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce All Results
```bash
cd rl
python src/run_all.py
```

All figures are written to `output/figures/`.

## Random Seeds
Seeds used: [42, 7, 13, 99, 2024]  
All randomization points (env reset, Q-table init, replay tie-breaks) are seeded explicitly.

## Hyperparameter Search Protocol
- **Stage 1**: N=24 random candidates, 200 pilot episodes, keep top 8
- **Stage 2**: 400 additional episodes for top 8, keep top 3
- **Stage 3**: Local refinement ±2× on α, ±25% on decay horizon around champion
- Final champion evaluated over 5 seeds × 2000 episodes

## CartPole Discretization
Bins: (x=3, ẋ=3, θ=8, θ̇=12)  
Clamps: x∈[-2.4,2.4], ẋ∈[-3,3], θ∈[-0.2,0.2], θ̇∈[-3.5,3.5]