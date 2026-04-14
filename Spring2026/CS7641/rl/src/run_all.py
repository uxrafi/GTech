"""
run_all.py
Main entry point for CS7641 RL Report experiments.

Usage:
  cd rl/src
  python run_all.py
"""

import sys, os
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

import time
import numpy as np
import gymnasium as gym

from config import (SEEDS, SARSA_CONFIG, QLEARN_CONFIG, VI_PI_CONFIG,
                    CARTPOLE_BINS, CARTPOLE_CLAMPS, CARTPOLE_GRIDS,
                    EVAL_EPISODES, MAX_STEPS)
from vi_pi      import value_iteration, policy_iteration
from sarsa      import sarsa
from qlearning  import qlearning
from discretize import make_bins, discretize, n_states, build_transition_reward
from evaluate   import (evaluate_q_policy, evaluate_vi_pi_policy,
                        aggregate_returns, smooth)
from plots      import (plot_vi_pi_convergence, plot_vi_pi_comparison,
                        plot_learning_curves, plot_delta_q,
                        plot_blackjack_policy, plot_blackjack_value,
                        plot_discretization_ablation, plot_episode_lengths,
                        plot_eval_distribution)

print("=" * 60)
print("CS7641 RL Report — Spring 2026")
print(f"Seeds: {SEEDS}")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# BLACKJACK
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Blackjack-v1 ──────────────────────────────────────────")

def _build_blackjack_P_manual():
    # manual fallback in case bettermdptools isnt installed
    # 200k steps gives decent coverage of the state space
    N_S = 32 * 11 * 2
    N_A = 2
    P   = {s: {a: [] for a in range(N_A)} for s in range(N_S)}
    visited = {s: {a: False for a in range(N_A)} for s in range(N_S)}

    def obs_to_idx(obs):
        ps, dc, ace = obs
        return int(ps) * 11 * 2 + int(dc) * 2 + int(ace)

    rng = np.random.default_rng(0)
    env = gym.make("Blackjack-v1", sab=True)
    for _ in range(200_000):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        while not done:
            s = obs_to_idx(obs)
            a = int(rng.integers(0, N_A))
            next_obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s_next = obs_to_idx(next_obs) if not done else s
            if not visited[s][a]:
                P[s][a] = [(1.0, s_next, float(r), bool(terminated))]
                visited[s][a] = True
            obs = next_obs
    env.close()
    # fill unvisited with self loops
    for s in range(N_S):
        for a in range(N_A):
            if not P[s][a]:
                P[s][a] = [(1.0, s, 0.0, False)]
    return P, N_S, N_A


def build_blackjack_P():
    # try bettermdptools first, fall back to manual if its not installed
    try:
        from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
        from bettermdptools.utils.mdp_helpers import openai_gym_apply
        env  = gym.make("Blackjack-v1", sab=True)
        bj   = BlackjackWrapper(env)
        P, R = openai_gym_apply(bj)
        return P, len(P), 2
    except Exception as e:
        print(f"  bettermdptools fallback ({e}); using manual Blackjack P.")
        return _build_blackjack_P_manual()


P_bj, n_s_bj, n_a_bj = build_blackjack_P()
print(f"  Blackjack: {n_s_bj} states, {n_a_bj} actions")

print("  Running VI on Blackjack...")
V_vi_bj, pol_vi_bj, hist_vi_bj = value_iteration(P_bj, n_s_bj, n_a_bj)
print(f"    VI: {hist_vi_bj['n_iters']} iters, {hist_vi_bj['wall_clock']:.2f}s")

print("  Running PI on Blackjack...")
V_pi_bj, pol_pi_bj, hist_pi_bj = policy_iteration(P_bj, n_s_bj, n_a_bj)
print(f"    PI: {hist_pi_bj['n_iters']} iters, {hist_pi_bj['wall_clock']:.2f}s")

plot_vi_pi_convergence(hist_vi_bj, hist_pi_bj, "Blackjack-v1")
plot_vi_pi_comparison(hist_vi_bj, hist_pi_bj, "Blackjack-v1")
plot_blackjack_value(V_vi_bj, V_pi_bj)


class BlackjackFlatEnv:
    # wrapper to flatten the tuple observation into a single int
    # needed so SARSA/Q-learning can index into Q directly
    def __init__(self):
        self._env = gym.make("Blackjack-v1", sab=True)
        self.action_space      = self._env.action_space
        self.observation_space = self._env.observation_space
    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        ps, dc, ace = obs
        return int(ps) * 11 * 2 + int(dc) * 2 + int(ace), info
    def step(self, a):
        obs, r, term, trunc, info = self._env.step(a)
        ps, dc, ace = obs
        return int(ps) * 11 * 2 + int(dc) * 2 + int(ace), r, term, trunc, info
    def close(self):
        self._env.close()


print("  Running SARSA on Blackjack (5 seeds)...")
bj_sarsa_returns, bj_sarsa_dq = [], []
for seed in SEEDS:
    Q, returns, hist = sarsa(BlackjackFlatEnv, n_s_bj, n_a_bj,
                             cfg=SARSA_CONFIG, seed=seed)
    bj_sarsa_returns.append(returns)
    bj_sarsa_dq.append(hist["delta_q"])
print(f"    Mean tail return: {np.mean([np.mean(r[-100:]) for r in bj_sarsa_returns]):.3f}")

print("  Running Q-Learning on Blackjack (5 seeds)...")
bj_qlearn_returns, bj_qlearn_dq = [], []
Q_bj_best = None
for seed in SEEDS:
    Q, returns, hist = qlearning(BlackjackFlatEnv, n_s_bj, n_a_bj,
                                  cfg=QLEARN_CONFIG, seed=seed)
    bj_qlearn_returns.append(returns)
    bj_qlearn_dq.append(hist["delta_q"])
    # just keep the first seed's Q for the policy plot -- they're all similar anyway
    if Q_bj_best is None:
        Q_bj_best = Q
print(f"    Mean tail return: {np.mean([np.mean(r[-100:]) for r in bj_qlearn_returns]):.3f}")

plot_learning_curves({"SARSA": bj_sarsa_returns,
                      "Q-Learning": bj_qlearn_returns}, "Blackjack-v1")
plot_delta_q({"SARSA": bj_sarsa_dq,
              "Q-Learning": bj_qlearn_dq}, "Blackjack-v1")
plot_blackjack_policy(Q_bj_best, title="Q-Learning Policy — Blackjack")
plot_eval_distribution({
    "SARSA":      [r for sr in bj_sarsa_returns  for r in sr[-EVAL_EPISODES:]],
    "Q-Learning": [r for sr in bj_qlearn_returns for r in sr[-EVAL_EPISODES:]],
}, "Blackjack-v1")


# ══════════════════════════════════════════════════════════════════════════════
# CARTPOLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n── CartPole-v1 ───────────────────────────────────────────")

print("  Running discretization ablation study...")
grid_labels, abl_returns, abl_stds, abl_clocks, abl_nstates = [], [], [], [], []

for grid in CARTPOLE_GRIDS:
    _edges = make_bins(grid, CARTPOLE_CLAMPS)
    ns = n_states(grid)

    class _AblEnv:
        edges = _edges
        def __init__(self):
            self._env = gym.make("CartPole-v1")
            self.action_space      = self._env.action_space
            self.observation_space = self._env.observation_space
        def reset(self, seed=None):
            obs, info = self._env.reset(seed=seed)
            return discretize(obs, self.edges), info
        def step(self, a):
            obs, r, term, trunc, info = self._env.step(a)
            return discretize(obs, self.edges), r, term, trunc, info
        def close(self):
            self._env.close()

    # 1000 episodes is enough to see which grids are hopeless
    abl_cfg = dict(QLEARN_CONFIG)
    abl_cfg["n_episodes"] = 1000
    t0 = time.perf_counter()
    Q_abl, _, _ = qlearning(_AblEnv, ns, 2, cfg=abl_cfg, seed=42)
    wall = time.perf_counter() - t0
    eval_r, _ = evaluate_q_policy(_AblEnv, Q_abl, n_episodes=100,
                                   max_steps=MAX_STEPS, seed=0)
    grid_labels.append(str(grid))
    abl_returns.append(float(eval_r.mean()))
    abl_stds.append(float(eval_r.std()))
    abl_clocks.append(wall)
    abl_nstates.append(ns)
    print(f"    Grid {grid}: n_states={ns}, mean_eval={eval_r.mean():.1f}, wall={wall:.1f}s")

plot_discretization_ablation(grid_labels, abl_returns, abl_stds,
                              abl_clocks, abl_nstates)

champion_edges = make_bins(CARTPOLE_BINS, CARTPOLE_CLAMPS)
n_s_cp = n_states(CARTPOLE_BINS)


class CartPoleFlatEnv:
    edges = champion_edges
    def __init__(self):
        self._env = gym.make("CartPole-v1")
        self.action_space      = self._env.action_space
        self.observation_space = self._env.observation_space
    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        return discretize(obs, self.edges), info
    def step(self, a):
        obs, r, term, trunc, info = self._env.step(a)
        return discretize(obs, self.edges), r, term, trunc, info
    def close(self):
        self._env.close()


print("  Building CartPole transition model for VI/PI...")
P_cp, n_s_cp, n_a_cp = build_transition_reward(
    bins=CARTPOLE_BINS, clamps=CARTPOLE_CLAMPS,
    gamma=VI_PI_CONFIG["gamma"], n_rollout=5, seed=42
)
print(f"  CartPole: {n_s_cp} states, {n_a_cp} actions")

print("  Running VI on CartPole...")
V_vi_cp, pol_vi_cp, hist_vi_cp = value_iteration(P_cp, n_s_cp, n_a_cp)
print(f"    VI: {hist_vi_cp['n_iters']} iters, {hist_vi_cp['wall_clock']:.2f}s")

print("  Running PI on CartPole...")
V_pi_cp, pol_pi_cp, hist_pi_cp = policy_iteration(P_cp, n_s_cp, n_a_cp)
print(f"    PI: {hist_pi_cp['n_iters']} iters, {hist_pi_cp['wall_clock']:.2f}s")

plot_vi_pi_convergence(hist_vi_cp, hist_pi_cp, "CartPole-v1")
plot_vi_pi_comparison(hist_vi_cp, hist_pi_cp, "CartPole-v1")

print("  Running SARSA on CartPole (5 seeds)...")
cp_sarsa_returns, cp_sarsa_dq = [], []
for seed in SEEDS:
    Q, returns, hist = sarsa(CartPoleFlatEnv, n_s_cp, n_a_cp,
                             cfg=SARSA_CONFIG, seed=seed)
    cp_sarsa_returns.append(returns)
    cp_sarsa_dq.append(hist["delta_q"])
print("    Done.")

print("  Running Q-Learning on CartPole (5 seeds)...")
cp_qlearn_returns, cp_qlearn_dq = [], []
for seed in SEEDS:
    Q, returns, hist = qlearning(CartPoleFlatEnv, n_s_cp, n_a_cp,
                                  cfg=QLEARN_CONFIG, seed=seed)
    cp_qlearn_returns.append(returns)
    cp_qlearn_dq.append(hist["delta_q"])
print("    Done.")

plot_learning_curves({"SARSA": cp_sarsa_returns,
                      "Q-Learning": cp_qlearn_returns}, "CartPole-v1")
plot_delta_q({"SARSA": cp_sarsa_dq,
              "Q-Learning": cp_qlearn_dq}, "CartPole-v1")
plot_episode_lengths({"SARSA": cp_sarsa_returns,
                      "Q-Learning": cp_qlearn_returns}, "CartPole-v1")
plot_eval_distribution({
    "SARSA":      [r for sr in cp_sarsa_returns  for r in sr[-EVAL_EPISODES:]],
    "Q-Learning": [r for sr in cp_qlearn_returns for r in sr[-EVAL_EPISODES:]],
}, "CartPole-v1")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nBlackjack-v1")
print(f"  VI: {hist_vi_bj['n_iters']} iters | {hist_vi_bj['wall_clock']:.2f}s")
print(f"  PI: {hist_pi_bj['n_iters']} iters | {hist_pi_bj['wall_clock']:.2f}s")
print(f"  SARSA  mean tail return: {np.mean([np.mean(r[-100:]) for r in bj_sarsa_returns]):.3f}")
print(f"  Q-Lrn  mean tail return: {np.mean([np.mean(r[-100:]) for r in bj_qlearn_returns]):.3f}")
print("\nCartPole-v1")
print(f"  VI: {hist_vi_cp['n_iters']} iters | {hist_vi_cp['wall_clock']:.2f}s")
print(f"  PI: {hist_pi_cp['n_iters']} iters | {hist_pi_cp['wall_clock']:.2f}s")
print(f"  SARSA  mean tail length: {np.mean([np.mean(r[-100:]) for r in cp_sarsa_returns]):.1f}")
print(f"  Q-Lrn  mean tail length: {np.mean([np.mean(r[-100:]) for r in cp_qlearn_returns]):.1f}")
print(f"\nAll figures saved to: ../output/figures/")
print(f"Seeds: {SEEDS}")