"""
discretize.py
Non-uniform binning for CartPole-v1's continuous state space.

State features:
  0: cart position      x       ∈ [-2.4,  2.4]
  1: cart velocity      x_dot   ∈ (-inf, inf) → clamp [-3, 3]
  2: pole angle         theta   ∈ [-0.2,  0.2]  (critical!)
  3: pole ang. velocity th_dot  ∈ (-inf, inf) → clamp [-3.5, 3.5]
"""

import numpy as np
from config import CARTPOLE_BINS, CARTPOLE_CLAMPS


def make_bins(bins=CARTPOLE_BINS, clamps=CARTPOLE_CLAMPS):
    """
    Build bin-edge arrays for each feature.
    Uses linspace within clamped range -- uniform within each feature.
    Returns a list of 1-D arrays (the edges).
    """
    edges = []
    for n_bins, (lo, hi) in zip(bins, clamps):
        edges.append(np.linspace(lo, hi, n_bins + 1))
    return edges


def discretize(obs, edges):
    """
    Map a continuous CartPole observation to a single integer state index.

    Parameters
    ----------
    obs   : array-like, length 4
    edges : list of 4 edge arrays (from make_bins)

    Returns
    -------
    state_idx : int
    """
    obs = np.asarray(obs, dtype=float)
    n_bins = [len(e) - 1 for e in edges]

    indices = []
    for val, edge in zip(obs, edges):
        # clamp to range then digitize
        # note: digitize returns 1-indexed so we subtract 1
        val = np.clip(val, edge[0], edge[-1])
        idx = int(np.digitize(val, edge)) - 1
        idx = np.clip(idx, 0, len(edge) - 2)
        indices.append(idx)

    # row-major ravelling to get a flat state index
    state_idx = 0
    for i, (idx, nb) in enumerate(zip(indices, n_bins)):
        stride = int(np.prod(n_bins[i + 1:]))
        state_idx += idx * stride
    return state_idx


def n_states(bins=CARTPOLE_BINS):
    """Total number of discrete states."""
    result = 1
    for b in bins:
        result *= b
    return result


def build_transition_reward(bins=CARTPOLE_BINS, clamps=CARTPOLE_CLAMPS,
                             gamma=0.99, n_rollout=3, seed=42):
    """
    Empirically estimate the transition and reward matrices P and R for VI/PI
    by rolling out the CartPole env.

    P[s, a] → list of (prob, s', reward, done) tuples  (gym-style)
    We approximate deterministic dynamics → P[s,a] = [(1.0, s', r, done)]

    Returns
    -------
    P : dict  {s: {a: [(prob, s', r, done)]}}
    n_s, n_a : int
    """
    import gymnasium as gym

    edges  = make_bins(bins, clamps)
    n_s    = n_states(bins)
    n_a    = 2   # CartPole: push left (0) or right (1)

    # bettermdptools-compatible P format
    P = {s: {a: [] for a in range(n_a)} for s in range(n_s)}
    visited = {s: {a: False for a in range(n_a)} for s in range(n_s)}

    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")

    # total steps is just n_s * n_a * n_rollout, probably overkill
    # but want decent coverage before we start solving
    total_steps = n_s * n_a * n_rollout
    step_count  = 0

    # DEBUG -- sanity check state space size looks right before we commit to the rollout
    # print(f"[build_transition_reward] n_s={n_s}, n_a={n_a}, total_steps={total_steps}")

    while step_count < total_steps:
        obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        done   = False
        while not done and step_count < total_steps:
            s = discretize(obs, edges)
            a = int(rng.integers(0, n_a))
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s_next = discretize(next_obs, edges)
            # only store first visit so the model stays deterministic
            if not visited[s][a]:
                P[s][a] = [(1.0, s_next, float(reward), bool(terminated))]
                visited[s][a] = True
            obs = next_obs
            step_count += 1

    env.close()

    # fill any unvisited (s,a) pairs with self-loop, zero reward
    # this shouldnt happen much if n_rollout is reasonable
    n_unvisited = 0
    for s in range(n_s):
        for a in range(n_a):
            if not P[s][a]:
                P[s][a] = [(1.0, s, 0.0, False)]
                n_unvisited += 1

    if n_unvisited > 0:
        print(f"  [warn] {n_unvisited} unvisited (s,a) pairs filled with self-loops")

    return P, n_s, n_a