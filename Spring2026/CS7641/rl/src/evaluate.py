"""
evaluate.py
Post-training policy evaluation: runs the greedy policy for N episodes
and reports mean ± std return and mean episode length.
"""

import numpy as np
from config import EVAL_EPISODES, MAX_STEPS


def evaluate_q_policy(env_fn, Q, n_episodes=EVAL_EPISODES,
                       max_steps=MAX_STEPS, seed=0,
                       state_fn=None):
    """
    Evaluate the greedy policy derived from Q.

    Parameters
    ----------
    env_fn     : callable → gymnasium.Env
    Q          : np.ndarray (n_s, n_a)
    n_episodes : int
    max_steps  : int
    seed       : int
    state_fn   : optional callable obs→int for continuous envs

    Returns
    -------
    returns  : np.ndarray (n_episodes,)
    lengths  : np.ndarray (n_episodes,)
    """
    rng = np.random.default_rng(seed)
    env = env_fn()

    returns = []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        s      = state_fn(obs) if state_fn else (obs if isinstance(obs, int) else int(obs))
        total_r = 0.0
        length  = 0

        for _ in range(max_steps):
            # pure greedy -- no exploration during eval
            a = int(np.argmax(Q[s]))
            obs, r, terminated, truncated, _ = env.step(a)
            s        = state_fn(obs) if state_fn else (obs if isinstance(obs, int) else int(obs))
            total_r += r
            length  += 1
            if terminated or truncated:
                break

        returns.append(total_r)
        lengths.append(length)

    env.close()
    return np.array(returns), np.array(lengths)


def evaluate_vi_pi_policy(env_fn, policy, n_episodes=EVAL_EPISODES,
                           max_steps=MAX_STEPS, seed=0, state_fn=None):
    """
    Evaluate a deterministic policy array π (indexed by state).
    Same signature as evaluate_q_policy for convenience.
    """
    rng = np.random.default_rng(seed)
    env = env_fn()

    returns = []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        s      = state_fn(obs) if state_fn else (obs if isinstance(obs, int) else int(obs))
        total_r = 0.0
        length  = 0

        for _ in range(max_steps):
            a = int(policy[s])
            obs, r, terminated, truncated, _ = env.step(a)
            s        = state_fn(obs) if state_fn else (obs if isinstance(obs, int) else int(obs))
            total_r += r
            length  += 1
            if terminated or truncated:
                break

        returns.append(total_r)
        lengths.append(length)

    env.close()
    return np.array(returns), np.array(lengths)


def aggregate_returns(all_returns):
    """
    Given a list of per-seed return arrays (each shape (n_episodes,)),
    return mean, lower CI, upper CI arrays (95% CI via percentile).

    Returns
    -------
    mean  : np.ndarray
    lo    : np.ndarray (2.5th percentile across seeds)
    hi    : np.ndarray (97.5th percentile across seeds)
    """
    mat  = np.stack(all_returns, axis=0)   # (n_seeds, n_episodes)
    mean = mat.mean(axis=0)
    # using percentile rather than ±1.96*std bc the distributions
    # arent really normal, especially early in training
    lo   = np.percentile(mat, 2.5,  axis=0)
    hi   = np.percentile(mat, 97.5, axis=0)
    return mean, lo, hi


def smooth(x, window=50):
    """Simple moving average for plotting.
    Returns shorter array than input -- caller needs to handle the length difference."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")