"""
sarsa.py
Tabular SARSA (on-policy TD control) with ε-greedy exploration.

Update rule:
  Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]
where a' is chosen by the same ε-greedy policy used to act.
"""

import time
import numpy as np
from config import SARSA_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _epsilon(step, eps_0, eps_min, eps_decay_steps):
    """Linear ε decay from eps_0 to eps_min over eps_decay_steps."""
    fraction = min(step / eps_decay_steps, 1.0)
    return eps_0 + fraction * (eps_min - eps_0)


def _alpha_linear(episode, n_episodes, alpha_0, alpha_min):
    """Linear α decay from alpha_0 to alpha_min."""
    fraction = min(episode / n_episodes, 1.0)
    return alpha_0 + fraction * (alpha_min - alpha_0)


def _alpha_visit(visits, alpha_0=1.0):
    """Visit-count based step size: α = 1 / (1 + visits).
    Theoretically nice but in practice converges slower on these envs."""
    return alpha_0 / (1.0 + visits)


def eps_greedy(Q, s, n_a, eps, rng):
    """ε-greedy action selection."""
    if rng.random() < eps:
        return int(rng.integers(0, n_a))
    return int(np.argmax(Q[s]))


# ─────────────────────────────────────────────────────────────────────────────
# SARSA
# ─────────────────────────────────────────────────────────────────────────────

def sarsa(env_fn, n_s, n_a, cfg=None, seed=42):
    """
    Run tabular SARSA on environment produced by env_fn().

    Parameters
    ----------
    env_fn : callable → gymnasium.Env
    n_s    : number of discrete states (int)
    n_a    : number of actions (int)
    cfg    : dict of hyperparameters (defaults from config.SARSA_CONFIG)
    seed   : int

    Returns
    -------
    Q           : np.ndarray shape (n_s, n_a)
    episode_returns : list[float]   — per-episode undiscounted return
    history     : dict with 'delta_q', 'wall_clock', 'n_episodes'
    """
    if cfg is None:
        cfg = SARSA_CONFIG

    gamma           = cfg["gamma"]
    alpha_0         = cfg["alpha_0"]
    alpha_min       = cfg["alpha_min"]
    eps_0           = cfg["eps_0"]
    eps_min         = cfg["eps_min"]
    eps_decay_steps = cfg["eps_decay_steps"]
    n_episodes      = cfg["n_episodes"]
    use_visit_alpha = cfg.get("use_visit_alpha", False)

    rng       = np.random.default_rng(seed)
    Q         = np.zeros((n_s, n_a))
    visits    = np.zeros((n_s, n_a), dtype=int)
    env       = env_fn()

    episode_returns = []
    delta_q_hist    = []
    global_step     = 0
    t0              = time.perf_counter()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        s      = obs if isinstance(obs, int) else int(obs)
        # select first action before the loop -- SARSA needs (s,a) pair upfront
        eps    = _epsilon(global_step, eps_0, eps_min, eps_decay_steps)
        a      = eps_greedy(Q, s, n_a, eps, rng)

        ep_return = 0.0
        max_dq    = 0.0
        done      = False

        while not done:
            next_obs, r, terminated, truncated, _ = env.step(a)
            done   = terminated or truncated
            s_next = next_obs if isinstance(next_obs, int) else int(next_obs)
            eps    = _epsilon(global_step, eps_0, eps_min, eps_decay_steps)
            # pick a_next here so we use it in both the update and the next iteration
            a_next = eps_greedy(Q, s_next, n_a, eps, rng)

            # ── α schedule ──
            if use_visit_alpha:
                visits[s, a] += 1
                alpha = _alpha_visit(visits[s, a], alpha_0)
            else:
                alpha = _alpha_linear(ep, n_episodes, alpha_0, alpha_min)

            # ── SARSA update ──
            # bootstrap off Q(s', a') not max_a Q(s', a) -- thats the key difference vs Q-learning
            td_target = r + gamma * (0.0 if done else Q[s_next, a_next])
            td_error  = td_target - Q[s, a]
            dq        = abs(alpha * td_error)
            Q[s, a]  += alpha * td_error
            max_dq    = max(max_dq, dq)

            ep_return   += r
            s, a         = s_next, a_next
            global_step += 1

        episode_returns.append(ep_return)
        delta_q_hist.append(max_dq)

    env.close()
    wall_clock = time.perf_counter() - t0

    return Q, episode_returns, {
        "delta_q":    delta_q_hist,
        "wall_clock": wall_clock,
        "n_episodes": n_episodes,
    }