"""
qlearning.py
Tabular Q-Learning (off-policy) with optional Double Q-Learning.

Standard update rule:
  Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]

Double Q-Learning (two tables, alternating targets):
  With prob 0.5:  Q1(s,a) ← Q1(s,a) + α [ r + γ Q2(s', argmax_a' Q1(s',a')) − Q1(s,a) ]
  otherwise:      Q2(s,a) ← Q2(s,a) + α [ r + γ Q1(s', argmax_a' Q2(s',a')) − Q2(s,a) ]
"""

import time
import numpy as np
from config import QLEARN_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (same schedule helpers as sarsa.py, duplicated for independence)
# ─────────────────────────────────────────────────────────────────────────────

def _epsilon(step, eps_0, eps_min, eps_decay_steps):
    fraction = min(step / eps_decay_steps, 1.0)
    return eps_0 + fraction * (eps_min - eps_0)


def _alpha_linear(episode, n_episodes, alpha_0, alpha_min):
    fraction = min(episode / n_episodes, 1.0)
    return alpha_0 + fraction * (alpha_min - alpha_0)


def _alpha_visit(visits, alpha_0=1.0):
    return alpha_0 / (1.0 + visits)


def eps_greedy(Q, s, n_a, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, n_a))
    return int(np.argmax(Q[s]))


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning
# ─────────────────────────────────────────────────────────────────────────────

def qlearning(env_fn, n_s, n_a, cfg=None, seed=42):
    """
    Run tabular Q-Learning on environment produced by env_fn().

    Parameters
    ----------
    env_fn  : callable → gymnasium.Env
    n_s     : number of discrete states (int)
    n_a     : number of actions (int)
    cfg     : dict of hyperparameters (defaults from config.QLEARN_CONFIG)
    seed    : int

    Returns
    -------
    Q               : np.ndarray shape (n_s, n_a)  — Q1 if double_q
    episode_returns : list[float]
    history         : dict with 'delta_q', 'wall_clock', 'n_episodes'
    """
    if cfg is None:
        cfg = QLEARN_CONFIG

    gamma           = cfg["gamma"]
    alpha_0         = cfg["alpha_0"]
    alpha_min       = cfg["alpha_min"]
    eps_0           = cfg["eps_0"]
    eps_min         = cfg["eps_min"]
    eps_decay_steps = cfg["eps_decay_steps"]
    n_episodes      = cfg["n_episodes"]
    double_q        = cfg.get("double_q", False)
    use_visit_alpha = cfg.get("use_visit_alpha", False)

    rng    = np.random.default_rng(seed)
    Q1     = np.zeros((n_s, n_a))
    Q2     = np.zeros((n_s, n_a)) if double_q else None
    visits = np.zeros((n_s, n_a), dtype=int)
    env    = env_fn()

    episode_returns = []
    delta_q_hist    = []
    global_step     = 0
    t0              = time.perf_counter()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        s      = obs if isinstance(obs, int) else int(obs)

        ep_return = 0.0
        max_dq    = 0.0
        done      = False

        while not done:
            eps = _epsilon(global_step, eps_0, eps_min, eps_decay_steps)

            # for double Q, act greedy w.r.t Q1+Q2 sum -- less biased than either alone
            if double_q:
                Q_act = Q1 + Q2
                a     = eps_greedy(Q_act, s, n_a, eps, rng)
            else:
                a = eps_greedy(Q1, s, n_a, eps, rng)

            next_obs, r, terminated, truncated, _ = env.step(a)
            done   = terminated or truncated
            s_next = next_obs if isinstance(next_obs, int) else int(next_obs)

            # ── α schedule ──
            if use_visit_alpha:
                visits[s, a] += 1
                alpha = _alpha_visit(visits[s, a], alpha_0)
            else:
                alpha = _alpha_linear(ep, n_episodes, alpha_0, alpha_min)

            if double_q:
                # ── Double Q update ──
                # alternating which table gets updated reduces maximization bias
                if rng.random() < 0.5:
                    # update Q1, evaluate with Q2
                    a_star    = int(np.argmax(Q1[s_next]))
                    td_target = r + gamma * (0.0 if done else Q2[s_next, a_star])
                    td_error  = td_target - Q1[s, a]
                    dq        = abs(alpha * td_error)
                    Q1[s, a] += alpha * td_error
                else:
                    # update Q2, evaluate with Q1
                    a_star    = int(np.argmax(Q2[s_next]))
                    td_target = r + gamma * (0.0 if done else Q1[s_next, a_star])
                    td_error  = td_target - Q2[s, a]
                    dq        = abs(alpha * td_error)
                    Q2[s, a] += alpha * td_error
            else:
                # ── Standard Q-Learning ──
                # max over next state -- this is what makes it off-policy
                td_target = r + gamma * (0.0 if done else np.max(Q1[s_next]))
                td_error  = td_target - Q1[s, a]
                dq        = abs(alpha * td_error)
                Q1[s, a] += alpha * td_error

            max_dq       = max(max_dq, dq)
            ep_return   += r
            s            = s_next
            global_step += 1

        episode_returns.append(ep_return)
        delta_q_hist.append(max_dq)

    env.close()
    wall_clock = time.perf_counter() - t0

    return Q1, episode_returns, {
        "delta_q":    delta_q_hist,
        "wall_clock": wall_clock,
        "n_episodes": n_episodes,
    }