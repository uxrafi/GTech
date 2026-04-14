"""
vi_pi.py
Value Iteration (VI) and Policy Iteration (PI) using vectorized numpy.
Works with any Gym-style P dict: P[s][a] = [(prob, s', r, done), ...]

Assumes deterministic transitions (one tuple per (s,a)) for speed.
Stochastic envs (Blackjack) still work correctly via fallback loop.
"""

import time
import numpy as np
from config import VI_PI_CONFIG


def _build_arrays(P, n_s, n_a):
    """
    Convert P dict to numpy arrays for vectorized computation.
    Handles both deterministic (1 tuple) and stochastic (multiple tuples).

    Returns
    -------
    T : (n_s, n_a) int   — next state (for deterministic)
    R : (n_s, n_a) float — expected reward
    D : (n_s, n_a) bool  — terminal flag
    is_det : bool        — True if all (s,a) have exactly 1 transition
    """
    T = np.zeros((n_s, n_a), dtype=int)
    R = np.zeros((n_s, n_a), dtype=float)
    D = np.zeros((n_s, n_a), dtype=bool)
    is_det = True

    for s in range(n_s):
        for a in range(n_a):
            transitions = P[s][a]
            if len(transitions) != 1:
                is_det = False
            # Expected values across transitions
            exp_r    = sum(p * r    for p, _, r, _    in transitions)
            exp_done = sum(p * done for p, _, _, done in transitions) > 0.5
            # Weighted next state (only meaningful for deterministic)
            next_s   = transitions[0][1] if transitions else s
            T[s, a]  = next_s
            R[s, a]  = exp_r
            D[s, a]  = exp_done

    return T, R, D, is_det


def _bellman_q_stochastic(P, V, s, a, gamma):
    """Expected Q-value for stochastic transitions."""
    return sum(
        prob * (r + gamma * (0.0 if done else V[s_]))
        for prob, s_, r, done in P[s][a]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Value Iteration
# ─────────────────────────────────────────────────────────────────────────────

def value_iteration(P, n_s, n_a,
                    gamma=None, theta=None, consecutive_m=None, max_iter=None):
    """
    Vectorized Value Iteration.

    Returns
    -------
    V       : np.ndarray (n_s,)
    policy  : np.ndarray (n_s,)
    history : dict
    """
    cfg           = VI_PI_CONFIG
    gamma         = gamma         or cfg["gamma"]
    theta         = theta         or cfg["theta"]
    consecutive_m = consecutive_m or cfg["consecutive_m"]
    max_iter      = max_iter      or cfg["max_iter"]

    T, R, D, is_det = _build_arrays(P, n_s, n_a)
    V          = np.zeros(n_s)
    delta_hist = []
    consec     = 0
    t0         = time.perf_counter()

    for it in range(max_iter):
        if is_det:
            # Fully vectorized: Q[s,a] = R[s,a] + gamma * V[T[s,a]] * (not done)
            Q     = R + gamma * V[T] * (~D)
            V_new = Q.max(axis=1)
        else:
            # Stochastic fallback (Blackjack)
            V_new = np.zeros(n_s)
            for s in range(n_s):
                V_new[s] = max(
                    _bellman_q_stochastic(P, V, s, a, gamma)
                    for a in range(n_a)
                )

        delta = float(np.max(np.abs(V_new - V)))
        V     = V_new
        delta_hist.append(delta)

        if delta < theta:
            consec += 1
            if consec >= consecutive_m:
                break
        else:
            consec = 0

    # Extract greedy policy
    if is_det:
        Q      = R + gamma * V[T] * (~D)
        policy = Q.argmax(axis=1).astype(int)
    else:
        policy = np.array([
            max(range(n_a), key=lambda a: _bellman_q_stochastic(P, V, s, a, gamma))
            for s in range(n_s)
        ], dtype=int)

    return V, policy, {
        "delta_v":    delta_hist,
        "wall_clock": time.perf_counter() - t0,
        "n_iters":    it + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Policy Iteration
# ─────────────────────────────────────────────────────────────────────────────

def policy_iteration(P, n_s, n_a,
                     gamma=None, theta=None, consecutive_m=None, max_iter=None):
    """
    Vectorized Policy Iteration.

    Returns
    -------
    V       : np.ndarray (n_s,)
    policy  : np.ndarray (n_s,)
    history : dict
    """
    cfg      = VI_PI_CONFIG
    gamma    = gamma    or cfg["gamma"]
    theta    = theta    or cfg["theta"]
    max_iter = max_iter or cfg["max_iter"]

    T, R, D, is_det = _build_arrays(P, n_s, n_a)
    policy     = np.zeros(n_s, dtype=int)
    V          = np.zeros(n_s)
    delta_hist = []
    t0         = time.perf_counter()

    for it in range(max_iter):
        # ── Policy Evaluation (vectorized, iterate to convergence) ────────────
        for _ in range(500):
            sa_idx = (np.arange(n_s), policy)
            if is_det:
                V_new = R[sa_idx] + gamma * V[T[sa_idx]] * (~D[sa_idx])
            else:
                V_new = np.array([
                    _bellman_q_stochastic(P, V, s, policy[s], gamma)
                    for s in range(n_s)
                ])
            eval_delta = float(np.max(np.abs(V_new - V)))
            V = V_new
            if eval_delta < theta:
                break

        # ── Policy Improvement ────────────────────────────────────────────────
        if is_det:
            Q = R + gamma * V[T] * (~D)
        else:
            Q = np.array([
                [_bellman_q_stochastic(P, V, s, a, gamma) for a in range(n_a)]
                for s in range(n_s)
            ])

        new_policy = Q.argmax(axis=1).astype(int)
        delta      = float(np.max(np.abs(Q.max(axis=1) - V)))
        delta_hist.append(delta)

        if np.all(new_policy == policy):
            policy = new_policy
            break
        policy = new_policy

    return V, policy, {
        "delta_v":    delta_hist,
        "wall_clock": time.perf_counter() - t0,
        "n_iters":    it + 1,
    }