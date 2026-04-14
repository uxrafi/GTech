"""
hyperparameter_search.py

Staged hyperparameter search for SARSA and Q-Learning:
  Stage 1: N=24 random candidates, 200 pilot episodes → keep top 8
  Stage 2: 400 more episodes for top 8               → keep top 3
  Stage 3: Local refinement ±2× on α, ±25% on decay horizon

Returns champion config for each algorithm.
"""

import copy
import numpy as np
from config import SEARCH_CONFIG, SARSA_CONFIG, QLEARN_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Candidate sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_candidates(n, rng, base_cfg, double_q=False):
    # log-uniform for alpha since we want to explore small values too
    cfg = SEARCH_CONFIG
    candidates = []
    for _ in range(n):
        c = copy.deepcopy(base_cfg)
        c["alpha_0"]         = float(10 ** rng.uniform(*np.log10(cfg["alpha_range"])))
        c["alpha_min"]       = c["alpha_0"] * rng.uniform(0.1, 0.5)
        c["eps_min"]         = float(rng.uniform(*cfg["eps_floor_range"]))
        c["eps_decay_steps"] = int(rng.uniform(*cfg["decay_range"]))
        c["gamma"]           = float(rng.uniform(*cfg["gamma_range"]))
        if double_q:
            c["double_q"] = True
        candidates.append(c)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper: run one config for n_episodes, return mean return
# ─────────────────────────────────────────────────────────────────────────────

def _eval_config(train_fn, env_fn, n_s, n_a, cfg, n_episodes, seed=42):
    """
    train_fn : sarsa or qlearning function
    Returns mean episodic return over last 20% of episodes.
    """
    cfg_run = copy.deepcopy(cfg)
    cfg_run["n_episodes"] = n_episodes
    _, returns, _ = train_fn(env_fn, n_s, n_a, cfg=cfg_run, seed=seed)
    # last 20% of episodes -- gives a better signal than the full run
    # since early episodes are mostly random exploration
    tail = max(1, len(returns) // 5)
    return float(np.mean(returns[-tail:]))


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def staged_search(train_fn, env_fn, n_s, n_a,
                  algo="sarsa", seed=0, verbose=True):
    """
    Run 3-stage hyperparameter search.

    Parameters
    ----------
    train_fn : sarsa.sarsa or qlearning.qlearning
    env_fn   : callable → gymnasium.Env
    n_s, n_a : state/action space sizes
    algo     : "sarsa" | "qlearning"
    seed     : int (for candidate sampling)
    verbose  : bool

    Returns
    -------
    champion_cfg : dict — best hyperparameter configuration found
    search_log   : list of dicts for each stage
    """
    cfg      = SEARCH_CONFIG
    rng      = np.random.default_rng(seed)
    base_cfg = copy.deepcopy(SARSA_CONFIG if algo == "sarsa" else QLEARN_CONFIG)
    double_q = (algo == "qlearning")

    log = []

    # ── Stage 1: Random search ──────────────────────────────────────────────
    # wide random sweep first -- no point being clever before we know the landscape
    candidates = _sample_candidates(cfg["n_candidates"], rng, base_cfg, double_q)
    scores1 = []
    for i, c in enumerate(candidates):
        score = _eval_config(train_fn, env_fn, n_s, n_a, c,
                              cfg["pilot_episodes"], seed=seed + i)
        scores1.append(score)
        if verbose:
            print(f"  Stage1 [{i+1}/{cfg['n_candidates']}]  score={score:.3f}")

    top_k1_idx = np.argsort(scores1)[::-1][:cfg["top_k1"]]
    top_k1     = [candidates[i] for i in top_k1_idx]
    log.append({"stage": 1, "scores": scores1, "top_k": top_k1_idx.tolist()})

    # ── Stage 2: Successive Halving ─────────────────────────────────────────
    # give the top-k more budget and re-rank -- some configs need more
    # episodes before they settle, so 200 pilot eps can be misleading
    scores2 = []
    for i, c in enumerate(top_k1):
        score = _eval_config(train_fn, env_fn, n_s, n_a, c,
                              cfg["pilot_episodes"] + cfg["stage2_episodes"],
                              seed=seed + 100 + i)
        scores2.append(score)
        if verbose:
            print(f"  Stage2 [{i+1}/{cfg['top_k1']}]  score={score:.3f}")

    top_k2_idx = np.argsort(scores2)[::-1][:cfg["top_k2"]]
    top_k2     = [top_k1[i] for i in top_k2_idx]
    log.append({"stage": 2, "scores": scores2, "top_k": top_k2_idx.tolist()})

    # ── Stage 3: Local refinement ────────────────────────────────────────────
    # small perturbations around the best config from stage 2
    # ±2x on alpha and ±25% on decay horizon usually enough
    champion = top_k2[0]
    local_candidates = [champion]

    for mult_alpha in [0.5, 2.0]:
        for mult_decay in [0.75, 1.25]:
            c = copy.deepcopy(champion)
            c["alpha_0"]         = np.clip(c["alpha_0"] * mult_alpha, 1e-4, 1.0)
            c["alpha_min"]       = np.clip(c["alpha_min"] * mult_alpha, 1e-5, c["alpha_0"])
            c["eps_decay_steps"] = int(c["eps_decay_steps"] * mult_decay)
            local_candidates.append(c)

    scores3 = []
    for i, c in enumerate(local_candidates):
        score = _eval_config(train_fn, env_fn, n_s, n_a, c,
                              cfg["pilot_episodes"] + cfg["stage2_episodes"],
                              seed=seed + 200 + i)
        scores3.append(score)
        if verbose:
            print(f"  Stage3 [{i+1}/{len(local_candidates)}]  score={score:.3f}")

    best_local_idx = int(np.argmax(scores3))
    champion_cfg   = local_candidates[best_local_idx]
    log.append({"stage": 3, "scores": scores3, "best_idx": best_local_idx})

    # DEBUG -- uncomment if stage3 keeps picking the original config back
    # print(f"  [debug] stage3 scores: {scores3}")
    # print(f"  [debug] best_local_idx={best_local_idx}, same as input? {best_local_idx == 0}")

    if verbose:
        print(f"\n  Champion ({algo}): "
              f"α={champion_cfg['alpha_0']:.4f}, "
              f"ε_decay={champion_cfg['eps_decay_steps']}, "
              f"γ={champion_cfg['gamma']:.4f}")

    return champion_cfg, log