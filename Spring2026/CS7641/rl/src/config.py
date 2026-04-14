"""
config.py
Central configuration: seeds, hyperparameters, paths.
"""
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
# using 5 seeds so results aren't just lucky -- averaged across these
SEEDS = [42, 7, 13, 99, 2024]

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(ROOT, "output", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── VI / PI ───────────────────────────────────────────────────────────────────
VI_PI_CONFIG = {
    "gamma":          0.99,
    "theta":          1e-6,       # convergence threshold (relaxed for speed)
    "consecutive_m":  3,          # how many sweeps below theta before we call it converged
    "max_iter":       500,        # hard cap -- PI normally converges way before this
}

# ── CartPole Discretization ───────────────────────────────────────────────────
# Bins: (x, x_dot, theta, theta_dot)
# theta and theta_dot get more bins bc thats where most of the
# interesting dynamics actually are
CARTPOLE_BINS    = (3, 3, 8, 12)
CARTPOLE_CLAMPS  = [(-2.4, 2.4), (-3.0, 3.0), (-0.2, 0.2), (-3.5, 3.5)]

# ablation grids for discretization study
# last one is the one we went with
CARTPOLE_GRIDS = [
    (1, 1, 6,  6),
    (1, 1, 6, 12),
    (3, 3, 6, 12),
    (3, 3, 8, 12),   # champion
    (5, 5, 10, 16),
]

# ── SARSA champion hyperparameters (post-search) ──────────────────────────────
# these came out of the staged search, dont change without re-running search
SARSA_CONFIG = {
    "gamma":          0.99,
    "alpha_0":        0.5,
    "alpha_min":      0.1,
    "alpha_decay":    "linear",   # linear decay over n_episodes
    "eps_0":          1.0,
    "eps_min":        0.01,
    "eps_decay_steps":10_000,
    "n_episodes":     5_000,
    "use_visit_alpha":False,      # if True, alpha = 1/(1+visits) -- tried it, wasnt better
}

# ── Q-Learning champion hyperparameters (post-search) ────────────────────────
# same structure as SARSA config, kept separate so we can tune independently
QLEARN_CONFIG = {
    "gamma":          0.99,
    "alpha_0":        0.5,
    "alpha_min":      0.1,
    "alpha_decay":    "linear",
    "eps_0":          1.0,
    "eps_min":        0.01,
    "eps_decay_steps":10_000,
    "n_episodes":     5_000,
    "double_q":       False,      # set True for Double Q-Learning variant
    "use_visit_alpha":False,
}

# ── Hyperparameter search budget ──────────────────────────────────────────────
# stage 1 casts a wide net, stage 2 is basically successive halving
# 24 candidates felt like enough given the search space isnt that huge
SEARCH_CONFIG = {
    "n_candidates":   24,
    "pilot_episodes": 200,
    "stage2_episodes":400,
    "top_k1":         8,
    "top_k2":         3,
    # search ranges (log-uniform where noted)
    "alpha_range":     (0.001, 1.0),    # log-uniform
    "eps_floor_range": (0.005, 0.05),   # uniform
    "decay_range":     (2_000, 20_000), # uniform (steps)
    "gamma_range":     (0.95, 0.999),   # uniform
}

# ── Evaluation ────────────────────────────────────────────────────────────────
# 200 episodes for eval felt reasonable, variance is still a bit high
# but running more would slow down the whole pipeline a lot
EVAL_EPISODES = 200    # episodes for policy evaluation after training
MAX_STEPS     = 500    # env step cap per episode