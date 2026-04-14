"""
plots.py
All figure generation for the RL report.
Saves PNGs to output/figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR
from evaluate import smooth

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
FIGSIZE = (7, 4)


def _savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# VI / PI convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_vi_pi_convergence(vi_hist, pi_hist, env_name):
    """
    Plot ΔV vs iteration for VI and PI side by side.
    semilogy so the exponential decay is visible -- linear scale just looks like
    it drops to zero immediately
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, hist, label in zip(axes,
                                [vi_hist, pi_hist],
                                ["Value Iteration", "Policy Iteration"]):
        iters = np.arange(1, len(hist["delta_v"]) + 1)
        ax.semilogy(iters, hist["delta_v"], linewidth=1.8, color="steelblue")
        ax.axhline(1e-8, linestyle="--", color="tomato", linewidth=1, label="θ=1e-8")
        ax.set_title(f"{label} — {env_name}\n"
                     f"({hist['n_iters']} iters, {hist['wall_clock']:.2f}s)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("max|ΔV|")
        ax.legend(fontsize=9)

    fig.suptitle(f"VI vs PI Convergence Diagnostics ({env_name})", fontweight="bold")
    plt.tight_layout()
    _savefig(f"vi_pi_convergence_{env_name.lower().replace('-','_')}.png")


def plot_vi_pi_comparison(vi_hist, pi_hist, env_name):
    """Bar chart comparing iterations and wall-clock for VI vs PI."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    labels = ["Value Iteration", "Policy Iteration"]
    iters  = [vi_hist["n_iters"], pi_hist["n_iters"]]
    times  = [vi_hist["wall_clock"], pi_hist["wall_clock"]]
    colors = ["steelblue", "darkorange"]

    axes[0].bar(labels, iters, color=colors, width=0.5)
    axes[0].set_ylabel("Iterations to Converge")
    axes[0].set_title("Iterations")

    axes[1].bar(labels, times, color=colors, width=0.5)
    axes[1].set_ylabel("Wall-Clock (s)")
    axes[1].set_title("Wall-Clock Time")

    fig.suptitle(f"VI vs PI Summary ({env_name})", fontweight="bold")
    plt.tight_layout()
    _savefig(f"vi_pi_summary_{env_name.lower().replace('-','_')}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Learning curves (SARSA / Q-Learning) — multi-seed
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(all_returns_dict, env_name, window=50):
    """
    Plot smoothed learning curves for each algorithm.

    all_returns_dict : {"SARSA": [[returns_seed0], ...], "Q-Learning": [...]}
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    palette = {"SARSA": "steelblue", "Q-Learning": "darkorange", "Double Q": "seagreen"}

    for algo, seed_returns in all_returns_dict.items():
        # smooth each seed separately then stack -- dont smooth the mean directly
        mat = np.stack([smooth(r, window) for r in seed_returns], axis=0)
        ep  = np.arange(mat.shape[1])
        mean = mat.mean(axis=0)
        lo   = np.percentile(mat, 2.5,  axis=0)
        hi   = np.percentile(mat, 97.5, axis=0)
        color = palette.get(algo, None)
        ax.plot(ep, mean, label=algo, linewidth=1.8, color=color)
        ax.fill_between(ep, lo, hi, alpha=0.2, color=color)

    ax.set_xlabel(f"Episode (smoothed window={window})")
    ax.set_ylabel("Undiscounted Return")
    ax.set_title(f"Learning Curves — {env_name}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _savefig(f"learning_curves_{env_name.lower().replace('-','_')}.png")


def plot_delta_q(all_dq_dict, env_name, window=50):
    """Plot max|ΔQ| per episode -- useful for checking if Q values are still changing."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    palette = {"SARSA": "steelblue", "Q-Learning": "darkorange"}

    for algo, seed_dq in all_dq_dict.items():
        mat  = np.stack([smooth(dq, window) for dq in seed_dq], axis=0)
        ep   = np.arange(mat.shape[1])
        mean = mat.mean(axis=0)
        lo   = np.percentile(mat, 2.5,  axis=0)
        hi   = np.percentile(mat, 97.5, axis=0)
        color = palette.get(algo, None)
        # add small epsilon before log to avoid log(0) issues
        ax.semilogy(ep, mean + 1e-10, label=algo, linewidth=1.8, color=color)
        ax.fill_between(ep, lo + 1e-10, hi + 1e-10, alpha=0.2, color=color)

    ax.set_xlabel(f"Episode (smoothed window={window})")
    ax.set_ylabel("max|ΔQ| (log scale)")
    ax.set_title(f"Q-Value Convergence Diagnostics — {env_name}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _savefig(f"delta_q_{env_name.lower().replace('-','_')}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Blackjack policy heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_blackjack_policy(Q, title="Blackjack Policy"):
    """
    Heatmap of the greedy policy derived from Q for Blackjack-v1.

    Blackjack-v1 state: (player_sum, dealer_card, usable_ace)
    player_sum  ∈ [4, 21]  (18 values)
    dealer_card ∈ [1, 10]  (10 values)
    usable_ace  ∈ {0, 1}   (2 values)
    Actions: 0=stick, 1=hit
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles    = ["No Usable Ace", "Usable Ace"]

    for ace_idx, (ax, subtitle) in enumerate(zip(axes, titles)):
        player_sums  = np.arange(4, 22)    # rows
        dealer_cards = np.arange(1, 11)    # cols
        grid = np.zeros((len(player_sums), len(dealer_cards)), dtype=int)

        for i, ps in enumerate(player_sums):
            for j, dc in enumerate(dealer_cards):
                # flat index: ps * 11 * 2 + dc * 2 + ace
                # Blackjack-v1 obs_space: Tuple(Discrete(32), Discrete(11), Discrete(2))
                s      = ps * 11 * 2 + dc * 2 + ace_idx
                grid[i, j] = int(np.argmax(Q[s]))

        cmap = matplotlib.colors.ListedColormap(["#d9534f", "#5cb85c"])
        sns.heatmap(grid, ax=ax,
                    xticklabels=dealer_cards,
                    yticklabels=player_sums,
                    cmap=cmap, vmin=0, vmax=1,
                    linewidths=0.3, linecolor="gray",
                    cbar_kws={"ticks": [0.25, 0.75]})
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_title(subtitle)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticklabels(["Stick (0)", "Hit (1)"])

    fig.suptitle(f"{title}", fontweight="bold")
    plt.tight_layout()
    _savefig("blackjack_policy_heatmap.png")


def plot_blackjack_value(V_vi, V_pi, n_s=704):
    """
    Side-by-side value function heatmaps for VI and PI on Blackjack (no usable ace).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    player_sums  = np.arange(4, 22)
    dealer_cards = np.arange(1, 11)

    for ax, V, label in zip(axes, [V_vi, V_pi], ["Value Iteration", "Policy Iteration"]):
        grid = np.zeros((len(player_sums), len(dealer_cards)))
        for i, ps in enumerate(player_sums):
            for j, dc in enumerate(dealer_cards):
                s          = ps * 11 * 2 + dc * 2 + 0   # no usable ace
                # bounds check -- state space can be slightly different depending on wrapper
                grid[i, j] = V[s] if s < len(V) else 0.0

        sns.heatmap(grid, ax=ax,
                    xticklabels=dealer_cards,
                    yticklabels=player_sums,
                    cmap="RdYlGn", center=0,
                    linewidths=0.2, linecolor="gray")
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_title(f"{label} — Value Function\n(No Usable Ace)")

    fig.suptitle("Blackjack Value Functions: VI vs PI", fontweight="bold")
    plt.tight_layout()
    _savefig("blackjack_value_heatmaps.png")


# ─────────────────────────────────────────────────────────────────────────────
# CartPole discretization ablation
# ─────────────────────────────────────────────────────────────────────────────

def plot_discretization_ablation(grid_labels, mean_returns, std_returns,
                                  wall_clocks, n_states_list):
    """
    Two-panel: (a) mean eval return vs grid label, (b) wall-clock vs grid.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(grid_labels))

    axes[0].bar(x, mean_returns, yerr=std_returns, color="steelblue",
                width=0.6, capsize=5, error_kw={"elinewidth": 1.5})
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(grid_labels, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean Eval Return (200 eps)")
    axes[0].set_title("Discretization vs. Performance")

    # secondary axis for n_states -- want to show the tradeoff between
    # state space size and compute
    axes[1].bar(x, wall_clocks, color="darkorange", width=0.6)
    ax2 = axes[1].twinx()
    ax2.plot(x, n_states_list, "k--o", markersize=5, linewidth=1.5)
    ax2.set_ylabel("N States", color="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(grid_labels, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Wall-Clock (s)")
    axes[1].set_title("Compute vs. Grid Size")

    fig.suptitle("CartPole Discretization Ablation Study", fontweight="bold")
    plt.tight_layout()
    _savefig("cartpole_discretization_ablation.png")


def plot_episode_lengths(all_lengths_dict, env_name, window=50):
    """
    CartPole-specific: episode length over training as proxy for balance time.
    Max is 500 steps so anything near that is basically a solved episode.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    palette = {"SARSA": "steelblue", "Q-Learning": "darkorange",
               "VI Policy": "seagreen", "PI Policy": "mediumpurple"}

    for algo, seed_lengths in all_lengths_dict.items():
        mat  = np.stack([smooth(l, window) for l in seed_lengths], axis=0)
        ep   = np.arange(mat.shape[1])
        mean = mat.mean(axis=0)
        lo   = np.percentile(mat, 2.5,  axis=0)
        hi   = np.percentile(mat, 97.5, axis=0)
        color = palette.get(algo, None)
        ax.plot(ep, mean, label=algo, linewidth=1.8, color=color)
        ax.fill_between(ep, lo, hi, alpha=0.2, color=color)

    ax.set_xlabel(f"Episode (smoothed window={window})")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title(f"CartPole Balance Performance over Training", fontweight="bold")
    ax.axhline(500, linestyle="--", color="red", linewidth=1, label="Max (500)")
    ax.legend()
    plt.tight_layout()
    _savefig(f"cartpole_episode_lengths.png")


# ─────────────────────────────────────────────────────────────────────────────
# Eval return distribution (box plot across seeds)
# ─────────────────────────────────────────────────────────────────────────────

def plot_eval_distribution(eval_returns_dict, env_name):
    """
    Violin plot of evaluation returns for each method.
    Violin rather than box bc we care about the shape of the distribution,
    not just the quartiles.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    labels  = list(eval_returns_dict.keys())
    data    = [eval_returns_dict[k] for k in labels]

    parts = ax.violinplot(data, positions=range(len(labels)),
                          showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Evaluation Return")
    ax.set_title(f"Evaluation Return Distribution — {env_name}", fontweight="bold")
    plt.tight_layout()
    _savefig(f"eval_distribution_{env_name.lower().replace('-','_')}.png")