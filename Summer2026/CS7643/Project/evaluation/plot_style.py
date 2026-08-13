"""shared matplotlib style for paper-quality, colorblind-safe figures.

import this before pyplot elsewhere so every figure in the paper shares one look:
vector PDF export, >=8pt fonts at final size, Okabe-Ito palette. figures render
headless (Agg backend) so the validation CLI works over ssh / in CI.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colorblind-safe qualitative palette
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "figure.constrained_layout.use": True,
}


def apply_style():
    """set the shared rcParams; call once before plotting."""
    plt.rcParams.update(_RC)


def save_figure(fig, out_dir, name):
    """save as vector PDF (for LaTeX) + PNG (for quick viewing); returns the pdf path."""
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(pdf)
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    return pdf
