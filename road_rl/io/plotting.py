"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Plotting utilities.

This module provides publication-quality plotting functions for
visualizing robustness evaluation results. All plots are saved
as vector PDFs with consistent styling suitable for papers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciencePlots style
import scienceplots  # noqa: F401

# ------------------------------------------------------------
# Global plotting style (paper-ready)
# ------------------------------------------------------------

plt.style.use(["science", "ieee"])

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
})


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def simple_moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def adaptive_sma(y: np.ndarray) -> np.ndarray:
    n = len(y)
    window = 50 if n < 500 else 100
    return simple_moving_average(y, window)


# ------------------------------------------------------------
# Plot 1: Return vs epsilon (robustness curve)
# ------------------------------------------------------------

def plot_return_vs_epsilon(
    csv_files: Iterable[str | Path],
    labels: Iterable[str],
    output_path: str | Path,
    *,
    title: Optional[str] = None,
    smooth: bool = True,
) -> None:
    """
    Plot mean episodic return as a function of epsilon.
    """
    ensure_dir(Path(output_path).parent)

    plt.figure()

    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)

        grouped = df.groupby("epsilon")["total_return"]
        eps = grouped.mean().index.to_numpy()
        mean_return = grouped.mean().to_numpy()

        if smooth:
            mean_return = adaptive_sma(mean_return)

        plt.plot(eps, mean_return, label=label)

    plt.xlabel(r"Attack strength $\epsilon$")
    plt.ylabel("Mean episodic return")

    if title:
        plt.title(title)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, format="pdf")
    plt.close()


# ------------------------------------------------------------
# Plot 2: Bar chart (AUC or mean return)
# ------------------------------------------------------------

def plot_bar_metric(
    values: Iterable[float],
    labels: Iterable[str],
    output_path: str | Path,
    *,
    ylabel: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot a bar chart for scalar robustness metrics (e.g. AUC).
    """
    ensure_dir(Path(output_path).parent)

    x = np.arange(len(values))

    plt.figure()
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=30, ha="right")

    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, format="pdf")
    plt.close()


# ------------------------------------------------------------
# Plot 3: Violin plot (return distribution)
# ------------------------------------------------------------

def plot_violin_returns(
    csv_files: Iterable[str | Path],
    labels: Iterable[str],
    output_path: str | Path,
    *,
    title: Optional[str] = None,
) -> None:
    """
    Plot violin distributions of episodic returns.
    """
    ensure_dir(Path(output_path).parent)

    data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        data.append(df["total_return"].to_numpy())

    plt.figure()
    plt.violinplot(data, showmeans=True, showextrema=False)

    plt.xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right")
    plt.ylabel("Episodic return")

    if title:
        plt.title(title)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, format="pdf")
    plt.close()


# ------------------------------------------------------------
# Plot 4: Raw reward curve (episode index)
# ------------------------------------------------------------

def plot_raw_rewards(
    csv_file: str | Path,
    output_path: str | Path,
    *,
    title: Optional[str] = None,
) -> None:
    """
    Plot raw episodic returns without smoothing.
    """
    ensure_dir(Path(output_path).parent)

    df = pd.read_csv(csv_file)

    plt.figure()
    plt.plot(df["episode"], df["total_return"])

    plt.xlabel("Episode")
    plt.ylabel("Episodic return")

    if title:
        plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, format="pdf")
    plt.close()
