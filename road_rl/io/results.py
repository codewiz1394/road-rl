"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Results management utilities.

This module defines a structured layout for storing evaluation results.
It centralizes path construction so that experiments are reproducible,
organized, and easy to inspect or share.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ResultPaths:
    """
    Canonical paths for a single evaluation run.
    """
    base_dir: Path
    run_dir: Path

    episodes_csv: Path
    metrics_json: Path
    meta_json: Path
    repro_json: Path

    return_vs_eps_pdf: Path
    violin_pdf: Path
    raw_rewards_pdf: Path


def make_result_paths(
    *,
    root_dir: str | Path,
    experiment_name: str,
    env_id: str,
    algorithm: str,
    attack: Optional[str],
    defense: Optional[str],
) -> ResultPaths:
    """
    Create and return standardized result paths for an evaluation run.

    Parameters
    ----------
    root_dir:
        Root results directory (e.g. "results").

    experiment_name:
        User-defined experiment name.

    env_id:
        Environment identifier.

    algorithm:
        Algorithm name.

    attack:
        Attack name or None.

    defense:
        Defense name or None.
    """
    root_dir = Path(root_dir)
    base_dir = root_dir / experiment_name

    attack_name = attack or "clean"
    defense_name = defense or "none"

    run_name = f"{env_id}_{algorithm}_{attack_name}_{defense_name}"
    run_dir = base_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=True)

    return ResultPaths(
        base_dir=base_dir,
        run_dir=run_dir,

        episodes_csv=run_dir / "episodes.csv",
        metrics_json=run_dir / "metrics.json",
        meta_json=run_dir / "meta.json",
        repro_json=run_dir / "repro.json",

        return_vs_eps_pdf=run_dir / "return_vs_eps.pdf",
        violin_pdf=run_dir / "violin.pdf",
        raw_rewards_pdf=run_dir / "raw_rewards.pdf",
    )
