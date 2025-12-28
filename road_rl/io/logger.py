"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Experiment logging utilities.

This module handles persistence of evaluation results produced by
the sweep runner. Results are written in simple, interoperable formats
(CSV and JSON) to support analysis, plotting, and paper reproduction.
"""

from __future__ import annotations

import json
import csv
import os
from pathlib import Path
from typing import Optional

from road_rl.core.types import ExperimentResult, EpisodeResult


def ensure_dir(path: Path) -> None:
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_experiment(
    result: ExperimentResult,
    output_dir: str | Path,
    *,
    prefix: Optional[str] = None,
) -> None:
    """
    Save an ExperimentResult to disk.

    This function writes:
    - episode-level results as CSV
    - experiment metadata as JSON

    Parameters
    ----------
    result:
        ExperimentResult produced by the sweep runner.

    output_dir:
        Directory where outputs will be written.

    prefix:
        Optional filename prefix (useful when running multiple sweeps).
    """
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    name_parts = [
        result.env_id,
        result.algorithm,
        result.attack_name or "clean",
        result.defense_name or "none",
    ]

    if prefix:
        name_parts.insert(0, prefix)

    base_name = "_".join(name_parts)

    # ------------------------------------------------------------
    # Save episode-level results (CSV)
    # ------------------------------------------------------------
    csv_path = out_dir / f"{base_name}_episodes.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "seed",
                "epsilon",
                "total_return",
                "length",
                "terminated",
                "truncated",
            ]
        )

        for ep in result.episode_results:
            writer.writerow(
                [
                    ep.episode,
                    ep.seed,
                    ep.epsilon,
                    ep.total_return,
                    ep.length,
                    ep.terminated,
                    ep.truncated,
                ]
            )

    # ------------------------------------------------------------
    # Save experiment metadata (JSON)
    # ------------------------------------------------------------
    meta_path = out_dir / f"{base_name}_meta.json"

    meta = {
        "env_id": result.env_id,
        "algorithm": result.algorithm,
        "attack": result.attack_name,
        "defense": result.defense_name,
        "epsilons": result.epsilons,
        "num_episodes": len(result.episode_results),
        "metadata": result.metadata,
    }

    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
