"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Episode logger.

This module defines canonical logging for episodic evaluation results.
It records returns, lengths, termination signals, and optional safety
flags so that all metrics (returns, risk, safety) can be computed
consistently from episodes.csv.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from road_rl.core.types import ExperimentResult


# ---------------------------------------------------------------------
# Episode record utilities
# ---------------------------------------------------------------------

# Known safety flags commonly exposed by envs (esp. HighwayEnv)
KNOWN_SAFETY_FLAGS = {
    "collision",
    "crash",
    "offroad",
    "lane_violation",
    "near_miss",
}


def _extract_safety_flags(info: Mapping[str, Any]) -> Dict[str, bool]:
    """
    Extract known boolean safety flags from env info if present.
    """
    flags: Dict[str, bool] = {}
    for k in KNOWN_SAFETY_FLAGS:
        if k in info:
            flags[k] = bool(info.get(k))
    return flags


def _extract_termination_reason(info: Mapping[str, Any]) -> Optional[str]:
    """
    Extract a termination reason if provided by the environment.
    """
    for key in ("termination_reason", "reason", "done_reason"):
        if key in info:
            return str(info.get(key))
    return None


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def save_experiment(
    result: ExperimentResult,
    output_dir: str | Path,
    *,
    prefix: Optional[str] = None,
) -> None:
    """
    Save an ExperimentResult to disk.

    This writes:
      - episodes.csv : episode-level records
      - meta.json    : experiment metadata
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Episodes CSV
    # -----------------------------
    name_parts = [
        result.env_id,
        result.algorithm,
        result.attack_name or "clean",
        result.defense_name or "none",
    ]
    if prefix:
        name_parts.insert(0, prefix)
    base_name = "_".join(name_parts)

    episodes_path = out_dir / f"{base_name}_episodes.csv"

    # Collect all episode rows and union of keys
    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []

    for ep in result.episodes:
        row: Dict[str, Any] = {
            "episode": ep.episode,
            "seed": ep.seed,
            "epsilon": ep.epsilon,
            "return": ep.return_,
            "length": ep.length,
            "terminated": bool(ep.terminated),
            "truncated": bool(ep.truncated),
        }

        # Optional termination reason
        if ep.info:
            reason = _extract_termination_reason(ep.info)
            if reason is not None:
                row["termination_reason"] = reason

            # Optional safety flags
            row.update(_extract_safety_flags(ep.info))

            # Optional violations count
            if "violations" in ep.info:
                try:
                    row["violations"] = int(ep.info["violations"])
                except Exception:
                    pass

        rows.append(row)
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with episodes_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # -----------------------------
    # Meta JSON
    # -----------------------------
    meta = {
        "env_id": result.env_id,
        "algorithm": result.algorithm,
        "attack": result.attack_name,
        "defense": result.defense_name,
        "metadata": result.metadata,
        "n_episodes": len(result.episodes),
    }

    meta_path = out_dir / f"{base_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
