"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Configuration utilities.

This module defines structured configuration objects for evaluation
runs and provides helpers to load/save them from JSON or YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import json


# ---------------------------------------------------------------------
# Optional YAML support (clean fallback)
# ---------------------------------------------------------------------

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


# ---------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------

@dataclass
class EvalConfig:
    """
    Configuration for a robustness evaluation run.
    """
    # Environment
    env_id: str
    adapter: str = "gym"
    render_mode: Optional[str] = None
    max_episode_steps: Optional[int] = None

    # Policy
    algorithm: str = "unknown"
    policy_loader: Optional[str] = None
    policy_path: Optional[str] = None

    # Sweep
    epsilons: Sequence[float] = field(default_factory=list)
    seeds: Sequence[int] = field(default_factory=list)
    episodes_per_seed: int = 1
    max_steps: Optional[int] = None

    # Attack / defense
    attack: str = "none"
    defense: str = "none"
    norm: str = "linf"
    attack_steps: int = 10
    attack_step_size: Optional[float] = None
    jsma_top_k: int = 1

    # Output
    output_dir: str = "runs"
    prefix: Optional[str] = None

    # Plotting
    sma_window: Optional[int] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate configuration values.
        """
        if not self.env_id:
            raise ValueError("env_id must be specified.")

        if self.episodes_per_seed <= 0:
            raise ValueError("episodes_per_seed must be positive.")

        if self.attack not in {"none", "fgsm", "pgd", "jsma"}:
            raise ValueError(f"Unknown attack '{self.attack}'.")

        if self.defense not in {"none", "normalize_clip"}:
            raise ValueError(f"Unknown defense '{self.defense}'.")

        if self.norm not in {"linf", "l2"}:
            raise ValueError(f"Unknown norm '{self.norm}'.")

        if self.attack_steps <= 0:
            raise ValueError("attack_steps must be positive.")

        if self.jsma_top_k <= 0:
            raise ValueError("jsma_top_k must be positive.")


# ---------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------

def load_config(path: str | Path) -> EvalConfig:
    """
    Load an EvalConfig from JSON or YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".yaml", ".yml"}:
        if not _HAS_YAML:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        data = yaml.safe_load(path.read_text())
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    else:
        raise ValueError("Config file must be .json or .yaml/.yml")

    cfg = EvalConfig(**data)
    cfg.validate()
    return cfg


def save_config(
    cfg: EvalConfig,
    path: str | Path,
) -> None:
    """
    Save an EvalConfig to JSON or YAML.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(cfg)

    if path.suffix.lower() in {".yaml", ".yml"}:
        if not _HAS_YAML:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    elif path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2))
    else:
        raise ValueError("Config file must be .json or .yaml/.yml")