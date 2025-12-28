"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Reproducibility utilities.

This module captures run metadata that helps reproduce and audit
benchmark results (software versions, git commit, platform info).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_run(cmd: list[str]) -> Optional[str]:
    """
    Run a command and return stdout, or None if it fails.
    """
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def get_git_commit(repo_root: Optional[str | Path] = None) -> Optional[str]:
    """
    Try to obtain current git commit hash.
    """
    cwd = str(repo_root) if repo_root is not None else None
    return _safe_run(["git", "-C", cwd, "rev-parse", "HEAD"]) if cwd else _safe_run(["git", "rev-parse", "HEAD"])


def get_git_dirty(repo_root: Optional[str | Path] = None) -> Optional[bool]:
    """
    Return True if repo has uncommitted changes, False if clean,
    or None if git is unavailable.
    """
    cwd = str(repo_root) if repo_root is not None else None
    out = _safe_run(["git", "-C", cwd, "status", "--porcelain"]) if cwd else _safe_run(["git", "status", "--porcelain"])
    if out is None:
        return None
    return len(out.strip()) > 0


def get_python_packages() -> Dict[str, str]:
    """
    Capture versions of key packages relevant for RoAd-RL.
    """
    versions: Dict[str, str] = {}

    def _try(pkg: str) -> None:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            versions[pkg] = str(ver)
        except Exception:
            versions[pkg] = "not_installed"

    for pkg in ["numpy", "pandas", "matplotlib", "gymnasium", "torch"]:
        _try(pkg)

    return versions


@dataclass(frozen=True)
class ReproRecord:
    """
    Reproducibility record stored alongside evaluation outputs.
    """
    platform: str
    python_version: str
    hostname: str
    cwd: str
    env_vars: Dict[str, Any]
    packages: Dict[str, str]
    git_commit: Optional[str]
    git_dirty: Optional[bool]


def collect_repro_record(
    *,
    repo_root: Optional[str | Path] = None,
    include_env_vars: bool = False,
) -> ReproRecord:
    """
    Collect a reproducibility record.

    Parameters
    ----------
    repo_root:
        Optional repo root used for git lookups.

    include_env_vars:
        If True, includes selected environment variables.
        Defaults to False to avoid leaking sensitive info.
    """
    env_vars: Dict[str, Any] = {}
    if include_env_vars:
        # Only store a conservative subset
        for k in ["CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED"]:
            if k in os.environ:
                env_vars[k] = os.environ.get(k)

    return ReproRecord(
        platform=platform.platform(),
        python_version=platform.python_version(),
        hostname=platform.node(),
        cwd=os.getcwd(),
        env_vars=env_vars,
        packages=get_python_packages(),
        git_commit=get_git_commit(repo_root=repo_root),
        git_dirty=get_git_dirty(repo_root=repo_root),
    )


def save_repro_record(
    record: ReproRecord,
    output_path: str | Path,
) -> None:
    """
    Save reproducibility record to JSON.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2))
