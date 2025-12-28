"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Component registry.

This module provides a lightweight registry mechanism for mapping
string identifiers (used in CLI/configs) to concrete implementations
(attacks, defenses, metrics, etc.).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type


class Registry:
    """
    Generic string-to-object registry.

    This registry is intentionally simple:
    - no global magic
    - explicit registration
    - clear error messages
    """

    def __init__(self, name: str):
        self._name = name
        self._items: Dict[str, Any] = {}

    def register(self, key: str, value: Any) -> None:
        """
        Register a component under a string key.
        """
        key = key.lower()
        if key in self._items:
            raise KeyError(
                f"{self._name} registry already contains key '{key}'."
            )
        self._items[key] = value

    def get(self, key: str) -> Any:
        """
        Retrieve a registered component.
        """
        key = key.lower()
        if key not in self._items:
            raise KeyError(
                f"Unknown {self._name} '{key}'. "
                f"Available: {sorted(self._items.keys())}"
            )
        return self._items[key]

    def has(self, key: str) -> bool:
        return key.lower() in self._items

    def keys(self):
        return sorted(self._items.keys())


# ---------------------------------------------------------------------
# Global registries (explicit, not magic)
# ---------------------------------------------------------------------

attack_registry = Registry("attack")
defense_registry = Registry("defense")
metric_registry = Registry("metric")
env_registry = Registry("environment")
