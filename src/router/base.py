"""Shared interface for interchangeable local decision-model backends."""
from __future__ import annotations

from typing import Protocol

from .decision import RoutingDecision


class DecisionBackend(Protocol):
    def decide(self, request: str, language: str = "en") -> RoutingDecision:
        """Classify a request without performing any tool or model action."""
        ...
