"""Model-agnostic decision object and safe model-routing policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import TOOL_FIELDS


@dataclass(frozen=True)
class RoutingDecision:
    skill: str
    complexity: str
    model_route: str
    tools: frozenset[str]
    needs_clarification: bool
    requires_confirmation: bool
    probabilities: dict[str, float]
    latency_ms: float | None = None

    @classmethod
    def from_answers(cls, answers: dict[str, dict[str, Any]], latency_ms: float | None = None) -> "RoutingDecision":
        """Convert either backend's normalized `answers` mapping to a stable type."""
        def choice(field: str, fallback: str) -> str:
            value = answers.get(field, {}).get("choice", fallback)
            return str(value) if value is not None else fallback

        def probability_yes(field: str) -> float:
            answer = answers.get(field, {})
            if "noul" in answer:
                try:
                    return float(answer["noul"])
                except (TypeError, ValueError):
                    pass
            probabilities = answer.get("probabilities", {})
            if isinstance(probabilities, dict) and "yes" in probabilities:
                try:
                    return float(probabilities["yes"])
                except (TypeError, ValueError):
                    pass
            return 0.0

        probabilities: dict[str, float] = {}
        for field, answer in answers.items():
            if isinstance(answer, dict):
                if isinstance(answer.get("confidence"), (int, float)):
                    probabilities[field] = float(answer["confidence"])
                elif "noul" in answer:
                    probabilities[field] = float(answer["noul"])

        tools = frozenset(field.removeprefix("tool_") for field in TOOL_FIELDS if probability_yes(field) >= 0.5)
        return cls(
            skill=choice("skill", "personal_assistant"),
            complexity=choice("complexity", "moderate"),
            model_route=choice("model_route", "local"),
            tools=tools,
            needs_clarification=probability_yes("needs_clarification") >= 0.5,
            requires_confirmation=probability_yes("requires_confirmation") >= 0.5,
            probabilities=probabilities,
            latency_ms=latency_ms,
        )

    def effective_model_route(
        self,
        *,
        cloud_enabled: bool = False,
        user_approved: bool = False,
        content_reviewed_as_safe: bool = False,
    ) -> str:
        """Fail closed: model suggestions never enable cloud processing by themselves."""
        if (
            self.model_route == "cloud_candidate"
            and cloud_enabled
            and user_approved
            and content_reviewed_as_safe
        ):
            return "cloud"
        return "local"
