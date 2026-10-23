"""Model-agnostic decision object and safe model-routing policy with two-stage sub-skill support."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import SUB_SKILL_FIELDS, TOOL_FIELDS


@dataclass(frozen=True)
class RoutingDecision:
    skill: str
    complexity: str
    model_route: str
    tools: frozenset[str]
    needs_clarification: bool
    requires_confirmation: bool
    probabilities: dict[str, float]
    sub_skills: frozenset[str] = field(default_factory=frozenset)
    sub_skill_confidences: dict[str, float] = field(default_factory=dict)
    needs_previous_context: bool = False
    skill_confidence: float = 0.0
    rerouted: bool = False
    latency_ms: float | None = None

    @classmethod
    def from_answers(
        cls,
        answers: dict[str, dict[str, Any]],
        *,
        sub_skill_threshold: float = 0.45,
        tool_threshold: float = 0.48,
        rerouted: bool = False,
        latency_ms: float | None = None,
    ) -> "RoutingDecision":
        """
        Convert Intern-Decision's raw answer mappings into a structured RoutingDecision.
        Extracts active tools and modular sub-skills exceeding the confidence threshold,
        recording exact confidence scores for downstream LLM evaluation.
        """
        def choice(field_name: str, fallback: str) -> str:
            value = answers.get(field_name, {}).get("choice", fallback)
            return str(value) if value is not None else fallback

        def probability_yes(field_name: str) -> float:
            answer = answers.get(field_name, {})
            if "noul" in answer:
                try:
                    return float(answer["noul"])
                except (TypeError, ValueError):
                    pass
            probs = answer.get("probabilities", {})
            if isinstance(probs, dict) and "yes" in probs:
                try:
                    return float(probs["yes"])
                except (TypeError, ValueError):
                    pass
            return 0.0

        probabilities: dict[str, float] = {}
        for field_name, answer in answers.items():
            if isinstance(answer, dict):
                if isinstance(answer.get("confidence"), (int, float)):
                    probabilities[field_name] = float(answer["confidence"])
                elif "noul" in answer:
                    probabilities[field_name] = float(answer["noul"])

        chosen_skill = choice("skill", "personal_assistant")
        skill_conf = float(probabilities.get("skill", 0.0))

        tools = frozenset(
            f.removeprefix("tool_") for f in TOOL_FIELDS if probability_yes(f) >= tool_threshold
        )

        sub_skills_dict: dict[str, float] = {}
        for f in SUB_SKILL_FIELDS:
            sub_name = f.removeprefix("sub_")
            prob = probability_yes(f)
            if prob >= sub_skill_threshold:
                sub_skills_dict[sub_name] = round(prob, 2)

        # Include Stage-2 competitive choice if selected with sufficient confidence
        primary_sub = answers.get("primary_sub_skill", {})
        if isinstance(primary_sub, dict) and primary_sub.get("choice"):
            chosen_sub = str(primary_sub["choice"])
            sub_conf = float(primary_sub.get("confidence", 1.0) or 1.0)
            if sub_conf >= 0.30:
                sub_skills_dict[chosen_sub] = round(max(sub_skills_dict.get(chosen_sub, 0.0), sub_conf), 2)

        return cls(
            skill=chosen_skill,
            complexity=choice("complexity", "moderate"),
            model_route=choice("model_route", "local"),
            tools=tools,
            needs_clarification=probability_yes("needs_clarification") >= 0.65,
            requires_confirmation=probability_yes("requires_confirmation") >= 0.60,
            probabilities=probabilities,
            sub_skills=frozenset(sub_skills_dict.keys()),
            sub_skill_confidences=sub_skills_dict,
            needs_previous_context=probability_yes("needs_previous_context") >= 0.50,
            skill_confidence=skill_conf,
            rerouted=rerouted,
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