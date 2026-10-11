"""Official Laya multilingual decision backend, intended for Apple Silicon/MPS."""
from __future__ import annotations

import time
from typing import Any

from .decision import RoutingDecision
from .schema import question_schema


class LayaDecisionBackend:
    def __init__(self, device: str = "mps") -> None:
        try:
            from laya import load
        except ImportError as exc:
            raise RuntimeError(
                "Laya is not installed in this Python environment. Install it with `python -m pip install laya`."
            ) from exc

        # Pin the multilingual checkpoint: our requests may be Spanish, English or German.
        self.model = load("convaiinnovations/laya", subfolder="multilingual", device=device)

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Predict using only model-visible fields; gold labels are never passed in."""
        started = time.perf_counter()
        result = self.model.predict(row["state"], row["questions"])
        wall_ms = (time.perf_counter() - started) * 1000
        answers = result.get("answers", result)
        return {
            "id": row["id"],
            "language": row["language"],
            "answers": answers,
            "latency_ms": round(wall_ms, 3),
            "backend": "laya-multilingual",
        }

    def decide(self, request: str, language: str = "en") -> RoutingDecision:
        """Route one live text request; this method chooses, but executes nothing."""
        if language not in {"en", "es", "de"}:
            language = "en"
        row = {
            "id": "live-request",
            "language": language,
            "state": {"user_request": request},
            "questions": question_schema(language),
        }
        prediction = self.predict(row)
        return RoutingDecision.from_answers(prediction["answers"], prediction["latency_ms"])
