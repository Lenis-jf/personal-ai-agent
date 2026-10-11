from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_utils import load_jsonl, visible_model_input
from src.router.decision import RoutingDecision
from src.router.schema import TOOL_FIELDS, question_schema


class RouterBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_jsonl(ROOT / "benchmarks/routing_v1.jsonl")

    def test_dataset_has_18_cases_per_language(self):
        counts = {lang: sum(row["language"] == lang for row in self.rows) for lang in ("es", "en", "de")}
        self.assertEqual(counts, {"es": 18, "en": 18, "de": 18})

    def test_unique_ids_and_12_fields_per_case(self):
        ids = [row["id"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(len(row["questions"]) == 12 for row in self.rows))

    def test_gold_labels_are_legal_options(self):
        for row in self.rows:
            for name, question in row["questions"].items():
                label = row["targets"][name]["label"]
                if question["type"] == "choice":
                    self.assertIn(label, question["criteria"], (row["id"], name, label))
                else:
                    self.assertIn(label, {"yes", "no"})

    def test_targets_never_pass_to_model(self):
        visible = visible_model_input(self.rows[0])
        self.assertNotIn("targets", visible)
        self.assertNotIn("category", visible)
        self.assertEqual(set(visible), {"id", "language", "state", "questions"})

    def test_schema_supports_all_tested_languages_and_fields(self):
        for language in ("es", "en", "de"):
            schema = question_schema(language)
            self.assertEqual(tuple(name for name in schema if name.startswith("tool_")), TOOL_FIELDS)
            self.assertEqual(len(schema), 12)

    def test_cloud_suggestion_fails_closed_by_default(self):
        decision = RoutingDecision(
            skill="research", complexity="complex", model_route="cloud_candidate",
            tools=frozenset({"web_search"}), needs_clarification=False,
            requires_confirmation=False, probabilities={},
        )
        self.assertEqual(decision.effective_model_route(), "local")
        self.assertEqual(decision.effective_model_route(cloud_enabled=True), "local")
        self.assertEqual(decision.effective_model_route(cloud_enabled=True, user_approved=True), "local")
        self.assertEqual(
            decision.effective_model_route(cloud_enabled=True, user_approved=True, content_reviewed_as_safe=True),
            "cloud",
        )

    def test_answer_normalization_extracts_tools(self):
        answers = {
            "skill": {"choice": "file_management", "confidence": 0.91},
            "complexity": {"choice": "moderate", "confidence": 0.77},
            "model_route": {"choice": "local", "confidence": 0.88},
            "tool_file_search": {"noul": 0.95},
            "tool_web_search": {"noul": 0.03},
            "needs_clarification": {"noul": 0.8},
            "requires_confirmation": {"noul": 0.6},
        }
        decision = RoutingDecision.from_answers(answers)
        self.assertEqual(decision.skill, "file_management")
        self.assertEqual(decision.tools, frozenset({"file_search"}))
        self.assertTrue(decision.needs_clarification)
        self.assertTrue(decision.requires_confirmation)


if __name__ == "__main__":
    unittest.main()
