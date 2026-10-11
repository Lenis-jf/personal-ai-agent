from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_utils import load_jsonl
from evaluate_routing import score_subset


class EvaluatorTests(unittest.TestCase):
    def test_perfect_predictions_score_100_percent(self):
        dataset = load_jsonl(ROOT / "benchmarks/routing_v1.jsonl")
        predictions = {}
        for row in dataset:
            answers = {}
            for field, question in row["questions"].items():
                label = row["targets"][field]["label"]
                if question["type"] == "choice":
                    answers[field] = {"choice": label}
                else:
                    answers[field] = {"noul": 0.99 if label == "yes" else 0.01}
            predictions[row["id"]] = {"id": row["id"], "answers": answers, "latency_ms": 10.0}
        result = score_subset(dataset, predictions)
        self.assertEqual(result["summary"]["skill_accuracy"], 1.0)
        self.assertEqual(result["summary"]["model_route_accuracy"], 1.0)
        self.assertEqual(result["summary"]["overall_decision_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
