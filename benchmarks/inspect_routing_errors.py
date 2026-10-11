import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "routing_v1.jsonl"
PREDICTIONS = ROOT / "benchmarks" / "results" / "intern-decision-mlx-v1.jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


gold = {row["id"]: row for row in load_jsonl(DATASET)}
predictions = load_jsonl(PREDICTIONS)

errors = []
total = 0

for prediction in predictions:
    case = gold[prediction["id"]]

    for field, expected in case["targets"].items():
        expected = expected.get("label", expected) if isinstance(expected, dict) else expected
        answer = prediction["answers"].get(field, {})
        actual = answer.get("decision", answer.get("choice"))

        if actual is None and "noul" in answer:
            actual = "yes" if answer["noul"] >= 0.5 else "no"

        total += 1

        if actual != expected:
            errors.append(
                (prediction["id"], field, expected, actual)
            )

print(f"Casos: {len(predictions)}")
print(f"Decisiones comparadas: {total}")
print(f"Errores: {len(errors)}")
print()

for case_id, field, expected, actual in errors:
    print(
        f"{case_id} | {field} | "
        f"esperado={expected} | predicho={actual}"
    )
