"""Evaluate saved router predictions against the human-labelled benchmark."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_utils import DEFAULT_DATASET, load_jsonl

CHOICE_FIELDS = ("skill", "complexity", "model_route")
BINARY_FIELDS = (
    "tool_file_search", "tool_file_read", "tool_file_modify", "tool_web_search",
    "tool_email_draft", "tool_email_send", "tool_system_change",
    "needs_clarification", "requires_confirmation",
)
ALL_FIELDS = CHOICE_FIELDS + BINARY_FIELDS


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if not row_id:
                raise ValueError(f"Missing id in predictions line {line_number}")
            if row_id in predictions:
                raise ValueError(f"Duplicate prediction id {row_id!r}")
            predictions[row_id] = row
    return predictions


def predicted_label(field: str, answer: dict[str, Any]) -> str | None:
    if field in CHOICE_FIELDS:
        choice = answer.get("choice")
        if choice is not None:
            return str(choice)
        probabilities = answer.get("probabilities")
        if isinstance(probabilities, dict) and probabilities:
            return str(max(probabilities, key=probabilities.get))
        return None

    if "noul" in answer:
        try:
            return "yes" if float(answer["noul"]) >= 0.5 else "no"
        except (TypeError, ValueError):
            pass
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and "yes" in probs and "no" in probs:
        return "yes" if float(probs["yes"]) >= float(probs["no"]) else "no"
    return None


def binary_metrics(gold: list[str], pred: list[str]) -> dict[str, float]:
    tp = sum(g == "yes" and p == "yes" for g, p in zip(gold, pred))
    # Invalid/missing answers count as errors, not as silently ignored cases.
    fp = sum(g == "no" and p != "no" for g, p in zip(gold, pred))
    fn = sum(g == "yes" and p != "yes" for g, p in zip(gold, pred))
    tn = sum(g == "no" and p == "no" for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(gold) if gold else 0.0,
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def score_subset(dataset: list[dict[str, Any]], preds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = [row["id"] for row in dataset if row["id"] not in preds]
    extra = sorted(set(preds) - {row["id"] for row in dataset})
    if missing or extra:
        raise ValueError(f"Prediction coverage mismatch. Missing={missing[:5]}, extra={extra[:5]}")

    per_field: dict[str, list[tuple[str, str]]] = defaultdict(list)
    latencies = []
    for row in dataset:
        pred_row = preds[row["id"]]
        answers = pred_row.get("answers", {})
        if isinstance(pred_row.get("latency_ms"), (int, float)):
            latencies.append(float(pred_row["latency_ms"]))
        for field in ALL_FIELDS:
            answer = answers.get(field, {})
            expected = str(row["targets"][field]["label"])
            actual = predicted_label(field, answer) if isinstance(answer, dict) else None
            per_field[field].append((expected, actual if actual is not None else "__missing__"))

    field_metrics: dict[str, Any] = {}
    for field, pairs in per_field.items():
        gold = [item[0] for item in pairs]
        pred = [item[1] for item in pairs]
        correct = sum(g == p for g, p in pairs)
        field_metrics[field] = {
            "accuracy": correct / len(pairs),
            "correct": correct,
            "total": len(pairs),
        }
        if field in BINARY_FIELDS:
            field_metrics[field].update(binary_metrics(gold, pred))

    choice_acc = statistics.mean(field_metrics[f]["accuracy"] for f in CHOICE_FIELDS)
    tool_f1 = statistics.mean(field_metrics[f]["f1_yes"] for f in BINARY_FIELDS[:7])
    safety_f1 = statistics.mean(field_metrics[f]["f1_yes"] for f in ("needs_clarification", "requires_confirmation"))

    languages: dict[str, Any] = {}
    for language in sorted({row["language"] for row in dataset}):
        subset = [row for row in dataset if row["language"] == language]
        language_metrics = score_subset_fields_only(subset, preds)
        languages[language] = {"n": len(subset), **language_metrics}

    result: dict[str, Any] = {
        "n_cases": len(dataset),
        "n_target_decisions": len(dataset) * len(ALL_FIELDS),
        "summary": {
            "skill_accuracy": field_metrics["skill"]["accuracy"],
            "complexity_accuracy": field_metrics["complexity"]["accuracy"],
            "model_route_accuracy": field_metrics["model_route"]["accuracy"],
            "mean_choice_accuracy": choice_acc,
            "mean_tool_f1_yes": tool_f1,
            "mean_safety_flag_f1_yes": safety_f1,
            "overall_decision_accuracy": sum(v["correct"] for v in field_metrics.values()) / (len(dataset) * len(ALL_FIELDS)),
            "median_latency_ms": statistics.median(latencies) if latencies else None,
            "p95_latency_ms": percentile(latencies, 0.95) if latencies else None,
        },
        "per_field": field_metrics,
        "per_language": languages,
    }
    return result


def score_subset_fields_only(dataset: list[dict[str, Any]], preds: dict[str, dict[str, Any]]) -> dict[str, float]:
    values = {}
    for field in ALL_FIELDS:
        pairs = []
        for row in dataset:
            answer = preds[row["id"]].get("answers", {}).get(field, {})
            gold = str(row["targets"][field]["label"])
            pred = predicted_label(field, answer) if isinstance(answer, dict) else None
            pairs.append((gold, pred if pred is not None else "__missing__"))
        if field in CHOICE_FIELDS:
            values[f"{field}_accuracy"] = sum(g == p for g, p in pairs) / len(pairs)
        elif field.startswith("tool_"):
            values[f"{field}_f1"] = binary_metrics([x[0] for x in pairs], [x[1] for x in pairs])["f1_yes"]
        else:
            values[f"{field}_accuracy"] = sum(g == p for g, p in pairs) / len(pairs)
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(math.ceil(len(ordered) * fraction) - 1)))
    return ordered[index]


def make_markdown(result: dict[str, Any], prediction_path: str | Path) -> str:
    s = result["summary"]
    lines = [
        "# Router benchmark results",
        "",
        f"Predictions: `{prediction_path}`",
        f"Cases: **{result['n_cases']}** | Gold decisions: **{result['n_target_decisions']}**",
        "",
        "## Primary metrics",
        "",
        "| Metric | Score |",
        "|---|---:|",
        f"| Skill accuracy | {s['skill_accuracy']:.1%} |",
        f"| Complexity accuracy | {s['complexity_accuracy']:.1%} |",
        f"| Model-route accuracy | {s['model_route_accuracy']:.1%} |",
        f"| Mean of 3 choice accuracies | {s['mean_choice_accuracy']:.1%} |",
        f"| Mean F1 for tool flags (positive class = yes) | {s['mean_tool_f1_yes']:.1%} |",
        f"| Mean F1 for clarification/confirmation flags | {s['mean_safety_flag_f1_yes']:.1%} |",
        f"| Overall decision accuracy (all 12 fields) | {s['overall_decision_accuracy']:.1%} |",
        f"| Median per-case latency | {s['median_latency_ms']:.1f} ms |" if s["median_latency_ms"] is not None else "| Median per-case latency | n/a |",
        f"| P95 per-case latency | {s['p95_latency_ms']:.1f} ms |" if s["p95_latency_ms"] is not None else "| P95 per-case latency | n/a |",
        "",
        "## By language",
        "",
        "| Language | Cases | Skill accuracy | Complexity accuracy | Model-route accuracy | Tool F1 (mean) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language, metrics in result["per_language"].items():
        tool_values = [metrics[f"tool_{name}_f1"] for name in (
            "file_search", "file_read", "file_modify", "web_search", "email_draft", "email_send", "system_change"
        )]
        lines.append(
            f"| {language} | {metrics['n']} | {metrics['skill_accuracy']:.1%} | "
            f"{metrics['complexity_accuracy']:.1%} | {metrics['model_route_accuracy']:.1%} | "
            f"{statistics.mean(tool_values):.1%} |"
        )
    lines += ["", "## By field", "", "| Field | Accuracy | Correct / total | F1 for yes |", "|---|---:|---:|---:|"]
    for field, m in result["per_field"].items():
        f1 = f"{m['f1_yes']:.1%}" if "f1_yes" in m else "n/a"
        lines.append(f"| {field} | {m['accuracy']:.1%} | {m['correct']}/{m['total']} | {f1} |")
    lines += [
        "",
        "> Interpretation: this is a small, policy-labelled pilot, not a general model leaderboard. "
        "Inspect individual errors before selecting a backend. The `cloud_candidate` label never enables cloud use.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    dataset = load_jsonl(args.dataset)
    preds = load_predictions(args.predictions)
    result = score_subset(dataset, preds)
    predictions_path = Path(args.predictions).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else predictions_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = predictions_path.stem
    json_path = out_dir / f"{stem}.metrics.json"
    md_path = out_dir / f"{stem}.summary.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(make_markdown(result, predictions_path), encoding="utf-8")
    print(make_markdown(result, predictions_path))
    print(f"\nWrote: {json_path}\nWrote: {md_path}")


if __name__ == "__main__":
    main()
