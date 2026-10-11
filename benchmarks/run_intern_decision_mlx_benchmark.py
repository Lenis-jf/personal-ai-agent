from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from intern_decision_mlx import Decider


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "routing_v1.jsonl"
OUTPUT = ROOT / "benchmarks" / "results" / "intern-decision-mlx-v1.jsonl"


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def make_request(row: dict) -> dict:
    # Allowlist: never send targets, labels, or evaluation metadata.
    return {
        "state": row["state"],
        "questions": row["questions"],
    }


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    rows = load_dataset(DATASET)
    if not rows:
        raise ValueError("The dataset is empty.")

    print(f"Dataset: {DATASET}")
    print(f"Cases: {len(rows)}")
    print("Loading Intern-Decision-0.8B with MLX...")

    load_start = time.perf_counter()
    decider = Decider()
    load_ms = (time.perf_counter() - load_start) * 1000

    print(f"Model loaded in {load_ms / 1000:.2f} seconds.")

    # Warm-up: initialize the prediction path before measuring cases.
    decider.predict(make_request(rows[0]))
    print("Warm-up complete. Running benchmark...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    latencies = []

    with OUTPUT.open("w", encoding="utf-8") as file:
        for index, row in enumerate(rows, start=1):
            request = make_request(row)

            start = time.perf_counter()
            result = decider.predict(request)
            latency_ms = (time.perf_counter() - start) * 1000

            latencies.append(latency_ms)

            # Save predictions only, never the gold targets.
            prediction = {
                "id": row["id"],
                "language": row["language"],
                "answers": result["answers"],
                "latency_ms": round(latency_ms, 3),
                "backend": "intern-decision-mlx",
            }

            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")

            print(
                f"[{index:02d}/{len(rows)}] "
                f"{row['id']} ({row['language']}) "
                f"{latency_ms:.1f} ms"
            )

    ordered = sorted(latencies)
    median = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]

    print("\nBenchmark complete.")
    print(f"Model load: {load_ms / 1000:.2f} seconds")
    print(f"Cases: {len(rows)}")
    print(f"Median latency: {median:.1f} ms")
    print(f"P95 latency: {p95:.1f} ms")
    print(f"Predictions saved to: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        raise