"""Run the multilingual benchmark against Laya, fully locally."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_utils import DEFAULT_DATASET, load_jsonl, visible_model_input, write_jsonl
from src.router.laya_backend import LayaDecisionBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(ROOT / "benchmarks/results/laya-v1.jsonl"))
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    print(f"Loading Laya multilingual on {args.device}...", flush=True)
    load_started = time.perf_counter()
    backend = LayaDecisionBackend(device=args.device)
    load_seconds = time.perf_counter() - load_started
    print(f"Model load: {load_seconds:.2f}s", flush=True)

    # Warm up kernels before measuring per-case latency; this result is discarded.
    backend.predict(visible_model_input(rows[0]))
    predictions = []
    for index, row in enumerate(rows, start=1):
        prediction = backend.predict(visible_model_input(row))
        predictions.append(prediction)
        print(f"[{index:02d}/{len(rows)}] {row['id']}: {prediction['latency_ms']:.1f} ms", flush=True)

    write_jsonl(args.output, predictions)
    print(f"\nSaved predictions to: {Path(args.output).resolve()}")
    print(f"Run evaluation: python benchmarks/evaluate_routing.py --predictions {args.output}")


if __name__ == "__main__":
    main()
