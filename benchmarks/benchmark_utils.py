"""Shared JSONL I/O for the offline router benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).resolve().parent / "routing_v1.jsonl"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if row.get("id") in seen:
                raise ValueError(f"Duplicate dataset id {row.get('id')!r} at line {line_number}")
            seen.add(row.get("id"))
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def visible_model_input(row: dict[str, Any]) -> dict[str, Any]:
    """Explicit allowlist ensures targets/metadata never reach the model."""
    return {key: row[key] for key in ("id", "language", "state", "questions")}


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
