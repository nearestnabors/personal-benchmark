"""Load the golden dataset into Phoenix.

Reads a JSONL file where each line is:

    {
      "task_type": "extract_fields",
      "scoring":   "exact" | "structured" | "judge",
      "input":     "<the task prompt>",
      "reference": <expected answer for exact/structured, else null>,
      "rubric":    "<judge rubric for subjective tasks, else null>",
      "label":     "SAMPLE"        # honesty flag
    }

and uploads it as a versioned Phoenix dataset using the current client API:
``client.datasets.create_dataset(...)``.

    python scripts/build_dataset.py               # uses config.yaml
    python scripts/build_dataset.py --path data/my_dataset.jsonl --name my-benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DIR = Path(__file__).resolve().parent.parent


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:
        if r.get("scoring") not in {"exact", "structured", "judge"}:
            raise ValueError(
                f"row has invalid scoring={r.get('scoring')!r}: {r.get('input')!r:.60}"
            )
        if r["scoring"] in {"exact", "structured"} and r.get("reference") is None:
            raise ValueError(f"{r['scoring']} task needs a reference: {r.get('input')!r:.60}")
        if r["scoring"] == "judge" and not r.get("rubric"):
            raise ValueError(f"judge task needs a rubric: {r.get('input')!r:.60}")
    return rows


def main() -> None:
    cfg = yaml.safe_load((DIR / "config.yaml").read_text())
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--path", type=Path, default=DIR / cfg["dataset"]["path"])
    p.add_argument("--name", default=cfg["dataset"]["name"])
    p.add_argument("--endpoint", default=cfg["phoenix"]["endpoint"])
    args = p.parse_args()

    rows = load_rows(args.path)

    # One Phoenix example per row. Keep input/reference/metadata cleanly separated
    # so evaluators can pick exactly what they need.
    inputs = [{"task": r["input"]} for r in rows]
    outputs = [{"reference": r.get("reference")} for r in rows]
    metadata = [
        {
            "task_type": r["task_type"],
            "scoring": r["scoring"],
            "rubric": r.get("rubric"),
            "label": r.get("label"),
        }
        for r in rows
    ]

    from phoenix.client import Client

    client = Client(base_url=args.endpoint)
    dataset = client.datasets.create_dataset(
        name=args.name,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_description="Personal-benchmark golden dataset (see 'label' for SAMPLE rows).",
    )

    kinds = {}
    for r in rows:
        kinds[r["scoring"]] = kinds.get(r["scoring"], 0) + 1
    print(f"Uploaded dataset '{dataset.name}'  (id={dataset.id}, version={dataset.version_id})")
    print(f"  {len(rows)} examples: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    print(f"  View it at {args.endpoint}")


if __name__ == "__main__":
    main()
