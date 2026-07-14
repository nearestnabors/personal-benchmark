"""Surface your RECURRING task types from harvested prompts — using a big model.

You can't recall your recurring tasks from memory; that's the point of harvesting.
Grouping messy real history well is a job for a LARGE model. Simple local keyword
matching is too crude and misses too much, so this relies on a big model only.

Two methods:

  paste (default) — write a copy-paste-ready prompt (your redacted prompts + a
                    strict JSON schema) to hand to a big model like Claude or
                    ChatGPT. No API key. Save the model's JSON reply as
                    data/taxonomy.json.
  llm             — send that same prompt to an OpenAI-compatible endpoint and
                    write data/taxonomy.json for you. Point --base-url at a
                    hosted API with --model and --api-key.

    python scripts/taxonomy.py --in data/harvested/all.jsonl
    python scripts/taxonomy.py --in data/harvested/all.jsonl --method llm \\
        --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key $OPENAI_API_KEY
    # (Claude: --base-url https://api.anthropic.com/v1 --model claude-haiku-4-5-20251001)

Human-in-the-loop: this surfaces and ranks your task types. YOU decide which ones
are worth an eval in Step 3. It does not auto-decide.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent

_HEADER = """I'm analyzing my own AI-assistant usage to build a personal eval set.
Below are {n} requests I've actually made (secrets/PII already redacted).

Group them into 8-15 recurring TASK TYPES. Reply with ONLY a JSON array, no prose,
ranked by count (most frequent first). Each item must look like:

  {{"task_type": "extract_fields", "count": 42,
    "representative_examples": ["<verbatim request>", "<verbatim request>"]}}

Use short snake_case task_type names. representative_examples must be 2-3 real
requests copied verbatim from the list. Here are my requests:
"""


def load_prompts(path: Path, limit: int) -> tuple[list[str], int]:
    """Real user requests: dedup (preserve order), then cap so it stays pasteable."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen: set[str] = set()
    prompts: list[str] = []
    for r in records:
        text = (r.get("text") or "").strip()
        if r.get("role") != "user" or not text or text in seen:
            continue
        seen.add(text)
        prompts.append(text)
    return prompts[:limit], len(seen)


def build_prompt(prompts: list[str]) -> str:
    body = "\n".join(f"{i}. {p}" for i, p in enumerate(prompts, 1))
    return _HEADER.format(n=len(prompts)) + "\n" + body + "\n"


def extract_json_array(text: str) -> list[dict]:
    """Pull the JSON array out of a model reply, tolerating fences/prose."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="inp", type=Path, default=DIR / "data/harvested/all.jsonl")
    p.add_argument("--method", choices=["paste", "llm"], default="paste")
    p.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible endpoint for --method llm "
        "(OpenAI: https://api.openai.com/v1 ; Claude: https://api.anthropic.com/v1)",
    )
    p.add_argument("--model", default=None, help="model id for --method llm (e.g. gpt-4o-mini)")
    p.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    )
    p.add_argument("--out", type=Path, default=DIR / "data/taxonomy.json")
    p.add_argument("--max", type=int, default=400, help="max prompts to send the model")
    args = p.parse_args()

    if not args.inp.exists():
        sys.exit(f"no harvested file at {args.inp} — run scripts/harvest.py first")

    prompts, total = load_prompts(args.inp, args.max)
    if not prompts:
        sys.exit("no user prompts found in the harvested file")
    prompt = build_prompt(prompts)

    if args.method == "paste":
        out = args.out.with_name("taxonomy_prompt.txt")
        out.write_text(prompt)
        print(f"Wrote {out}")
        capped = f" (capped at {args.max}; raise --max for more)" if total > args.max else ""
        print(f"  {len(prompts)} of {total} unique prompts included{capped}")
        print("\nNext:")
        print("  1. Paste the whole file into Claude or ChatGPT.")
        print("  2. Save its JSON reply as data/taxonomy.json.")
        print("  3. Pick the task types worth an eval, then build your dataset (Step 3).")
        return

    # llm: send it and write taxonomy.json for them
    sys.path.insert(0, str(DIR / "scripts"))
    from openai_client import LocalLLM

    llm = LocalLLM(
        base_url=args.base_url,
        api_key=args.api_key or "not-needed-for-local",
        model=args.model or "local",
    )
    taxonomy = extract_json_array(llm.chat(prompt))
    if not taxonomy:
        sys.exit("model returned no parseable JSON array — try --method paste, or a stronger model")
    args.out.write_text(json.dumps(taxonomy, indent=2))

    print(f"Wrote {args.out} — your task types, ranked:\n")
    for i, t in enumerate(taxonomy, 1):
        print(f"  {i:2d}. {t.get('task_type', '?'):24s} {t.get('count', '?')}×")
    print("\nPick the task types worth an eval, then build your dataset (Step 3).")


if __name__ == "__main__":
    main()
