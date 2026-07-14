"""Draft a golden dataset from your REAL prompts — not synthetic ones.

Hand-writing eval examples is slow, but you also don't want a model inventing
fake tasks. So this feeds your ACTUAL harvested requests to a model and asks it
only to (a) keep each request verbatim as the eval `input`, (b) pick a scoring
kind, and (c) write the reference/rubric. It must not invent or reword inputs,
and it skips requests that can't be evaluated on their own.

Output rows match the schema build_dataset.py expects:

    { "task_type", "scoring": exact|structured|judge, "input" (a real request),
      "reference" (exact/structured) | "rubric" (judge), "label": "DRAFT" }

IMPORTANT: a drafted dataset is NOT golden until you review it. Models get
"correct" answers wrong, and a bad reference silently corrupts every experiment.
Skim the draft, fix references, drop weak rows — THEN build_dataset.py it.

Two methods (same as taxonomy.py):

  paste (default) — write a copy-paste prompt to hand to Claude/ChatGPT. No API
                    key. Save the reply as data/dataset.jsonl.
  llm             — call an OpenAI-compatible endpoint (local llama-server, or a
                    hosted API via --base-url/--model/--api-key) and write
                    data/dataset.jsonl directly.

    python scripts/make_dataset.py --in data/harvested/all.jsonl --max 40
    python scripts/make_dataset.py --in data/harvested/all.jsonl --method llm \\
        --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key $OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent

_RULES = """Rules:
- input: use one of MY requests below EXACTLY as written. Do NOT invent,
  paraphrase, translate, merge, or trim inputs. You may DROP a request, but you
  may never rewrite one.
- SKIP any request that can't be graded on its own: it points at a paste, file,
  screenshot, or earlier turn not shown here ("[Pasted text ...]", "this log",
  "the file above"), or is too vague ("continue", "go on", "/login").
- Choose scoring HONESTLY:
    exact       -> answer is ONE short canonical string. Put it in "reference".
    structured  -> answer is a specific JSON object. Put it in "reference".
    judge       -> many good answers / subjective. Put a strict pass-or-fail
                   "rubric", and set "reference" to null.
- Use exact/structured ONLY when there is genuinely one correct answer you are
  sure of. If unsure, use judge. Never invent a reference you cannot verify.
- task_type: a short snake_case label for the kind of request.
- Aim for a mix of scoring kinds. Fewer good rows beats many shaky ones."""

_SCHEMA = (
    '{"task_type": "...", "scoring": "exact|structured|judge", "input": "<one of my requests, verbatim>", '
    '"reference": <value or null>, "rubric": <string or null>, "label": "DRAFT"}'
)


def load_real_prompts(path: Path, min_chars: int, cap: int) -> list[str]:
    """Real user requests from the harvested file: dedup, drop tiny ones, cap."""
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    seen: set[str] = set()
    prompts: list[str] = []
    for r in records:
        if r.get("role") != "user":
            continue
        text = (r.get("text") or "").strip()
        if len(text) < min_chars or text in seen:
            continue
        seen.add(text)
        prompts.append(text)
    return prompts[:cap]


def build_prompt(prompts: list[str]) -> str:
    lines = [
        "You are helping me build a GOLDEN EVAL DATASET for my personal AI benchmark.",
        "Below are REAL requests I have actually made. Turn the gradable ones into eval",
        "examples. Use my wording verbatim — do not invent new tasks.",
        "",
        "Write JSON Lines (one JSON object per line), using EXACTLY these keys:",
        "",
        "  " + _SCHEMA,
        "",
        _RULES,
        "",
        "Output ONLY the JSON Lines — no prose, no code fences.",
        "",
        "My real requests:",
    ]
    lines += [f"{i}. {p}" for i, p in enumerate(prompts, 1)]
    return "\n".join(lines) + "\n"


def parse_jsonl(text: str) -> list[dict]:
    """Extract valid JSON objects from a model reply, tolerating fences/prose."""
    rows = []
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--in", dest="inp", type=Path, default=DIR / "data/harvested/all.jsonl"
    )
    p.add_argument(
        "--max", type=int, default=40, help="max real prompts to offer the model"
    )
    p.add_argument(
        "--min-chars", type=int, default=25, help="skip requests shorter than this"
    )
    p.add_argument("--method", choices=["paste", "llm"], default="paste")
    p.add_argument("--base-url", default="http://127.0.0.1:8102/v1")
    p.add_argument("--model", default=None)
    p.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
    )
    p.add_argument("--out", type=Path, default=DIR / "data/dataset.jsonl")
    args = p.parse_args()

    if not args.inp.exists():
        sys.exit(f"no harvested file at {args.inp} — run scripts/harvest.py first")

    prompts = load_real_prompts(args.inp, args.min_chars, args.max)
    if not prompts:
        sys.exit("no usable real prompts found (try lowering --min-chars)")
    print(f"Offering {len(prompts)} of your real requests to the model.")

    prompt = build_prompt(prompts)

    if args.method == "paste":
        out = args.out.with_name("dataset_prompt.txt")
        out.write_text(prompt)
        print(f"\nWrote {out}")
        print("Next:")
        print("  1. Paste it into Claude or ChatGPT.")
        print(f"  2. Save the JSON-Lines reply as {args.out}.")
        print("  3. REVIEW it — fix any wrong references, drop weak rows.")
        print("  4. python scripts/build_dataset.py --path data/dataset.jsonl")
        return

    # llm: generate directly
    sys.path.insert(0, str(DIR / "scripts"))
    from openai_client import LocalLLM

    llm = LocalLLM(
        base_url=args.base_url,
        api_key=args.api_key or "not-needed-for-local",
        model=args.model or "local",
    )
    rows = parse_jsonl(llm.chat(prompt))
    if not rows:
        sys.exit(
            "model returned no parseable JSON Lines — try --method paste, or a stronger model"
        )
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\nWrote {len(rows)} DRAFT examples to {args.out}")
    print("REVIEW them (especially references), then:")
    print("  python scripts/build_dataset.py --path data/dataset.jsonl")


if __name__ == "__main__":
    main()
