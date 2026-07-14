"""Surface your RECURRING task types from harvested prompts.

You can't recall these from memory — that's the point. This groups your user
requests into task types, ranks them by how often you actually do them, and
prints representative examples so you can pick which ones are worth an eval.

Two methods, swappable with --method:

  keyword  (default, offline, zero deps) — bucket prompts by simple verb/keyword
           heuristics. Crude but instant and fully local.
  llm      — ask a local model (llama-server) to label the task type of each
           prompt. Slower, needs a served model, but far better grouping.

    python scripts/taxonomy.py --in data/harvested/all.jsonl
    python scripts/taxonomy.py --in data/harvested/all.jsonl --method llm --base-url http://127.0.0.1:8102/v1

Human-in-the-loop: this only PRINTS the ranked list and writes taxonomy.json.
You choose which task types to carry into your dataset. It does not auto-decide.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent

# Ordered heuristic buckets: first match wins. Intentionally simple and legible.
_KEYWORD_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("extract_fields", ("extract", "pull", "parse", "into json", "as json", "fields")),
    ("summarize", ("summarize", "summary", "tl;dr", "recap", "bullet")),
    (
        "triage_error",
        ("error", "stack trace", "traceback", "failed", "failing", "triage", "log"),
    ),
    ("rewrite", ("rewrite", "reword", "make it", "more concise", "tone", "polish", "draft")),
    ("classify", ("classify", "categorize", "label", "which category", "intent")),
    ("explain", ("explain", "what does", "how does", "why does", "understand")),
    ("write_code", ("write a", "implement", "function", "script", "refactor", "add a test")),
]


def bucket_keyword(text: str) -> str:
    low = text.lower()
    for name, keys in _KEYWORD_BUCKETS:
        if any(k in low for k in keys):
            return name
    return "other"


def bucket_llm(texts: list[str], base_url: str) -> list[str]:
    sys.path.insert(0, str(DIR / "scripts"))
    from openai_client import LocalLLM

    llm = LocalLLM(base_url=base_url)
    labels = []
    for t in texts:
        prompt = (
            "Label this user request with a short snake_case task type "
            "(e.g. extract_fields, summarize, triage_error, rewrite_email). "
            "Reply with only the label.\n\nRequest: " + t[:500]
        )
        labels.append(llm.chat(prompt).strip().split()[0].lower() if t else "other")
    return labels


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="inp", type=Path, default=DIR / "data/harvested/all.jsonl")
    p.add_argument("--method", choices=["keyword", "llm"], default="keyword")
    p.add_argument(
        "--base-url", default="http://127.0.0.1:8102/v1", help="local model for --method llm"
    )
    p.add_argument("--out", type=Path, default=DIR / "data/taxonomy.json")
    p.add_argument("--examples", type=int, default=3, help="representative examples per task type")
    args = p.parse_args()

    if not args.inp.exists():
        sys.exit(f"no harvested file at {args.inp} — run scripts/harvest.py first")

    records = [json.loads(line) for line in args.inp.read_text().splitlines() if line.strip()]
    prompts = [r["text"] for r in records if r.get("role") == "user" and r.get("text")]
    if not prompts:
        sys.exit("no user prompts found in the harvested file")

    if args.method == "llm":
        labels = bucket_llm(prompts, args.base_url)
    else:
        labels = [bucket_keyword(t) for t in prompts]

    counts = Counter(labels)
    examples: dict[str, list[str]] = defaultdict(list)
    for text, label in zip(prompts, labels):
        if len(examples[label]) < args.examples:
            examples[label].append(text[:120])

    taxonomy = [
        {"task_type": label, "count": count, "representative_examples": examples[label]}
        for label, count in counts.most_common()
    ]
    args.out.write_text(json.dumps(taxonomy, indent=2))

    print(f"Ranked task types from {len(prompts)} prompts ({args.method}):\n")
    for i, t in enumerate(taxonomy, 1):
        print(f"  {i:2d}. {t['task_type']:16s} {t['count']:4d}×")
        for ex in t["representative_examples"]:
            print(f"        - {ex}")
    print(f"\nWrote {args.out}. Pick the task types worth an eval, then build your dataset.")


if __name__ == "__main__":
    main()
