"""Surface your RECURRING task types from harvested prompts.

You can't recall these from memory — that's the point. This groups your user
requests into task types, ranks them by how often you actually do them, and
prints representative examples so you can pick which ones are worth an eval.

Three methods, swappable with --method:

  keyword  (default, offline, zero deps) — bucket prompts by simple verb/keyword
           heuristics. Crude and dumps a lot into "other" on real history; treat
           it as a rough first pass you curate, not ground truth.
  paste    — emit a copy-paste-ready prompt (your redacted prompts + a strict
           JSON schema) to hand to a BIG model like Claude or ChatGPT. No API key
           needed and by far the best grouping. Save the model's JSON reply as
           taxonomy.json.
  llm      — call an OpenAI-compatible endpoint to label each prompt. Point
           --base-url at a local llama-server or a hosted API.

    python scripts/taxonomy.py --in data/harvested/all.jsonl
    python scripts/taxonomy.py --in data/harvested/all.jsonl --method paste
    python scripts/taxonomy.py --in data/harvested/all.jsonl --method llm --base-url http://127.0.0.1:8102/v1

Human-in-the-loop: this only PRINTS the ranked list and writes taxonomy.json.
You choose which task types to carry into your dataset. It does not auto-decide.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent

# Ordered heuristic buckets: first match wins. Intentionally simple and legible.
#
# Keyword matching is deliberately crude. On real coding-agent history it WILL
# mislabel and dump a lot into "other" — treat the output as a rough starting
# point you curate by hand, not ground truth. For real grouping use --method llm.
# Single-word keys match on WORD BOUNDARIES (so "log" won't match "login"); keys
# containing a space match as a literal phrase.
_KEYWORD_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("extract_fields", ("extract", "parse", "into json", "as json", "field", "fields")),
    ("summarize", ("summarize", "summary", "tl;dr", "recap", "bullet", "bullets")),
    (
        "triage_error",
        (
            "error",
            "errors",
            "exception",
            "stack trace",
            "traceback",
            "failing",
            "triage",
            "log",
            "logs",
        ),
    ),
    ("rewrite", ("rewrite", "reword", "rephrase", "more concise", "tone", "polish", "proofread")),
    ("classify", ("classify", "categorize", "categorise", "which category", "intent")),
    ("explain", ("explain", "what does", "how does", "why does", "walk me through")),
    (
        "write_code",
        ("implement", "refactor", "write a test", "write tests", "add a test", "fix the bug"),
    ),
]


def _matches(low: str, key: str) -> bool:
    """Phrase keys match literally; single-word keys match on word boundaries."""
    if " " in key:
        return key in low
    return re.search(rf"\b{re.escape(key)}\b", low) is not None


def bucket_keyword(text: str) -> str:
    low = text.lower()
    for name, keys in _KEYWORD_BUCKETS:
        if any(_matches(low, k) for k in keys):
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


_PASTE_HEADER = """I'm analyzing my own AI-assistant usage to build a personal eval set.
Below are {n} requests I've actually made (secrets/PII already redacted).

Group them into 8-15 recurring TASK TYPES. Reply with ONLY a JSON array, no prose,
ranked by count (most frequent first). Each item must look like:

  {{"task_type": "extract_fields", "count": 42,
    "representative_examples": ["<verbatim request>", "<verbatim request>"]}}

Use short snake_case task_type names. representative_examples must be 2-3 real
requests copied verbatim from the list. Here are my requests:
"""


def emit_paste_prompt(prompts: list[str], out: Path, limit: int) -> None:
    """Write a copy-paste-ready prompt for a big model (Claude, ChatGPT, ...).

    Grouping messy real history is exactly what a large model is good at, and it
    needs no API key: you paste this file in, then save the JSON reply as
    taxonomy.json. Your prompts are already redacted, but you are sending them to
    a third-party model — that's your call to make.
    """
    # De-duplicate while preserving order, then cap so it stays pasteable.
    seen: set[str] = set()
    unique = [p for p in prompts if not (p in seen or seen.add(p))]
    sample = unique[:limit]
    body = "\n".join(f"{i}. {p}" for i, p in enumerate(sample, 1))
    out.write_text(_PASTE_HEADER.format(n=len(sample)) + "\n" + body + "\n")
    print(f"Wrote {out}")
    print(
        f"  {len(sample)} of {len(unique)} unique prompts included"
        + (f" (capped at {limit}; raise --max-paste for more)" if len(unique) > limit else "")
    )
    print("\nNext:")
    print("  1. Paste the whole file into Claude or ChatGPT.")
    print("  2. Save its JSON reply as data/taxonomy.json.")
    print("  3. Pick the task types worth an eval and build your dataset.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="inp", type=Path, default=DIR / "data/harvested/all.jsonl")
    p.add_argument(
        "--method",
        choices=["keyword", "llm", "paste"],
        default="keyword",
        help="keyword=offline heuristic; llm=local/OpenAI-compatible endpoint; "
        "paste=emit a prompt to hand to a big model (Claude/ChatGPT)",
    )
    p.add_argument(
        "--base-url", default="http://127.0.0.1:8102/v1", help="local model for --method llm"
    )
    p.add_argument("--out", type=Path, default=DIR / "data/taxonomy.json")
    p.add_argument("--examples", type=int, default=3, help="representative examples per task type")
    p.add_argument("--max-paste", type=int, default=400, help="max prompts for --method paste")
    args = p.parse_args()

    if not args.inp.exists():
        sys.exit(f"no harvested file at {args.inp} — run scripts/harvest.py first")

    records = [json.loads(line) for line in args.inp.read_text().splitlines() if line.strip()]
    prompts = [r["text"] for r in records if r.get("role") == "user" and r.get("text")]
    if not prompts:
        sys.exit("no user prompts found in the harvested file")

    if args.method == "paste":
        emit_paste_prompt(prompts, args.out.with_name("taxonomy_prompt.txt"), args.max_paste)
        return

    if args.method == "llm":
        labels = bucket_llm(prompts, args.base_url)
    else:
        labels = [bucket_keyword(t) for t in prompts]

    counts = Counter(labels)
    examples: dict[str, list[str]] = defaultdict(list)
    for text, label in zip(prompts, labels):
        if len(examples[label]) < args.examples:
            examples[label].append(text)  # keep the FULL prompt in the JSON

    taxonomy = [
        {"task_type": label, "count": count, "representative_examples": examples[label]}
        for label, count in counts.most_common()
    ]
    args.out.write_text(json.dumps(taxonomy, indent=2))

    print(f"Ranked task types from {len(prompts)} prompts ({args.method}):\n")
    for i, t in enumerate(taxonomy, 1):
        print(f"  {i:2d}. {t['task_type']:16s} {t['count']:4d}×")
        for ex in t["representative_examples"]:
            oneline = ex.replace("\n", " ")
            print(f"        - {oneline[:120]}{'…' if len(oneline) > 120 else ''}")
    print(f"\nWrote {args.out}. Pick the task types worth an eval, then build your dataset.")


if __name__ == "__main__":
    main()
