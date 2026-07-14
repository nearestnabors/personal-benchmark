"""Harvest your real, recurring prompts from Claude and ChatGPT usage.

Each source is normalized to one common record:

    {
      "source":     "claude_history" | "claude_transcript" | "codex" |
                    "chatgpt_export" | "claude_export",
      "timestamp":  ISO-8601 string (UTC) or null,
      "role":       "user" | "assistant",
      "text":       redacted message text,
      "session_id": string or null,
      "project":    string or null   # working dir / conversation title when known
    }

Only USER records are needed downstream (they are the tasks you actually ask
for), but assistant turns are kept for context.

Every field of text passes through ``redact.redact`` before it is written, so
secrets and PII never land on disk in the clear.

The per-line formats of the CLI transcripts are INTERNAL and change between
tool versions (this is documented by both Claude Code and Codex). We therefore
parse defensively: unknown line shapes are skipped, never assumed. For Claude,
prefer ``~/.claude/history.jsonl`` (your raw prompts) over the project
transcripts (mostly tool-result noise).

Usage
-----
    # Auto-detect the standard local locations:
    python scripts/harvest.py --out data/harvested

    # Run against the committed fixtures (no private data, good for CI/tests):
    python scripts/harvest.py --fixtures --out /tmp/harvested

    # Point at an emailed app export (.zip) from ChatGPT or Claude:
    python scripts/harvest.py --chatgpt-zip ~/Downloads/chatgpt-export.zip --out data/harvested
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact import redact  # noqa: E402

Record = dict[str, Any]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _iso(ts: Any) -> str | None:
    """Coerce an epoch (s or ms) or ISO string into an ISO-8601 UTC string."""
    if ts is None:
        return None
    if isinstance(ts, str):
        # Assume already ISO-ish; return as-is rather than risk misparsing.
        return ts
    if isinstance(ts, (int, float)):
        # Heuristic: values past ~year 2001 in seconds are < 1e10; ms are larger.
        seconds = ts / 1000 if ts > 1e11 else ts
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _record(
    source: str, role: str, text: str, *, timestamp=None, session_id=None, project=None
) -> Record | None:
    """Build a normalized, redacted record. Returns None for empty text."""
    text = (text or "").strip()
    if not text:
        return None
    return {
        "source": source,
        "timestamp": _iso(timestamp),
        "role": role,
        "text": redact(text),
        "session_id": session_id,
        "project": project,
    }


def _read_jsonl(lines: Iterable[str]) -> Iterator[dict]:
    """Yield parsed JSON objects from JSONL lines, skipping blanks/garbage."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _text_from_content(content: Any) -> str:
    """Flatten a message 'content' that may be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Claude: {"type":"text","text":...}; Codex: {"type":"input_text","text":...}
                if block.get("type") in {"text", "input_text", "output_text"} and isinstance(
                    block.get("text"), str
                ):
                    parts.append(block["text"])
        return "\n".join(parts)
    return ""


# --------------------------------------------------------------------------- #
# Source parsers
# --------------------------------------------------------------------------- #
def parse_claude_history(path: Path) -> Iterator[Record]:
    """~/.claude/history.jsonl — the cleanest source of your raw prompts.

    Schema (verified): {display, pastedContents, project, sessionId, timestamp(ms)}.
    """
    for obj in _read_jsonl(path.read_text(errors="ignore").splitlines()):
        rec = _record(
            "claude_history",
            "user",
            obj.get("display", ""),
            timestamp=obj.get("timestamp"),
            session_id=obj.get("sessionId"),
            project=obj.get("project"),
        )
        if rec:
            yield rec


def parse_claude_transcript(path: Path) -> Iterator[Record]:
    """~/.claude/projects/<proj>/<session>.jsonl — defensive fallback.

    Most 'user' lines here are tool-results, not human prompts; we keep only
    genuine text blocks. Assistant turns are kept for context.
    """
    project = path.parent.name
    for obj in _read_jsonl(path.read_text(errors="ignore").splitlines()):
        if obj.get("type") not in {"user", "assistant"}:
            continue
        message = obj.get("message") or {}
        role = message.get("role") or obj.get("type")
        if role not in {"user", "assistant"}:
            continue
        text = _text_from_content(message.get("content"))
        rec = _record(
            "claude_transcript",
            role,
            text,
            timestamp=obj.get("timestamp"),
            session_id=obj.get("sessionId"),
            project=project,
        )
        if rec:
            yield rec


def parse_codex_rollout(path: Path) -> Iterator[Record]:
    """~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl[.zst].

    First line is a session_meta block. User turns are
    response_item/message/role=user with content=[{type:input_text,text}].
    """
    if path.suffix == ".zst":
        try:
            import zstandard  # type: ignore
        except ImportError:
            print(f"  ! skipping {path.name}: install 'zstandard' to read .zst", file=sys.stderr)
            return
        with path.open("rb") as fh:
            raw = zstandard.ZstdDecompressor().stream_reader(fh).read().decode("utf-8", "ignore")
        lines = raw.splitlines()
    else:
        lines = path.read_text(errors="ignore").splitlines()

    session_id = None
    for obj in _read_jsonl(lines):
        if obj.get("type") == "session_meta":
            payload = obj.get("payload") or {}
            session_id = payload.get("id") or payload.get("session_id")
            continue
        if obj.get("type") != "response_item":
            continue
        payload = obj.get("payload") or {}
        if payload.get("type") != "message":
            continue
        role = payload.get("role")
        if role not in {"user", "assistant"}:
            continue
        rec = _record(
            "codex",
            role,
            _text_from_content(payload.get("content")),
            timestamp=obj.get("timestamp"),
            session_id=session_id,
        )
        if rec:
            yield rec


def parse_chatgpt_export(conversations: list) -> Iterator[Record]:
    """conversations.json from a ChatGPT 'Export data' archive.

    Each conversation has a 'mapping' of message nodes; we walk them and emit
    user/assistant turns. Structure is defensive against version drift.
    """
    for convo in conversations or []:
        if not isinstance(convo, dict):
            continue
        title = convo.get("title")
        mapping = convo.get("mapping") or {}
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            message = node.get("message") or {}
            author = (message.get("author") or {}).get("role")
            if author not in {"user", "assistant"}:
                continue
            content = message.get("content") or {}
            parts = content.get("parts") if isinstance(content, dict) else None
            text = "\n".join(p for p in (parts or []) if isinstance(p, str))
            rec = _record(
                "chatgpt_export",
                author,
                text,
                timestamp=message.get("create_time"),
                session_id=convo.get("id") or convo.get("conversation_id"),
                project=title,
            )
            if rec:
                yield rec


def parse_claude_export(conversations: list) -> Iterator[Record]:
    """conversations.json from a Claude app 'Export data' archive.

    Each conversation has 'chat_messages' with sender human/assistant.
    """
    for convo in conversations or []:
        if not isinstance(convo, dict):
            continue
        title = convo.get("name") or convo.get("title")
        convo_id = convo.get("uuid") or convo.get("id")
        for message in convo.get("chat_messages") or []:
            if not isinstance(message, dict):
                continue
            sender = message.get("sender")
            role = {"human": "user", "assistant": "assistant"}.get(sender)
            if role is None:
                continue
            text = message.get("text") or _text_from_content(message.get("content"))
            rec = _record(
                "claude_export",
                role,
                text,
                timestamp=message.get("created_at"),
                session_id=convo_id,
                project=title,
            )
            if rec:
                yield rec


def _load_export_zip(zip_path: Path, filename: str) -> list:
    """Extract and JSON-parse `filename` from an export .zip (handles nesting)."""
    with zipfile.ZipFile(zip_path) as zf:
        match = next((n for n in zf.namelist() if n.endswith(filename)), None)
        if match is None:
            print(f"  ! {filename} not found in {zip_path.name}", file=sys.stderr)
            return []
        with zf.open(match) as fh:
            return json.loads(fh.read().decode("utf-8", "ignore"))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def harvest(args: argparse.Namespace) -> list[Record]:
    records: list[Record] = []

    if args.claude_history and args.claude_history.exists():
        records.extend(parse_claude_history(args.claude_history))

    if args.claude_projects and args.claude_projects.exists():
        for transcript in sorted(args.claude_projects.rglob("*.jsonl")):
            records.extend(parse_claude_transcript(transcript))

    if args.codex_dir and args.codex_dir.exists():
        for rollout in sorted(args.codex_dir.rglob("rollout-*.jsonl*")):
            records.extend(parse_codex_rollout(rollout))

    if args.chatgpt_zip and args.chatgpt_zip.exists():
        records.extend(
            parse_chatgpt_export(_load_export_zip(args.chatgpt_zip, "conversations.json"))
        )

    if args.claude_zip and args.claude_zip.exists():
        records.extend(parse_claude_export(_load_export_zip(args.claude_zip, "conversations.json")))

    return records


def _fixture_args(fixtures: Path, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        claude_history=fixtures / "claude_history.jsonl",
        claude_projects=fixtures / "claude_projects",
        codex_dir=fixtures / "codex_sessions",
        chatgpt_zip=None,
        claude_zip=None,
        out=out,
    )


def main() -> None:
    home = Path.home()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("data/harvested"), help="output directory")
    p.add_argument(
        "--fixtures",
        action="store_true",
        help="harvest the committed fixtures instead of real data",
    )
    p.add_argument("--claude-history", type=Path, default=home / ".claude/history.jsonl")
    p.add_argument("--claude-projects", type=Path, default=home / ".claude/projects")
    p.add_argument("--codex-dir", type=Path, default=home / ".codex/sessions")
    p.add_argument("--chatgpt-zip", type=Path, default=None)
    p.add_argument("--claude-zip", type=Path, default=None)
    args = p.parse_args()

    if args.fixtures:
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        args = _fixture_args(fixtures, args.out)

    records = harvest(args)
    args.out.mkdir(parents=True, exist_ok=True)

    # Per-source files plus a combined stream.
    by_source: dict[str, list[Record]] = {}
    for rec in records:
        by_source.setdefault(rec["source"], []).append(rec)

    for source, recs in by_source.items():
        (args.out / f"{source}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    combined = args.out / "all.jsonl"
    combined.write_text("".join(json.dumps(r) + "\n" for r in records))

    users = sum(1 for r in records if r["role"] == "user")
    print(f"Harvested {len(records)} records ({users} user) from {len(by_source)} source(s):")
    for source, recs in sorted(by_source.items()):
        u = sum(1 for r in recs if r["role"] == "user")
        print(f"  {source:20s} {len(recs):5d} records ({u} user)")
    print(f"Wrote {combined}")


if __name__ == "__main__":
    main()
