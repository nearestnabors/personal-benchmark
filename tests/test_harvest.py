"""Unit tests for the harvester + redaction, run against committed fixtures.

    cd tutorials/personal-benchmark
    python -m pytest tests/ -q

No private data required — everything here uses fixtures/.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import harvest  # noqa: E402
from redact import redact, redaction_summary  # noqa: E402

FIXTURES = ROOT / "fixtures"


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #
def test_redacts_common_secrets():
    text = "key sk-abcd1234efgh5678ijkl and mail me at a@b.com"
    out = redact(text)
    assert "sk-abcd1234efgh5678ijkl" not in out
    assert "a@b.com" not in out
    assert "REDACTED_OPENAI_KEY" in out
    assert "REDACTED_EMAIL" in out


def test_redaction_summary_counts():
    summary = redaction_summary("a@b.com and c@d.com")
    assert summary.get("email") == 2


def test_redact_is_idempotent_on_clean_text():
    clean = "Summarize this thread into three bullets."
    assert redact(clean) == clean


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def test_claude_history_parser():
    records = list(harvest.parse_claude_history(FIXTURES / "claude_history.jsonl"))
    assert len(records) == 12
    assert all(r["role"] == "user" for r in records)
    assert all(r["source"] == "claude_history" for r in records)
    # Timestamps become ISO strings.
    assert all(r["timestamp"].startswith("20") for r in records)
    # No raw secret survived into any record.
    blob = " ".join(r["text"] for r in records)
    assert "sk-abcd1234" not in blob
    assert "@example.com" not in blob


def test_claude_transcript_keeps_text_drops_tool_results():
    path = FIXTURES / "claude_projects" / "-Users-demo-inbox-tools" / "fix-sess-1.jsonl"
    records = list(harvest.parse_claude_transcript(path))
    roles = [r["role"] for r in records]
    # 1 user text + 1 assistant text; the tool_result-only user line is dropped.
    assert roles == ["user", "assistant"]
    assert records[0]["project"] == "-Users-demo-inbox-tools"


def test_codex_parser_skips_meta_and_extracts_user_turns():
    path = next((FIXTURES / "codex_sessions").rglob("rollout-*.jsonl"))
    records = list(harvest.parse_codex_rollout(path))
    user_turns = [r for r in records if r["role"] == "user"]
    assert len(user_turns) == 2
    assert all(r["source"] == "codex" for r in records)


def test_export_zip_parsers_roundtrip(tmp_path):
    # Build tiny export archives on the fly so we exercise the zip path.
    import json

    chatgpt = [
        {
            "title": "extract",
            "id": "c1",
            "mapping": {
                "n1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Extract to JSON"]},
                    }
                },
            },
        }
    ]
    claude = [
        {
            "name": "rewrite",
            "uuid": "u1",
            "chat_messages": [
                {
                    "sender": "human",
                    "text": "Rewrite this email",
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
        }
    ]
    cz = tmp_path / "chatgpt.zip"
    with zipfile.ZipFile(cz, "w") as zf:
        zf.writestr("conversations.json", json.dumps(chatgpt))
    az = tmp_path / "claude.zip"
    with zipfile.ZipFile(az, "w") as zf:
        zf.writestr("conversations.json", json.dumps(claude))

    cg = list(harvest.parse_chatgpt_export(harvest._load_export_zip(cz, "conversations.json")))
    cl = list(harvest.parse_claude_export(harvest._load_export_zip(az, "conversations.json")))
    assert cg[0]["text"] == "Extract to JSON" and cg[0]["source"] == "chatgpt_export"
    assert cl[0]["text"] == "Rewrite this email" and cl[0]["source"] == "claude_export"
