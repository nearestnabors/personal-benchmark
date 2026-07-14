"""Secret / PII redaction for harvested prompts.

Runs BEFORE any harvested text is written to disk or embedded, so private data
never leaves the local machine in the clear. Patterns are intentionally
conservative: we would rather over-redact a workshop demo than leak a live key.

This is best-effort, not a compliance tool. Review `data/harvested/*.jsonl`
before sharing it.
"""

from __future__ import annotations

import re
from typing import Pattern

# (name, compiled pattern, replacement) applied in order.
_RULES: list[tuple[str, Pattern[str], str]] = [
    # Provider API keys (check these before the generic long-token rule).
    ("openai_key", re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "[REDACTED_OPENAI_KEY]"),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"), "[REDACTED_ANTHROPIC_KEY]"),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    ("google_key", re.compile(r"AIza[0-9A-Za-z_-]{35}"), "[REDACTED_GOOGLE_KEY]"),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"), "[REDACTED_GITHUB_TOKEN]"),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "[REDACTED_SLACK_TOKEN]"),
    # Bearer tokens and JWTs.
    ("bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}"), "Bearer [REDACTED_TOKEN]"),
    (
        "jwt",
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}"),
        "[REDACTED_JWT]",
    ),
    # Personal identifiers.
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED_CC]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    # Generic secrets assigned in text, e.g. API_KEY="...", token: ...
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\b\s*[:=]\s*['\"]?[A-Za-z0-9._/+-]{8,}['\"]?"
        ),
        r"\1=[REDACTED_SECRET]",
    ),
]


def redact(text: str) -> str:
    """Return ``text`` with secrets and PII replaced by labeled placeholders."""
    if not text:
        return text
    for _name, pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def redaction_summary(text: str) -> dict[str, int]:
    """Count matches per rule without mutating the input (useful for tests/logs)."""
    return {
        name: len(pattern.findall(text)) for name, pattern, _ in _RULES if pattern.findall(text)
    }


# Expose the rule names for docs/tests that want to enumerate coverage.
RULE_NAMES: list[str] = [name for name, _p, _r in _RULES]
