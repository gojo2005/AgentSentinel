"""Shared redaction for credentials in strings and structured values."""

import re
from typing import Any


_SECRET_KEY_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|password|secret|authorization|credential|"
    r"private[_-]?key|cookie)"
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"""(?i)((?:\b|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"""
    r"""token|password|secret|authorization|credential|private[_-]?key|cookie))"""
    r"""(\s*[:=]\s*)"""
    r"""(?:"[^"\r\n]*"|'[^'\r\n]*'|(?:bearer\s+)?[^\s,;]+)"""
)
_TOKEN_PATTERN = re.compile(
    r"(?i)\b(?:gsk_|sk-|sk_|xox[baprs]-|gh[pousr]_|github_pat_|AIza)"
    r"[A-Za-z0-9_-]{16,}\b"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"|\bBearer\s+[A-Za-z0-9._~+/-]+=*"
)


def redact_sensitive_data(value: Any, *, key: str = "") -> Any:
    """Recursively redact common credential fields and secret string formats."""
    if _SECRET_KEY_PATTERN.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(child_key): redact_sensitive_data(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    if isinstance(value, str):
        redacted = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1\2[REDACTED]", value)
        return _TOKEN_PATTERN.sub("[REDACTED]", redacted)
    return value
