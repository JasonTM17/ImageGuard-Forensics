"""Redact host paths and URLs before exception text reaches the UI."""

from __future__ import annotations

import re
from pathlib import Path

from imageguard.ui import strings

_RE_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"']+")
_RE_QUOTED_PATH = re.compile(
    r'"(?:[A-Za-z]:[\\/]|\\\\|/)[^"]*"|'
    r"'(?:[A-Za-z]:[\\/]|\\\\|/)[^']*'"
)
_RE_UNC_PATH = re.compile(r"\\\\[^\\]+(?:\\[^\\\"'\n\r]+)+")
_RE_WIN_ABS_PATH = re.compile(
    r"[A-Za-z]:(?:[\\/][^\\/\"'\n\r]+)+"
)
_RE_POSIX_ABS_PATH = re.compile(r"/(?:[^/\\\"'\n\r]+(?:/[^/\\\"'\n\r]+)+)")
_RE_PATH_LIKE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|tmp|var|etc)(?:/|\\))"
)


def _path_match_to_basename(match: re.Match[str]) -> str:
    token = match.group(0).strip("\"'")
    return Path(token.replace("\\", "/")).name


def redact_paths_in_message(raw: str) -> str:
    """Thay URL và đường dẫn tuyệt đối bằng placeholder/tên file an toàn."""
    if raw is None:
        raise ValueError("message must be a string")
    out = _RE_URL.sub("<url>", raw)
    for pattern in (
        _RE_QUOTED_PATH,
        _RE_UNC_PATH,
        _RE_WIN_ABS_PATH,
        _RE_POSIX_ABS_PATH,
    ):
        out = pattern.sub(_path_match_to_basename, out)
    return out


def sanitize_exception_text(exc: BaseException) -> str:
    """Rút gọn thông báo lỗi — không lộ URL hoặc đường dẫn host."""
    raw = str(exc).strip()
    if not raw:
        return f"{type(exc).__name__}: {strings.MSG_EXCEPTION_PATH_REDACTED}"
    sanitized = redact_paths_in_message(raw)
    if _RE_PATH_LIKE.search(sanitized):
        return f"{type(exc).__name__}: {strings.MSG_EXCEPTION_PATH_REDACTED}"
    return f"{type(exc).__name__}: {sanitized}"
