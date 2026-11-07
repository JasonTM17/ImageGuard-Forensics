"""Course lock: cancelled is false until a cooperative event is set."""

from __future__ import annotations

import threading

from imageguard.forensics.base import cancelled


def test_cancelled_none_unset_and_set() -> None:
    assert cancelled(None) is False
    event = threading.Event()
    assert cancelled(event) is False
    event.set()
    assert cancelled(event) is True
