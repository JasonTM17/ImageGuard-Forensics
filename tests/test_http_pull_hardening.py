from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from imageguard.ingest import http_pull
from imageguard.ingest.http_pull import pull_due


@pytest.fixture(autouse=True)
def _reset_pull_due_state() -> Iterator[None]:
    with http_pull._LAST_ATTEMPTS_LOCK:
        state = getattr(pull_due, "_last_attempts", None)
        if state is not None:
            state.clear()
    yield


def _config(url: str, **overrides: object) -> dict:
    config = {
        "enabled": True,
        "interval_s": 30,
        "timeout_s": 2,
        "max_retries": 0,
        "allow_loopback": True,
        "allow_link_local": False,
        "hosts": ["127.0.0.1"],
        "urls": [url],
        "username_env": "",
        "password_env": "",
        "secrets_file": "",
    }
    config.update(overrides)
    return config


@contextmanager
def _serve(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "hostname",
    [
        "metadata.google.internal",
        "metadata.google.com",
    ],
)
def test_metadata_hostnames_blocked_even_when_link_local_allowed(
    tmp_path: Path,
    hostname: str,
) -> None:
    result = pull_due(
        _config(
            f"http://{hostname}/latest/meta-data/",
            hosts=[hostname],
            allow_link_local=True,
        ),
        tmp_path,
    )

    assert result == [{"status": "refused", "reason": "metadata_blocked"}]
    assert list(tmp_path.iterdir()) == []


def test_metadata_link_local_ip_blocked_even_when_link_local_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        http_pull.socket,
        "getaddrinfo",
        lambda host, port, type: [
            (http_pull.socket.AF_INET, http_pull.socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))
        ],
    )
    result = pull_due(
        _config(
            "http://camera.example/shot.jpg",
            hosts=["camera.example"],
            allow_link_local=True,
        ),
        tmp_path,
    )

    assert result == [{"status": "refused", "reason": "metadata_blocked"}]
    assert list(tmp_path.iterdir()) == []


def test_non_metadata_link_local_allowed_when_flag_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        http_pull.socket,
        "getaddrinfo",
        lambda host, port, type: [
            (http_pull.socket.AF_INET, http_pull.socket.SOCK_STREAM, 6, "", ("169.254.10.2", port))
        ],
    )
    parsed, host, port = http_pull._validate_target(
        "http://camera.example/shot.jpg",
        {"camera.example"},
        allow_loopback=False,
        allow_link_local=True,
    )

    assert host == "camera.example"
    assert port == 80
    assert parsed.hostname == "camera.example"


def test_total_attempt_deadline_stops_trickling_body(tmp_path: Path) -> None:
    payload = b"\xff\xd8\xff" + b"x" * 50

    class TrickleHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            for offset in range(len(payload)):
                self.wfile.write(payload[offset : offset + 1])
                self.wfile.flush()
                time.sleep(0.15)

        def log_message(self, format: str, *args: object) -> None:
            return

    with _serve(TrickleHandler) as base_url:
        started = time.monotonic()
        result = pull_due(
            _config(f"{base_url}/slow.jpg", timeout_s=1, max_retries=0),
            tmp_path,
        )
        elapsed = time.monotonic() - started

    assert result == [{"status": "refused", "reason": "timeout"}]
    assert elapsed < 5.0
    assert list(tmp_path.glob("*.jpg")) == []


def test_one_attempt_rejects_non_positive_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        http_pull._one_attempt(
            "http://127.0.0.1:9/missing.jpg",
            tmp_path,
            allowed_hosts={"127.0.0.1"},
            allow_loopback=True,
            allow_link_local=False,
            timeout_s=0,
            authorization=None,
            initial_origin=("http", "127.0.0.1", 9),
            max_bytes=1024,
            secret_values=(),
        )
