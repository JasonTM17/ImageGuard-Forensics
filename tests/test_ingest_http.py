from __future__ import annotations

import base64
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError

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


def _handler(
    body: bytes,
    *,
    content_type: str = "image/jpeg",
    status: int = 200,
    location: str | None = None,
    observed: dict[str, object] | None = None,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if observed is not None:
                observed["path"] = self.path
                observed["authorization"] = self.headers.get("Authorization")
            self.send_response(status)
            if location is not None:
                self.send_header("Location", location)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if status == 200:
                self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def _assert_no_temporary_files(directory: Path) -> None:
    assert list(directory.glob("*.tmp")) == []
    assert list(directory.glob(".*.tmp")) == []


def test_success_streams_and_publishes_complete_jpeg(
    tmp_path: Path, valid_jpg: Path
) -> None:
    payload = valid_jpg.read_bytes()
    with _serve(_handler(payload)) as base_url:
        result = pull_due(_config(f"{base_url}/shot.jpg"), tmp_path)

    assert result == [{"status": "written", "basename": "shot.jpg"}]
    saved = tmp_path / "shot.jpg"
    assert saved.read_bytes() == payload
    assert saved.read_bytes().startswith(b"\xff\xd8\xff")
    _assert_no_temporary_files(tmp_path)


def test_refuses_redirect_to_host_outside_allowlist(tmp_path: Path) -> None:
    with _serve(_handler(b"", status=302, location="http://not-allowed.invalid/shot.jpg")) as base_url:
        result = pull_due(_config(f"{base_url}/redirect"), tmp_path)

    assert result == [{"status": "refused", "reason": "redirect_host_not_allowed"}]
    assert list(tmp_path.glob("*.jpg")) == []
    _assert_no_temporary_files(tmp_path)


def test_stops_after_three_redirect_hops(tmp_path: Path) -> None:
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            self.send_response(302)
            self.send_header("Location", f"/{int(self.path[1:]) + 1}")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    with _serve(Handler) as base_url:
        result = pull_due(_config(f"{base_url}/0", max_retries=3), tmp_path)

    assert result == [{"status": "refused", "reason": "redirect_limit"}]
    assert paths == ["/0", "/1", "/2", "/3"]
    assert list(tmp_path.glob("*.jpg")) == []
    _assert_no_temporary_files(tmp_path)


def test_oversize_body_is_stopped_before_any_image_decoder(
    tmp_path: Path, valid_jpg: Path
) -> None:
    payload = valid_jpg.read_bytes()
    destination = tmp_path / "out"
    with _serve(_handler(payload)) as base_url:
        result = pull_due(_config(f"{base_url}/large.jpg"), destination, max_bytes=10)

    assert result == [{"status": "refused", "reason": "too_large"}]
    assert list(destination.glob("*.jpg")) == []
    assert "cv2" not in http_pull.__dict__
    _assert_no_temporary_files(destination)


def test_refuses_jpeg_bytes_with_wrong_content_type(tmp_path: Path) -> None:
    with _serve(_handler(b"\xff\xd8\xffpayload", content_type="text/plain")) as base_url:
        result = pull_due(_config(f"{base_url}/plain.jpg"), tmp_path)

    assert result == [{"status": "refused", "reason": "content_type_not_jpeg"}]
    assert list(tmp_path.glob("*.jpg")) == []
    _assert_no_temporary_files(tmp_path)


def test_refuses_non_jpeg_magic_even_with_jpeg_content_type(tmp_path: Path) -> None:
    with _serve(_handler(b"not a jpeg")) as base_url:
        result = pull_due(_config(f"{base_url}/wrong-magic.jpg"), tmp_path)

    assert result == [{"status": "refused", "reason": "invalid_jpeg"}]
    assert list(tmp_path.glob("*.jpg")) == []
    _assert_no_temporary_files(tmp_path)


def test_password_authentication_redacts_logs_results_and_userinfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "SENTINEL-PASSWORD-VALUE"
    monkeypatch.setenv("IG_TEST_CAMERA_USERNAME", "camera-user")
    monkeypatch.setenv("IG_TEST_CAMERA_PASSWORD", secret)
    observed: dict[str, object] = {}
    with _serve(_handler(b"\xff\xd8\xffpayload", observed=observed)) as base_url:
        url = f"{base_url}/{secret}.jpg"
        config = _config(
            url,
            username_env="IG_TEST_CAMERA_USERNAME",
            password_env="IG_TEST_CAMERA_PASSWORD",
        )
        with caplog.at_level(logging.INFO):
            result = pull_due(config, tmp_path)
            userinfo_result = pull_due(
                _config(
                    url.replace("http://", f"http://camera-user:{secret}@", 1),
                    username_env="IG_TEST_CAMERA_USERNAME",
                    password_env="IG_TEST_CAMERA_PASSWORD",
                ),
                tmp_path,
                now=100.0,
            )

    expected = base64.b64encode(f"camera-user:{secret}".encode()).decode()
    assert observed["authorization"] == f"Basic {expected}"
    assert result[0]["status"] == "written"
    assert result[0]["basename"].startswith("snapshot-")
    assert result[0]["basename"].endswith(".jpg")
    assert userinfo_result == [{"status": "refused", "reason": "userinfo_not_allowed"}]
    assert secret not in caplog.text
    assert secret not in repr(result)
    assert secret not in repr(userinfo_result)
    _assert_no_temporary_files(tmp_path)


def test_external_secrets_file_supplies_password(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "EXTERNAL-FILE-PASSWORD"
    secret_path = tmp_path / "camera-secret.txt"
    secret_path.write_text(secret, encoding="utf-8")
    monkeypatch.setenv("IG_TEST_CAMERA_USERNAME", "camera-user")
    observed: dict[str, object] = {}
    with _serve(_handler(b"\xff\xd8\xffpayload", observed=observed)) as base_url:
        config = _config(
            f"{base_url}/shot.jpg",
            username_env="IG_TEST_CAMERA_USERNAME",
            secrets_file=str(secret_path.resolve()),
        )
        with caplog.at_level(logging.INFO):
            result = pull_due(config, tmp_path / "out")

    expected = base64.b64encode(f"camera-user:{secret}".encode()).decode()
    assert observed["authorization"] == f"Basic {expected}"
    assert result == [{"status": "written", "basename": "shot.jpg"}]
    assert secret not in caplog.text
    assert secret not in repr(result)


def test_auth_exception_text_is_not_returned_or_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "SENTINEL-PASSWORD-VALUE"
    monkeypatch.setenv("IG_TEST_CAMERA_PASSWORD", secret)
    monkeypatch.setattr(
        http_pull.socket,
        "getaddrinfo",
        lambda host, port, type: [
            (http_pull.socket.AF_INET, http_pull.socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
        ],
    )

    def fail_with_secret(self, request, timeout=None):
        raise URLError(RuntimeError(f"request failed with {secret}"))

    monkeypatch.setattr("urllib.request.OpenerDirector.open", fail_with_secret)
    config = _config(
        "http://camera.example/exception.jpg",
        hosts=["camera.example"],
        password_env="IG_TEST_CAMERA_PASSWORD",
    )
    with caplog.at_level(logging.WARNING):
        result = pull_due(config, tmp_path)

    assert result == [{"status": "refused", "reason": "request_error"}]
    assert secret not in repr(result)
    assert secret not in caplog.text


def test_refuses_secrets_file_inside_repository(tmp_path: Path) -> None:
    config = _config(
        "http://127.0.0.1:9/shot.jpg",
        secrets_file=str(Path(__file__).resolve()),
    )

    result = pull_due(config, tmp_path)

    assert result == [{"status": "refused", "reason": "secrets_file_inside_repository"}]
    assert list(tmp_path.iterdir()) == []


def test_loopback_requires_explicit_opt_in(tmp_path: Path) -> None:
    with _serve(_handler(b"\xff\xd8\xffpayload")) as base_url:
        result = pull_due(
            _config(f"{base_url}/shot.jpg", allow_loopback=False),
            tmp_path,
        )

    assert result == [{"status": "refused", "reason": "loopback_blocked"}]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("address", "reason"),
    [
        ("169.254.10.2", "link_local_blocked"),
        ("169.254.169.254", "metadata_blocked"),
    ],
)
def test_refuses_link_local_and_metadata_addresses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    address: str,
    reason: str,
) -> None:
    monkeypatch.setattr(
        http_pull.socket,
        "getaddrinfo",
        lambda host, port, type: [(http_pull.socket.AF_INET, http_pull.socket.SOCK_STREAM, 6, "", (address, port))],
    )
    config = _config("http://camera.example/shot.jpg", hosts=["camera.example"])

    result = pull_due(config, tmp_path)

    assert result == [{"status": "refused", "reason": reason}]
    assert list(tmp_path.iterdir()) == []


def test_interval_can_be_forced_in_a_test_without_network_access(tmp_path: Path) -> None:
    config = _config("http://127.0.0.1:9/shot.jpg", interval_s=5)

    result = pull_due(config, tmp_path, now=100.0, last_at=97.0)

    assert result == [{"status": "refused", "reason": "interval"}]
    assert list(tmp_path.iterdir()) == []


def test_interval_floor_refuses_without_fetching(tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    with _serve(_handler(b"\xff\xd8\xffpayload", observed=observed)) as base_url:
        result = pull_due(
            _config(f"{base_url}/shot.jpg", interval_s=4),
            tmp_path,
        )

    assert result == [{"status": "refused", "reason": "interval_floor"}]
    assert observed == {}
    assert list(tmp_path.iterdir()) == []


def test_successful_attempt_is_rate_limited_per_endpoint(tmp_path: Path) -> None:
    with _serve(_handler(b"\xff\xd8\xffpayload")) as base_url:
        config = _config(f"{base_url}/shot.jpg")
        first = pull_due(config, tmp_path)
        second = pull_due(config, tmp_path)

    assert first == [{"status": "written", "basename": "shot.jpg"}]
    assert second == [{"status": "refused", "reason": "interval"}]


def test_retries_only_connection_timeouts_and_stops_at_configured_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    class Response:
        def __init__(self) -> None:
            self.headers = {"Content-Type": "image/jpeg"}
            self.body = b"\xff\xd8\xffpayload"

        def getcode(self) -> int:
            return 200

        def read(self, size: int = -1) -> bytes:
            body, self.body = self.body[:size], self.body[size:]
            return body

        def close(self) -> None:
            return

    def open_request(self, request, timeout=None):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise URLError(TimeoutError("connection timed out"))
        return Response()

    monkeypatch.setattr(
        http_pull.socket,
        "getaddrinfo",
        lambda host, port, type: [(http_pull.socket.AF_INET, http_pull.socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))],
    )
    monkeypatch.setattr("urllib.request.OpenerDirector.open", open_request)
    config = _config(
        "http://camera.example/shot.jpg",
        hosts=["camera.example"],
        max_retries=2,
    )

    result = pull_due(config, tmp_path)

    assert attempts == 3
    assert result == [{"status": "written", "basename": "shot.jpg"}]
    assert (tmp_path / "shot.jpg").read_bytes() == b"\xff\xd8\xffpayload"
    _assert_no_temporary_files(tmp_path)


def test_existing_jpeg_is_never_overwritten(tmp_path: Path) -> None:
    existing = tmp_path / "shot.jpg"
    existing.write_bytes(b"existing")
    with _serve(_handler(b"\xff\xd8\xffnew")) as base_url:
        result = pull_due(_config(f"{base_url}/shot.jpg"), tmp_path)

    assert result == [{"status": "written", "basename": "shot-1.jpg"}]
    assert existing.read_bytes() == b"existing"
    assert (tmp_path / "shot-1.jpg").read_bytes() == b"\xff\xd8\xffnew"
    _assert_no_temporary_files(tmp_path)


def test_filename_collision_suffix_cannot_expose_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "-1"
    monkeypatch.setenv("IG_TEST_CAMERA_PASSWORD", secret)
    existing = tmp_path / "shot.jpg"
    existing.write_bytes(b"existing")
    with _serve(_handler(b"\xff\xd8\xffnew")) as base_url:
        result = pull_due(
            _config(
                f"{base_url}/shot.jpg",
                password_env="IG_TEST_CAMERA_PASSWORD",
            ),
            tmp_path,
        )

    assert result == [{"status": "written", "basename": "shot-2.jpg"}]
    assert secret not in repr(result)
    assert existing.read_bytes() == b"existing"
    assert (tmp_path / "shot-2.jpg").read_bytes() == b"\xff\xd8\xffnew"
    _assert_no_temporary_files(tmp_path)


def test_unsafe_url_filename_uses_sanitized_utc_snapshot_name(tmp_path: Path) -> None:
    with _serve(_handler(b"\xff\xd8\xffpayload")) as base_url:
        result = pull_due(_config(f"{base_url}/%2e%2e%2fsecret.jpg"), tmp_path)

    assert result[0]["status"] == "written"
    basename = result[0]["basename"]
    assert basename.startswith("snapshot-") and basename.endswith(".jpg")
    assert ".." not in basename
    assert "/" not in basename and "\\" not in basename
    assert set(basename) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    _assert_no_temporary_files(tmp_path)
