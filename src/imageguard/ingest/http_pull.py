"""Pull bounded JPEG snapshots from explicitly allowlisted HTTP sources."""

from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import logging
import os
import re
import shutil
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imageguard.core.limits import MAX_FILE_BYTES

_SAFE_BASENAME = re.compile(r"[A-Za-z0-9._-]+\Z")
_METADATA_HOSTS = {"metadata.google.internal", "metadata.google.com"}
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_METADATA_IP = ipaddress.ip_address("169.254.169.254")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAST_ATTEMPTS_LOCK = threading.Lock()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _Refusal(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _AttemptFailure(Exception):
    def __init__(self, reason: str, *, retryable: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


def _refused(reason: str, logger: logging.Logger) -> dict[str, str]:
    logger.warning("HTTP snapshot refused: %s", reason)
    return {"status": "refused", "reason": reason}


def _interval_skipped(logger: logging.Logger) -> dict[str, str]:
    logger.debug("HTTP snapshot skipped: interval")
    return {"status": "refused", "reason": "interval"}


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not (0 < result < float("inf")):
        return None
    return result


def _clock_value(value: float | datetime) -> float:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.timestamp()
    return float(value)


def _state_key(http: dict[str, Any], urls: list[Any]) -> bytes:
    # Rate-limit by endpoint origin without retaining URL paths, queries, or userinfo.
    origins: set[str] = set()
    for url in urls:
        if not isinstance(url, str):
            continue
        try:
            parsed = urllib.parse.urlsplit(url)
            scheme = parsed.scheme.lower()
            host = _normalize_host(parsed.hostname or "")
            port = parsed.port
            if port is None:
                port = 443 if scheme == "https" else 80
        except ValueError:
            continue
        if host is not None:
            origins.add(f"{scheme}://{host}:{port}")
    material = "\0".join(sorted(origins))
    return hashlib.sha256(material.encode("utf-8", "surrogatepass")).digest()


def _normalize_host(value: str) -> str | None:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    value = value.rstrip(".")
    if not value:
        return None
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        try:
            hostname = value.encode("idna").decode("ascii").lower()
        except UnicodeError:
            return None
        if len(hostname) > 253 or any(
            not _HOST_LABEL.fullmatch(label) for label in hostname.split(".")
        ):
            return None
        return hostname
    return str(address)


def _allowlist(value: Any) -> set[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    allowed: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            return None
        normalized = _normalize_host(item)
        if normalized is None:
            return None
        allowed.add(normalized)
    return allowed


def _parse_url(url: Any) -> tuple[urllib.parse.SplitResult, str, int]:
    if not isinstance(url, str):
        raise _Refusal("invalid_url")
    try:
        parsed = urllib.parse.urlsplit(url)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            raise _Refusal("scheme_not_allowed")
        if parsed.username is not None or parsed.password is not None:
            raise _Refusal("userinfo_not_allowed")
        hostname = parsed.hostname
        if not hostname:
            raise _Refusal("invalid_url")
        normalized = _normalize_host(hostname)
        if normalized is None:
            raise _Refusal("invalid_url")
        port = parsed.port
        if port is None:
            port = 443 if scheme == "https" else 80
    except _Refusal:
        raise
    except (ValueError, UnicodeError):
        raise _Refusal("invalid_url") from None
    if not 1 <= port <= 65535:
        raise _Refusal("invalid_url")
    return parsed, normalized, port


def _retryable(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            TimeoutError,
            socket.timeout,
            socket.gaierror,
            socket.herror,
            ConnectionError,
            http.client.IncompleteRead,
        ),
    ):
        return True
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, BaseException) and _retryable(exc.reason)
    return False


def _network_failure(exc: BaseException) -> _AttemptFailure:
    reason = "timeout" if isinstance(exc, (TimeoutError, socket.timeout)) else "connection_error"
    if isinstance(exc, urllib.error.URLError) and isinstance(
        exc.reason, (TimeoutError, socket.timeout)
    ):
        reason = "timeout"
    return _AttemptFailure(reason, retryable=_retryable(exc))


def _validate_target(
    url: str,
    allowed_hosts: set[str],
    *,
    allow_loopback: bool,
    allow_link_local: bool,
) -> tuple[urllib.parse.SplitResult, str, int]:
    parsed, host, port = _parse_url(url)
    if host not in allowed_hosts:
        raise _Refusal("host_not_allowed")
    if host in _METADATA_HOSTS and not allow_link_local:
        raise _Refusal("metadata_blocked")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        if _retryable(exc):
            raise _network_failure(exc) from None
        raise _Refusal("dns_resolution_failed") from None
    if not addresses:
        raise _Refusal("dns_resolution_failed")

    for address_info in addresses:
        try:
            address = ipaddress.ip_address(address_info[4][0].split("%", 1)[0])
        except (IndexError, ValueError, TypeError):
            raise _Refusal("dns_resolution_failed") from None
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        if address == _METADATA_IP and not allow_link_local:
            raise _Refusal("metadata_blocked")
        if address.is_loopback and not allow_loopback:
            raise _Refusal("loopback_blocked")
        if address.is_link_local and not allow_link_local:
            raise _Refusal("link_local_blocked")
    return parsed, host, port


def _credentials(http: dict[str, Any]) -> tuple[str, str] | None:
    username_env = http.get("username_env", "")
    password_env = http.get("password_env", "")
    secrets_file = http.get("secrets_file", "")
    if any(not isinstance(value, str) for value in (username_env, password_env, secrets_file)):
        raise _Refusal("credentials_invalid")

    username = os.environ.get(username_env) if username_env else None
    password = os.environ.get(password_env) if password_env else None

    resolved_secret: Path | None = None
    if secrets_file:
        try:
            secret_path = Path(secrets_file)
            if not secret_path.is_absolute():
                raise _Refusal("secrets_path_must_be_absolute")
            lexical_path = Path(os.path.abspath(secret_path))
            try:
                lexical_path.relative_to(_REPO_ROOT)
            except ValueError:
                pass
            else:
                raise _Refusal("secrets_file_inside_repository")
            resolved_secret = secret_path.resolve(strict=True)
        except _Refusal:
            raise
        except (OSError, RuntimeError, ValueError):
            raise _Refusal("secrets_file_unavailable") from None
        try:
            resolved_secret.relative_to(_REPO_ROOT)
        except ValueError:
            pass
        else:
            raise _Refusal("secrets_file_inside_repository")
        if not resolved_secret.is_file():
            raise _Refusal("secrets_file_unavailable")

    if password is None and resolved_secret is not None:
        try:
            password = resolved_secret.read_text(encoding="utf-8").rstrip("\r\n")
        except (OSError, UnicodeError):
            raise _Refusal("secrets_file_unavailable") from None

    if username_env and username is None:
        raise _Refusal("credentials_unavailable")
    if (password_env or secrets_file) and not password:
        raise _Refusal("credentials_unavailable")
    if username is not None and password is None:
        raise _Refusal("credentials_incomplete")
    if password is None:
        return None
    return username or "", password


def _authorization(credentials: tuple[str, str] | None) -> str | None:
    if credentials is None:
        return None
    token = base64.b64encode(f"{credentials[0]}:{credentials[1]}".encode())
    return f"Basic {token.decode('ascii')}"


def _origin(parsed: urllib.parse.SplitResult, port: int) -> tuple[str, str, int]:
    return parsed.scheme.lower(), _normalize_host(parsed.hostname or "") or "", port


def _contains_secret(value: str, secret_values: tuple[str, ...]) -> bool:
    return any(secret and secret in value for secret in secret_values)


def _filename_from_url(url: str, secret_values: tuple[str, ...] = ()) -> str:
    try:
        path = urllib.parse.urlsplit(url).path
        raw = urllib.parse.unquote(path.rsplit("/", 1)[-1], errors="replace")
    except (ValueError, UnicodeError):
        raw = ""
    if (
        not raw
        or _contains_secret(raw, secret_values)
        or len(raw) > 180
        or not _SAFE_BASENAME.fullmatch(raw)
        or ".." in raw
    ):
        raw = ""
    if raw.lower().endswith(".jpeg"):
        stem = raw[:-5]
    elif raw.lower().endswith(".jpg"):
        stem = raw[:-4]
    else:
        stem = raw
    filename = f"{stem}.jpg"
    reserved_windows_name = stem.split(".", 1)[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    if (
        not stem
        or len(filename) > 184
        or not _SAFE_BASENAME.fullmatch(stem)
        or ".." in filename
        or reserved_windows_name
    ):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        stem = f"snapshot-{stamp}"
    filename = f"{stem}.jpg"
    if _contains_secret(filename, secret_values):
        raise _Refusal("filename_contains_secret")
    return filename


def _name_at(base_name: str, index: int) -> str:
    if index == 0:
        return base_name
    return f"{base_name[:-4]}-{index}.jpg"


def _open_temp(
    dest_dir: Path,
    base_name: str,
    secret_values: tuple[str, ...],
) -> tuple[str, Path, Path, Any]:
    for index in range(10000):
        basename = _name_at(base_name, index)
        if _contains_secret(basename, secret_values):
            continue
        final_path = dest_dir / basename
        if final_path.exists():
            continue
        temp_path = dest_dir / f".{basename}.tmp"
        try:
            descriptor = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            continue
        return basename, final_path, temp_path, os.fdopen(descriptor, "wb")
    raise _Refusal("filename_unavailable")


def _move_temp(temp_path: Path, next_temp: Path) -> None:
    descriptor = os.open(next_temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as target, temp_path.open("rb") as source:
            shutil.copyfileobj(source, target, length=64 * 1024)
            target.flush()
            os.fsync(target.fileno())
        temp_path.unlink()
    except BaseException:
        next_temp.unlink(missing_ok=True)
        raise


def _publish_temp(
    temp_path: Path,
    base_name: str,
    secret_values: tuple[str, ...],
) -> str:
    dest_dir = temp_path.parent
    current_temp = temp_path
    try:
        for index in range(10000):
            basename = _name_at(base_name, index)
            if _contains_secret(basename, secret_values):
                continue
            final_path = dest_dir / basename
            if final_path.exists():
                continue
            next_temp = dest_dir / f".{basename}.tmp"
            if next_temp != current_temp:
                if next_temp.exists():
                    continue
                try:
                    _move_temp(current_temp, next_temp)
                except FileExistsError:
                    continue
                current_temp = next_temp
            try:
                descriptor = os.open(
                    final_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                continue
            try:
                os.close(descriptor)
                os.replace(current_temp, final_path)
            except BaseException:
                final_path.unlink(missing_ok=True)
                raise
            return basename
        raise _Refusal("filename_unavailable")
    except BaseException:
        current_temp.unlink(missing_ok=True)
        raise


def _stream_jpeg(
    response: Any,
    dest_dir: Path,
    base_name: str,
    max_bytes: int,
    secret_values: tuple[str, ...],
) -> str:
    content_type = response.headers.get("Content-Type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "image/jpeg":
        raise _Refusal("content_type_not_jpeg")

    dest_dir.mkdir(parents=True, exist_ok=True)
    _basename, _final_path, temp_path, output = _open_temp(
        dest_dir,
        base_name,
        secret_values,
    )
    prefix = bytearray()
    total = 0
    try:
        with output:
            while True:
                read_size = min(64 * 1024, max_bytes - total + 1)
                try:
                    chunk = response.read(read_size)
                except (OSError, http.client.HTTPException) as exc:
                    if _retryable(exc):
                        raise _network_failure(exc) from None
                    raise _AttemptFailure("read_error", retryable=False) from None
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise _Refusal("too_large")
                if len(prefix) < 3:
                    prefix.extend(chunk[: 3 - len(prefix)])
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if bytes(prefix) != b"\xff\xd8\xff":
            raise _Refusal("invalid_jpeg")
        return _publish_temp(temp_path, base_name, secret_values)
    finally:
        temp_path.unlink(missing_ok=True)


def _one_attempt(
    url: str,
    dest_dir: Path,
    *,
    allowed_hosts: set[str],
    allow_loopback: bool,
    allow_link_local: bool,
    timeout_s: float,
    authorization: str | None,
    initial_origin: tuple[str, str, int],
    max_bytes: int,
    secret_values: tuple[str, ...],
) -> str:
    current_url = url
    redirected = False
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    for redirects in range(4):
        try:
            parsed, _host, port = _validate_target(
                current_url,
                allowed_hosts,
                allow_loopback=allow_loopback,
                allow_link_local=allow_link_local,
            )
        except _Refusal as refusal:
            if redirected:
                raise _Refusal(f"redirect_{refusal.reason}") from None
            raise
        headers: dict[str, str] = {}
        if authorization is not None and _origin(parsed, port) == initial_origin:
            headers["Authorization"] = authorization
        try:
            response = opener.open(urllib.request.Request(current_url, headers=headers), timeout=timeout_s)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                location = exc.headers.get("Location") if exc.headers is not None else None
                exc.close()
                if not location:
                    raise _Refusal("redirect_missing_location")
                if redirects >= 3:
                    raise _Refusal("redirect_limit")
                try:
                    current_url = urllib.parse.urljoin(current_url, location)
                except ValueError:
                    raise _Refusal("invalid_redirect") from None
                redirected = True
                continue
            exc.close()
            raise _Refusal("http_status")
        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
            if _retryable(exc):
                raise _network_failure(exc) from None
            raise _AttemptFailure("request_error", retryable=False) from None

        try:
            code = response.getcode()
            if code is not None and 300 <= code < 400:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise _Refusal("redirect_missing_location")
                if redirects >= 3:
                    raise _Refusal("redirect_limit")
                try:
                    current_url = urllib.parse.urljoin(current_url, location)
                except ValueError:
                    raise _Refusal("invalid_redirect") from None
                redirected = True
                continue
            return _stream_jpeg(
                response,
                dest_dir,
                _filename_from_url(current_url, secret_values),
                max_bytes,
                secret_values,
            )
        finally:
            response.close()
    raise _Refusal("redirect_limit")


def pull_due(
    http: dict,
    dest_dir: Path,
    *,
    max_bytes: int | None = None,
    logger: logging.Logger | None = None,
    now: float | datetime | None = None,
    last_at: float | datetime | None = None,
) -> list[dict]:
    """Pull the configured HTTP JPEG snapshots without analyzing their pixels."""
    if not isinstance(http, dict):
        return []
    if http.get("enabled") is not True:
        return []

    raw_urls = http.get("urls", [])
    if isinstance(raw_urls, (list, tuple)):
        urls = list(raw_urls)
    elif raw_urls is None:
        urls = []
    else:
        urls = [raw_urls]
    if not urls:
        return []

    active_logger = logger or logging.getLogger(__name__)
    interval = _as_number(http.get("interval_s"))
    timeout = _as_number(http.get("timeout_s"))
    retries = http.get("max_retries")
    byte_limit = MAX_FILE_BYTES if max_bytes is None else max_bytes
    if interval is None:
        return [_refused("invalid_interval", active_logger) for _ in urls]
    if interval < 5:
        return [_refused("interval_floor", active_logger) for _ in urls]
    if timeout is None or timeout > 60:
        return [_refused("invalid_timeout", active_logger) for _ in urls]
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 3:
        return [_refused("invalid_retries", active_logger) for _ in urls]
    if isinstance(byte_limit, bool) or not isinstance(byte_limit, int) or byte_limit <= 0:
        return [_refused("invalid_max_bytes", active_logger) for _ in urls]

    allowed_hosts = _allowlist(http.get("hosts"))
    if allowed_hosts is None:
        return [_refused("invalid_allowlist", active_logger) for _ in urls]
    try:
        credentials = _credentials(http)
        authorization = _authorization(credentials)
    except _Refusal as refusal:
        return [_refused(refusal.reason, active_logger) for _ in urls]
    except UnicodeError:
        return [_refused("credentials_invalid", active_logger) for _ in urls]
    secret_values = tuple(value for value in (credentials or ()) if value)

    has_clock_override = now is not None or last_at is not None
    if now is not None:
        current_time = _clock_value(now)
    elif isinstance(last_at, datetime):
        current_time = datetime.now(UTC).timestamp()
    else:
        current_time = time.monotonic()
    key = _state_key(http, urls)
    with _LAST_ATTEMPTS_LOCK:
        attempt_state = getattr(pull_due, "_last_attempts", None)
        if attempt_state is None:
            attempt_state = {}
            pull_due._last_attempts = attempt_state
        if last_at is not None:
            prior_time = _clock_value(last_at)
        elif has_clock_override:
            prior_time = None
        else:
            prior_time = attempt_state.get(key)
        if prior_time is not None and current_time - prior_time < interval:
            return [_interval_skipped(active_logger) for _ in urls]
        if not has_clock_override:
            if key not in attempt_state and len(attempt_state) >= 128:
                attempt_state.pop(next(iter(attempt_state)))
            attempt_state[key] = current_time

    allow_loopback = http.get("allow_loopback") is True
    allow_link_local = http.get("allow_link_local") is True
    results: list[dict[str, str]] = []
    for url in urls:
        try:
            parsed, _normalized_host, port = _parse_url(url)
            initial_origin = _origin(parsed, port)
        except _Refusal as refusal:
            results.append(_refused(refusal.reason, active_logger))
            continue
        initial_url = url
        for attempt in range(retries + 1):
            try:
                basename = _one_attempt(
                    initial_url,
                    Path(dest_dir),
                    allowed_hosts=allowed_hosts,
                    allow_loopback=allow_loopback,
                    allow_link_local=allow_link_local,
                    timeout_s=timeout,
                    authorization=authorization,
                    initial_origin=initial_origin,
                    max_bytes=byte_limit,
                    secret_values=secret_values,
                )
            except _Refusal as refusal:
                results.append(_refused(refusal.reason, active_logger))
                break
            except _AttemptFailure as failure:
                if failure.retryable and attempt < retries:
                    continue
                results.append(_refused(failure.reason, active_logger))
                break
            except OSError:
                results.append(_refused("write_error", active_logger))
                break
            except (ValueError, TypeError, urllib.error.URLError, http.client.HTTPException):
                results.append(_refused("request_error", active_logger))
                break
            else:
                active_logger.info("HTTP snapshot written")
                results.append({"status": "written", "basename": basename})
                break
    return results
