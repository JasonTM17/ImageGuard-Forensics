"""Coverage gaps: core.errors taxonomy helpers."""

from __future__ import annotations

from imageguard.core.errors import ErrorCode, ErrorResult, ImageGuardError


def test_error_result_str_includes_code_and_message() -> None:
    err = ErrorResult(ErrorCode.PATH_INVALID, "missing file", detail={"path": "x"})
    assert str(err) == "[PATH_INVALID] missing file"
    assert err.detail["path"] == "x"


def test_image_guard_error_carries_code_and_detail() -> None:
    exc = ImageGuardError(
        ErrorCode.DECODE_FAILED,
        "corrupt header",
        offset=12,
    )
    assert exc.code is ErrorCode.DECODE_FAILED
    assert exc.detail == {"offset": 12}
    assert str(exc) == "corrupt header"
