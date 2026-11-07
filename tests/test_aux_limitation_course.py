"""Course lock: aux_limitation is silent at full scale and names the factor when reduced."""

from __future__ import annotations

from imageguard.core.preprocess import aux_limitation


def test_aux_limitation_full_scale_and_half_scale() -> None:
    assert aux_limitation(1.0) is None
    assert aux_limitation(0.5) == (
        "ảnh lớn — aux phân tích trên bản downscale ×0.50 "
        "(kernel đổi tỉ lệ thuận, mask/heatmap upscale về kích thước gốc)"
    )
