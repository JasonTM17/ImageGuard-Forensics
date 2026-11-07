"""Course lock: a PNG with no EXIF reports every key tag missing."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from imageguard.core.metadata import KEY_TAGS, extract_exif


def test_extract_exif_empty_png_lists_every_key_tag_missing() -> None:
    buf = BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(buf, format="PNG")
    result = extract_exif(buf.getvalue())
    assert result["tag_count"] == 0
    assert result["present"] == {}
    assert result["missing"] == list(KEY_TAGS)
