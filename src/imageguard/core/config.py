"""Load configs/thresholds.yaml → ThresholdSet có nguồn cho từng giá trị.

Mỗi threshold trong yaml: {value, unit?, source, range?}. `source` giải thích
con số đến từ đâu (literature / default / calibrated:<experiment_id>) — bắt
buộc để trace về mặt khoa học khi viết báo cáo.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from imageguard.core.paths import config_path

DEFAULT_CONFIG = config_path()


@dataclass(frozen=True, slots=True)
class Threshold:
    value: Any
    source: str
    unit: str | None = None
    range: tuple[float, float] | None = None

    def validate(self) -> None:
        """Clamp-check: nếu có range mà value ngoài miền → cảnh báo sớm."""
        if self.range is not None and isinstance(self.value, (int, float)):
            lo, hi = self.range
            if not lo <= self.value <= hi:
                raise ValueError(
                    f"threshold {self.value} ngoài range [{lo}, {hi}] (source: {self.source})"
                )


@dataclass(slots=True)
class ThresholdSet:
    raw: dict[str, dict[str, Any]]
    config_hash: str = ""

    def get(self, section: str, key: str) -> Threshold:
        try:
            node = self.raw[section][key]
        except KeyError as e:
            raise KeyError(f"thresholds.yaml thiếu '{section}.{key}'") from e
        if not isinstance(node, dict) or "value" not in node:
            raise ValueError(f"threshold '{section}.{key}' phải là dict có 'value'")
        rng = node.get("range")
        th = Threshold(
            value=node["value"],
            source=node.get("source", "unknown"),
            unit=node.get("unit"),
            range=tuple(rng) if rng else None,
        )
        th.validate()
        return th

    def value(self, section: str, key: str) -> Any:
        return self.get(section, key).value

    def limit(self, key: str, fallback: int) -> int:
        """Lấy override trong section `limits`; thiếu thì dùng hard const."""
        try:
            return int(self.value("limits", key))
        except KeyError:
            return fallback

    def snapshot_hash(self) -> str:
        """sha256[:16] của toàn bộ cấu hình — ghi vào report để tái hiện."""
        return self.config_hash


def load_thresholds(path: Path | None = None) -> ThresholdSet:
    p = path or DEFAULT_CONFIG
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004 - ValueError nhất quán với ThresholdSet.get cho node không phải mapping
            f"{p} không phải mapping yaml"
        )
    data.pop("version", None)
    canonical = json.dumps(data, sort_keys=True, default=str).encode()
    return ThresholdSet(raw=data, config_hash=hashlib.sha256(canonical).hexdigest()[:16])
