"""Analyze method presets shared by the GUI, demo samples, and tests.

Camera/AI screening must include at least one detector that is allowed to
return ``SUSPICIOUS_INDICATION`` (copy-move ``keypoint``/``block``, or
``deepfake_ml`` / ``deepfake_screen`` / ``deepfake_ensemble``). Heuristic
``deepfake`` cues and other AUX methods stay exploratory-only.

Demo deepfake PNGs use a periodic FFT grid that Block-DCT treats as
copy-move; keep that sample off ``block``.
"""

from __future__ import annotations

DEFAULT_ANALYZE_METHODS: tuple[str, ...] = (
    "keypoint",
    "block",
    "ela",
)

CAMERA_AI_METHODS: tuple[str, ...] = (
    "keypoint",
    "block",
    "ela",
    "noise_local",
    "freq",
    "splicing",
    "deepfake",
    "deepfake_ml",
    "deepfake_screen",
    "deepfake_ensemble",
)

DEEPFAKE_SCREENING_METHODS: tuple[str, ...] = (
    "deepfake_ml",
    "deepfake_screen",
    "deepfake_ensemble",
)

DEMO_DEEPFAKE_METHODS: tuple[str, ...] = (
    "ela",
    "noise_local",
    "freq",
    "deepfake",
    "deepfake_ml",
    "deepfake_screen",
    "deepfake_ensemble",
)

DEMO_SPLICING_METHODS: tuple[str, ...] = (
    "keypoint",
    "block",
    "ela",
    "noise",
    "noise_local",
    "splicing",
)

DEMO_COPYMOVE_METHODS: tuple[str, ...] = (
    "keypoint",
    "block",
    "ela",
)

ALL_ANALYZE_METHODS: tuple[str, ...] = (
    "keypoint",
    "block",
    "ela",
    "noise",
    "noise_local",
    "freq",
    "splicing",
    "deepfake",
    "deepfake_ml",
    "deepfake_screen",
    "deepfake_ensemble",
)
