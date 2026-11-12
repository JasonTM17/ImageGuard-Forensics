# Course failure cases (limitations)

This note lists three **limitations** of the pipeline as taught in the project:
they describe what the software refuses to assert or cannot do, not authentication
of origin and not a finding of malice. Outcomes are methodological bounds.

## 1. ELA on PNG is `NOT_APPLICABLE`

Error Level Analysis compares re-encode residuals and assumes a JPEG
quantization history. In this pipeline the stage is gated to JPEG: for a PNG it
returns `NOT_APPLICABLE` and records that limitation.

**Locked by:** `tests/test_ela.py` — `test_png_not_applicable` asserts
`Verdict.NOT_APPLICABLE` and a limitation stating that ELA analyzes JPEG
compression artifacts.

## 2. Flat / low-texture images and copy-move (keypoint path)

Copy-move keypoint analysis needs stable local structure to detect and match
features. A uniform or nearly flat region does not yield enough keypoints to
support a copy-move *claim* through this method; the run ends `INCONCLUSIVE` with
an explicit texture limitation.

**Locked by:** `tests/test_keypoint.py` — `test_low_texture_inconclusive`.

## 3. Splicing cues on non-JPEG stay `INCONCLUSIVE` without a cue mask

The splicing experiment cross-checks noise outliers with ELA hot regions. On a
non-JPEG document, ELA does not produce a comparable mask, so the cue pair is
degraded: the result stays `INCONCLUSIVE`, there is no corroboration mask, and
`evidence["cue_unavailable"]` is set.

**Related behavior:** `tests/test_auxiliary_l2.py` — `test_splicing_nonjpeg_degrades`.

**Explicit no-mask lock:** `tests/test_failure_cases.py` —
`test_splicing_nonjpeg_inconclusive_no_cue_mask` (in-memory PNG,
default threshold set).

See `src/imageguard/forensics/auxiliary/splicing.py` for the branch where
`noise.mask` or `ela.mask` is missing.
