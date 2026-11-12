# Synthetic splicing course probe

This probe creates three seeded (seed `20261106`), 256 × 256 JPEG composites
re-encoded at quality 88. Each composite starts with a generated image, receives
a rectangular region from a second generated image, and carries a matching
ground-truth rectangle. The existing `SplicingCues` analyzer runs on each
decoded JPEG.

The summary reports how many samples received a score, how many were excluded,
the count of `SUSPICIOUS_INDICATION` verdicts, missing cue masks, cue absence,
and the mean IoU of available cue masks against the paste rectangles. IoU is
computed only for masks that the analyzer returned. If a required cue is
unavailable, the sample records that condition and its reason.

This is a synthetic course probe, not a camera-photo benchmark. A high IoU is
not a forgery verdict. The analyzer's thresholds and behavior are unchanged;
results describe only this fixed synthetic setup. The accompanying test
(`tests/test_splicing_study.py`) asserts that this set yields zero
`SUSPICIOUS_INDICATION` verdicts and complete sample accounting.

Run it from the repository root with:

```powershell
$env:PYTHONPATH = 'src'
python -m pytest tests/test_splicing_study.py -q
```
