# Classical enhancement study

Evaluation only. Detectors ran on derived copies. The source files were not modified, and no result was written into the app database or into fusion.

PSNR/SSIM gain never establishes a detection gain; negative results are successful

## Limits

- Synthetic synthforge images only. This page is not a result about a camera, a room, or a recovered trace.
- N = 4 manipulated samples and 4 matched originals, 256×256, tier `easy`, generation seed `20261106`.
- Thresholds were the frozen snapshot below. They were not tuned on this set.
- Richardson–Lucy and Wiener are `NOT_RUN` on AWGN, JPEG, and downsample (no generating blur PSF): 48 arms.
- Downsample PSNR and SSIM are null because the arm grid is not the sharp grid. No second resize was applied to force a score.
- Alignment failures: `0`.
- The run recorded git `3b0305e` with a clean tree. The merged study commit is `cb83c38`; the later write-guard commit `65dd294` does not change these measurements.

## How it was run

Outside the repository, in a temp directory:

```python
generate_set(4, 20261106, set_dir, size=(256, 256), tier="easy")
run_study(set_dir, out_dir, seed=20261106)
```

`run_study` defaults: degradations `gaussian,disk,motion,awgn,jpeg,downsample`; restorers `unsharp,clahe,nlm,richardson_lucy,wiener`; methods `keypoint,block,ela,noise`.

## Provenance

- N: `4` sample pairs
- seed: `20261106`
- config_hash: `df80dcb9a6d32070`
- git_sha at run: `3b0305e`
- working_tree_clean: `True`
- dataset_version: `f0e28012ee96`

Operator parameters recorded in the JSON:

| operator | parameters |
|---|---|
| gaussian | ksize 7, sigma 1.4 |
| disk | radius 4 |
| motion | length 9, angle 32 degrees |
| awgn | sigma 12 |
| jpeg | quality 55 |
| downsample | factor 2 |
| unsharp | ksize 5, sigma 1.0, amount 0.7 |
| clahe | clip 2.0, tiles 8×8 |
| nlm | h 3, hColor 3, template 7, search 21 |
| richardson_lucy | 10 iterations; oracle kernel and a wrong kernel |
| wiener | nsr 0.01; oracle kernel and a wrong kernel |

## Matched originals

Across 592 method cells on matched originals, `suspicious_indication` occurred 0 times. Non-empty masks still appear, especially from keypoint and noise, and those masks are not a positive detection.

## Manipulated samples

Mean IoU is over the 4 samples. A missing prediction mask scores 0 and stays in that mean. `sus` is the count of `suspicious_indication` out of 4. PSNR and SSIM are means against the sharp manipulated image. Null distortion is shown as —.

| arm | keypoint IoU | keypoint sus | block IoU | block sus | PSNR | SSIM |
|---|---|---|---|---|---|---|
| sharp | 0.3108 | 2/4 | 0.2413 | 2/4 | — | — |
| gaussian | 0.2009 | 1/4 | 0.3726 | 4/4 | 24.230 | 0.688 |
| gaussian / unsharp | 0.2418 | 1/4 | 0.3903 | 4/4 | 24.832 | 0.725 |
| gaussian / clahe | 0.2421 | 2/4 | 0.0000 | 0/4 | 20.300 | 0.655 |
| gaussian / nlm | 0.1200 | 1/4 | 0.2792 | 3/4 | 23.647 | 0.600 |
| gaussian / richardson_lucy / oracle | 0.2768 | 3/4 | 0.2958 | 3/4 | 25.968 | 0.783 |
| gaussian / richardson_lucy / wrong | 0.3162 | 3/4 | 0.1768 | 2/4 | 24.916 | 0.736 |
| gaussian / wiener / oracle | 0.3201 | 3/4 | 0.2989 | 3/4 | 24.839 | 0.779 |
| gaussian / wiener / wrong | 0.3395 | 4/4 | 0.0590 | 1/4 | 21.122 | 0.599 |
| disk | 0.0295 | 0/4 | 0.3668 | 4/4 | 21.582 | 0.479 |
| disk / unsharp | 0.0331 | 0/4 | 0.3604 | 4/4 | 21.617 | 0.477 |
| disk / clahe | 0.0422 | 0/4 | 0.0000 | 0/4 | 18.767 | 0.432 |
| disk / nlm | 0.0041 | 0/4 | 0.2206 | 3/4 | 21.478 | 0.472 |
| disk / richardson_lucy / oracle | 0.1480 | 1/4 | 0.1268 | 2/4 | 22.924 | 0.572 |
| disk / richardson_lucy / wrong | 0.1694 | 1/4 | 0.0767 | 1/4 | 22.087 | 0.513 |
| disk / wiener / oracle | 0.2082 | 3/4 | 0.0000 | 0/4 | 21.112 | 0.603 |
| disk / wiener / wrong | 0.1810 | 2/4 | 0.0000 | 0/4 | 19.042 | 0.391 |
| motion | 0.1897 | 2/4 | 0.3893 | 4/4 | 22.599 | 0.599 |
| motion / unsharp | 0.2142 | 2/4 | 0.3685 | 4/4 | 22.491 | 0.602 |
| motion / clahe | 0.1765 | 3/4 | 0.0000 | 0/4 | 19.123 | 0.529 |
| motion / nlm | 0.0997 | 0/4 | 0.2475 | 3/4 | 22.467 | 0.556 |
| motion / richardson_lucy / oracle | 0.2644 | 3/4 | 0.1587 | 2/4 | 25.514 | 0.765 |
| motion / richardson_lucy / wrong | 0.2333 | 3/4 | 0.0911 | 1/4 | 22.848 | 0.593 |
| motion / wiener / oracle | 0.2155 | 2/4 | 0.0000 | 0/4 | 22.331 | 0.752 |
| motion / wiener / wrong | 0.1763 | 2/4 | 0.0000 | 0/4 | 18.825 | 0.359 |
| awgn | 0.1261 | 0/4 | 0.0000 | 0/4 | 26.572 | 0.699 |
| awgn / unsharp | 0.1716 | 0/4 | 0.0000 | 0/4 | 21.944 | 0.551 |
| awgn / clahe | 0.1694 | 0/4 | 0.0000 | 0/4 | 19.085 | 0.523 |
| awgn / nlm | 0.0760 | 0/4 | 0.0000 | 0/4 | 27.064 | 0.725 |
| jpeg | 0.2188 | 1/4 | 0.0000 | 0/4 | 24.537 | 0.686 |
| jpeg / unsharp | 0.1996 | 1/4 | 0.0000 | 0/4 | 23.524 | 0.658 |
| jpeg / clahe | 0.2189 | 1/4 | 0.0000 | 0/4 | 19.683 | 0.572 |
| jpeg / nlm | 0.1543 | 1/4 | 0.0000 | 0/4 | 24.486 | 0.659 |
| downsample | 0.1155 | 0/4 | 0.0000 | 0/4 | — | — |
| downsample / unsharp | 0.0000 | 0/4 | 0.0000 | 0/4 | — | — |
| downsample / clahe | 0.0722 | 0/4 | 0.0000 | 0/4 | — | — |
| downsample / nlm | 0.2107 | 0/4 | 0.0000 | 0/4 | — | — |

A higher PSNR on a row is not evidence that detection improved. On this set, Gaussian degradation left block at IoU 0.3726 with 4/4 suspicious, while the oracle Lucy arm rose to PSNR 25.968 and block IoU fell to 0.2958 with 3/4 suspicious. The wrong Wiener kernel on that same blur fell to PSNR 21.122 and block IoU 0.0590 with 1/4 suspicious.
