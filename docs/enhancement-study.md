# Classical enhancement study

Evaluation only. Detectors ran on derived copies. The source files were not modified, and no result was written into the app database or into fusion.

PSNR/SSIM gain never establishes a detection gain; negative results are successful

## Limits

- Synthetic synthforge images only. This page is not a result about a camera, a room, or a recovered trace.
- N = 24 manipulated samples and 24 matched originals, 256×256, tier `easy`, generation seed `20261106`. One fixed tier, not a mix.
- Thresholds were the frozen snapshot below. They were not tuned on this set.
- Richardson–Lucy and Wiener are `NOT_RUN` on AWGN, JPEG, and downsample (no generating blur PSF): 288 arms (24 samples × 2 roles × 3 degradations × 2 restorers).
- Downsample PSNR and SSIM are null because the arm grid is not the sharp grid. Those nulls are omitted from the bootstrap (`omitted` = 24). No second resize was applied to force a score.
- A missing prediction mask is stored as IoU 0 and stays in the mean. An uncomputed IoU (alignment failure, or a method that was not run) is omitted, not imputed. Alignment failures: `0`.
- The earlier N = 4 table in this file is replaced by this run. It is not a second independent confirmation.

## How it was run

Outside the repository, in a temp directory, on commit `2be3bf1` with a clean worktree. Wall time was 1025.6 seconds (generate 1.2 s, study 1024.4 s).

```python
generate_set(24, 20261106, set_dir, size=(256, 256), tier="easy")
run_study(set_dir, out_dir, seed=20261106)
```

`run_study` defaults: degradations `gaussian,disk,motion,awgn,jpeg,downsample`; restorers `unsharp,clahe,nlm,richardson_lucy,wiener`; methods `keypoint,block,ela,noise`.

## Bootstrap

Percentile bootstrap, not BCa and not a p-value. One `numpy.random.default_rng(seed)` draws every resample for the report, with `n_boot` = 2000. Non-finite values are dropped. When fewer than 2 finite values remain, the bounds are null. Otherwise each resample is drawn with replacement via `Generator.integers`, and the interval is the 2.5 and 97.5 percentiles of the resample means. Paired intervals are enhanced minus degraded on the same sample and the same degradation. A confidence interval which overlaps zero does not establish a change.

## Provenance

- N: `24` sample pairs
- seed: `20261106`
- config_hash: `df80dcb9a6d32070`
- git_sha at run: `2be3bf1`
- working_tree_clean: `True`
- dataset_version: `37eb9971ab03` (hash of this temp synthforge set, not of the N = 4 set)
- runtime_seconds: `1025.6`

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
| richardson_lucy | 10 iterations; oracle kernel and a wrong kernel on blur, otherwise NOT_RUN |
| wiener | nsr 0.01; oracle kernel and a wrong kernel on blur, otherwise NOT_RUN |

## Matched originals

Across 3456 method cells on matched originals (24 images × 36 comparison arms × 4 methods), `suspicious_indication` occurred 3 times. All three are keypoint on the oracle Wiener arm of Gaussian blur, samples `easy_0009`, `easy_0011`, and `easy_0015`. Those are false cues on unmodified originals. Non-empty masks still appear and are not a positive detection.

## Manipulated samples

Each cell is the sample mean and the percentile 95% interval. `sus` is the suspicious count out of 24, with the rate interval beside it. Sharp has no PSNR or SSIM against itself. `NOT_RUN` arms are listed and have no interval.

| arm | keypoint IoU | keypoint sus | block IoU | block sus | PSNR | SSIM |
|---|---|---|---|---|---|---|
| sharp | 0.2708 [0.2191, 0.3201] | 11/24 0.458 [0.250, 0.667] | 0.1178 [0.0403, 0.1989] | 6/24 0.250 [0.083, 0.417] | — | — |
| gaussian | 0.1715 [0.1161, 0.2256] | 7/24 0.292 [0.125, 0.458] | 0.2705 [0.1921, 0.3438] | 16/24 0.667 [0.500, 0.833] | 24.474 [23.978, 24.962] | 0.693 [0.674, 0.711] |
| gaussian / unsharp | 0.2012 [0.1511, 0.2534] | 10/24 0.417 [0.250, 0.583] | 0.2620 [0.1826, 0.3362] | 16/24 0.667 [0.458, 0.833] | 25.081 [24.542, 25.593] | 0.730 [0.712, 0.748] |
| gaussian / clahe | 0.1781 [0.1289, 0.2280] | 8/24 0.333 [0.167, 0.542] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 20.384 [20.108, 20.652] | 0.659 [0.642, 0.675] |
| gaussian / nlm | 0.0962 [0.0568, 0.1354] | 3/24 0.125 [0.000, 0.251] | 0.1955 [0.1235, 0.2671] | 14/24 0.583 [0.375, 0.792] | 23.858 [23.446, 24.283] | 0.603 [0.592, 0.614] |
| gaussian / richardson_lucy / oracle | 0.2694 [0.2201, 0.3158] | 16/24 0.667 [0.500, 0.833] | 0.1619 [0.0887, 0.2331] | 11/24 0.458 [0.250, 0.667] | 26.232 [25.654, 26.804] | 0.788 [0.769, 0.806] |
| gaussian / richardson_lucy / wrong | 0.2737 [0.2184, 0.3217] | 17/24 0.708 [0.542, 0.875] | 0.0870 [0.0297, 0.1498] | 6/24 0.250 [0.083, 0.417] | 25.189 [24.632, 25.715] | 0.742 [0.722, 0.762] |
| gaussian / wiener / oracle | 0.2929 [0.2427, 0.3345] | 19/24 0.792 [0.625, 0.917] | 0.1380 [0.0703, 0.2192] | 9/24 0.375 [0.208, 0.583] | 24.981 [24.558, 25.368] | 0.783 [0.767, 0.800] |
| gaussian / wiener / wrong | 0.2889 [0.2362, 0.3335] | 20/24 0.833 [0.667, 0.958] | 0.0348 [0.0000, 0.0750] | 3/24 0.125 [0.000, 0.292] | 21.206 [20.907, 21.496] | 0.607 [0.587, 0.627] |
| disk | 0.0480 [0.0232, 0.0770] | 1/24 0.042 [0.000, 0.125] | 0.2485 [0.1772, 0.3241] | 16/24 0.667 [0.500, 0.833] | 21.802 [21.364, 22.213] | 0.483 [0.469, 0.498] |
| disk / unsharp | 0.0559 [0.0289, 0.0850] | 0/24 0.000 [0.000, 0.000] | 0.2400 [0.1588, 0.3190] | 15/24 0.625 [0.417, 0.792] | 21.839 [21.433, 22.266] | 0.481 [0.466, 0.496] |
| disk / clahe | 0.0711 [0.0394, 0.1083] | 0/24 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 18.921 [18.627, 19.205] | 0.435 [0.423, 0.447] |
| disk / nlm | 0.0398 [0.0156, 0.0657] | 0/24 0.000 [0.000, 0.000] | 0.1915 [0.1261, 0.2538] | 15/24 0.625 [0.417, 0.792] | 21.691 [21.314, 22.082] | 0.476 [0.463, 0.488] |
| disk / richardson_lucy / oracle | 0.1366 [0.0926, 0.1805] | 3/24 0.125 [0.000, 0.292] | 0.1201 [0.0684, 0.1721] | 12/24 0.500 [0.292, 0.708] | 23.164 [22.705, 23.609] | 0.577 [0.560, 0.592] |
| disk / richardson_lucy / wrong | 0.1253 [0.0786, 0.1733] | 3/24 0.125 [0.000, 0.250] | 0.0371 [0.0093, 0.0717] | 5/24 0.208 [0.042, 0.375] | 22.319 [21.868, 22.729] | 0.517 [0.500, 0.533] |
| disk / wiener / oracle | 0.1747 [0.1235, 0.2248] | 10/24 0.417 [0.249, 0.625] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 21.147 [20.903, 21.400] | 0.606 [0.593, 0.620] |
| disk / wiener / wrong | 0.1398 [0.0933, 0.1861] | 6/24 0.250 [0.083, 0.458] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 19.098 [18.847, 19.359] | 0.394 [0.379, 0.408] |
| motion | 0.1682 [0.1168, 0.2175] | 9/24 0.375 [0.167, 0.583] | 0.2656 [0.1831, 0.3433] | 16/24 0.667 [0.500, 0.833] | 22.860 [22.424, 23.286] | 0.604 [0.590, 0.618] |
| motion / unsharp | 0.1812 [0.1293, 0.2335] | 10/24 0.417 [0.208, 0.625] | 0.2437 [0.1747, 0.3143] | 16/24 0.667 [0.458, 0.833] | 22.760 [22.307, 23.226] | 0.608 [0.594, 0.621] |
| motion / clahe | 0.1715 [0.1217, 0.2201] | 9/24 0.375 [0.167, 0.583] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 19.259 [19.000, 19.511] | 0.533 [0.520, 0.545] |
| motion / nlm | 0.1182 [0.0738, 0.1625] | 4/24 0.167 [0.042, 0.333] | 0.1743 [0.1033, 0.2484] | 12/24 0.500 [0.292, 0.708] | 22.711 [22.284, 23.118] | 0.561 [0.548, 0.573] |
| motion / richardson_lucy / oracle | 0.1997 [0.1454, 0.2567] | 11/24 0.458 [0.250, 0.667] | 0.1066 [0.0482, 0.1675] | 9/24 0.375 [0.167, 0.583] | 25.803 [25.361, 26.255] | 0.769 [0.759, 0.779] |
| motion / richardson_lucy / wrong | 0.2016 [0.1383, 0.2615] | 12/24 0.500 [0.292, 0.708] | 0.0520 [0.0151, 0.0951] | 5/24 0.208 [0.042, 0.375] | 23.096 [22.638, 23.546] | 0.600 [0.581, 0.619] |
| motion / wiener / oracle | 0.2064 [0.1498, 0.2588] | 12/24 0.500 [0.292, 0.708] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 22.320 [22.113, 22.547] | 0.754 [0.748, 0.760] |
| motion / wiener / wrong | 0.1421 [0.0956, 0.1880] | 6/24 0.250 [0.083, 0.417] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 18.916 [18.585, 19.242] | 0.365 [0.346, 0.384] |
| awgn | 0.1280 [0.0868, 0.1683] | 2/24 0.083 [0.000, 0.208] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 26.574 [26.567, 26.582] | 0.698 [0.693, 0.703] |
| awgn / unsharp | 0.1184 [0.0761, 0.1652] | 0/24 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 21.967 [21.917, 22.014] | 0.550 [0.543, 0.556] |
| awgn / clahe | 0.1094 [0.0724, 0.1465] | 0/24 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 18.989 [18.876, 19.099] | 0.524 [0.519, 0.528] |
| awgn / nlm | 0.1340 [0.0854, 0.1822] | 3/24 0.125 [0.000, 0.292] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 27.074 [27.007, 27.147] | 0.726 [0.723, 0.729] |
| awgn / richardson_lucy | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| awgn / wiener | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| jpeg | 0.2308 [0.1776, 0.2823] | 7/24 0.292 [0.125, 0.500] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 24.869 [24.102, 25.566] | 0.696 [0.664, 0.727] |
| jpeg / unsharp | 0.1816 [0.1316, 0.2292] | 6/24 0.250 [0.083, 0.458] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 23.900 [23.146, 24.625] | 0.670 [0.636, 0.704] |
| jpeg / clahe | 0.1912 [0.1395, 0.2428] | 7/24 0.292 [0.125, 0.500] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 19.731 [19.373, 20.084] | 0.582 [0.555, 0.610] |
| jpeg / nlm | 0.1918 [0.1466, 0.2359] | 7/24 0.292 [0.125, 0.458] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | 24.814 [24.091, 25.523] | 0.669 [0.642, 0.695] |
| jpeg / richardson_lucy | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| jpeg / wiener | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| downsample | 0.1046 [0.0616, 0.1507] | 0/24 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | — | — |
| downsample / unsharp | 0.0635 [0.0297, 0.1034] | 0/24 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | — | — |
| downsample / clahe | 0.0789 [0.0376, 0.1247] | 0/24 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | — | — |
| downsample / nlm | 0.1252 [0.0760, 0.1815] | 0/24 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0/24 0.000 [0.000, 0.000] | — | — |
| downsample / richardson_lucy | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| downsample / wiener | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

## Paired differences (enhanced minus degraded)

`0` means the 95% interval overlaps zero. `excludes 0` means it does not. `null` means there were fewer than 2 finite pairs (downsample PSNR and SSIM).

| arm | PSNR | SSIM | keypoint IoU | block IoU |
|---|---|---|---|---|
| gaussian / unsharp | 0.6068 [0.5831, 0.6291] (excludes 0) | 0.0369 [0.0368, 0.0371] (excludes 0) | 0.0297 [0.0073, 0.0535] (excludes 0) | -0.0085 [-0.0190, 0.0019] (overlaps 0) |
| gaussian / clahe | -4.0900 [-4.3848, -3.7945] (excludes 0) | -0.0344 [-0.0374, -0.0315] (excludes 0) | 0.0065 [-0.0284, 0.0404] (overlaps 0) | -0.2705 [-0.3459, -0.1879] (excludes 0) |
| gaussian / nlm | -0.6162 [-0.6902, -0.5376] (excludes 0) | -0.0899 [-0.0966, -0.0829] (excludes 0) | -0.0753 [-0.1238, -0.0343] (excludes 0) | -0.0750 [-0.1115, -0.0407] (excludes 0) |
| gaussian / richardson_lucy / oracle | 1.7582 [1.6784, 1.8349] (excludes 0) | 0.0946 [0.0939, 0.0954] (excludes 0) | 0.0978 [0.0690, 0.1300] (excludes 0) | -0.1086 [-0.1696, -0.0532] (excludes 0) |
| gaussian / richardson_lucy / wrong | 0.7153 [0.6758, 0.7562] (excludes 0) | 0.0491 [0.0475, 0.0506] (excludes 0) | 0.1021 [0.0616, 0.1454] (excludes 0) | -0.1834 [-0.2544, -0.1121] (excludes 0) |
| gaussian / wiener / oracle | 0.5069 [0.3772, 0.6494] (excludes 0) | 0.0898 [0.0876, 0.0922] (excludes 0) | 0.1213 [0.0879, 0.1559] (excludes 0) | -0.1325 [-0.2046, -0.0679] (excludes 0) |
| gaussian / wiener / wrong | -3.2681 [-3.5020, -3.0445] (excludes 0) | -0.0864 [-0.0889, -0.0836] (excludes 0) | 0.1173 [0.0786, 0.1587] (excludes 0) | -0.2356 [-0.3085, -0.1610] (excludes 0) |
| disk / unsharp | 0.0368 [0.0290, 0.0442] (excludes 0) | -0.0022 [-0.0025, -0.0019] (excludes 0) | 0.0078 [-0.0096, 0.0246] (overlaps 0) | -0.0085 [-0.0278, 0.0029] (overlaps 0) |
| disk / clahe | -2.8816 [-3.0823, -2.6837] (excludes 0) | -0.0484 [-0.0512, -0.0455] (excludes 0) | 0.0230 [-0.0004, 0.0480] (overlaps 0) | -0.2485 [-0.3178, -0.1737] (excludes 0) |
| disk / nlm | -0.1116 [-0.1344, -0.0887] (excludes 0) | -0.0078 [-0.0100, -0.0056] (excludes 0) | -0.0082 [-0.0244, 0.0079] (overlaps 0) | -0.0570 [-0.0845, -0.0325] (excludes 0) |
| disk / richardson_lucy / oracle | 1.3622 [1.3146, 1.4092] (excludes 0) | 0.0932 [0.0910, 0.0954] (excludes 0) | 0.0886 [0.0599, 0.1194] (excludes 0) | -0.1284 [-0.1794, -0.0804] (excludes 0) |
| disk / richardson_lucy / wrong | 0.5169 [0.4945, 0.5407] (excludes 0) | 0.0339 [0.0320, 0.0357] (excludes 0) | 0.0773 [0.0431, 0.1157] (excludes 0) | -0.2114 [-0.2770, -0.1448] (excludes 0) |
| disk / wiener / oracle | -0.6555 [-0.8907, -0.4144] (excludes 0) | 0.1223 [0.1184, 0.1264] (excludes 0) | 0.1266 [0.0847, 0.1674] (excludes 0) | -0.2485 [-0.3227, -0.1736] (excludes 0) |
| disk / wiener / wrong | -2.7045 [-2.9082, -2.4979] (excludes 0) | -0.0895 [-0.0932, -0.0857] (excludes 0) | 0.0918 [0.0525, 0.1332] (excludes 0) | -0.2485 [-0.3186, -0.1767] (excludes 0) |
| motion / unsharp | -0.0999 [-0.1113, -0.0886] (excludes 0) | 0.0034 [0.0030, 0.0038] (excludes 0) | 0.0130 [-0.0168, 0.0429] (overlaps 0) | -0.0218 [-0.0388, -0.0071] (excludes 0) |
| motion / clahe | -3.6005 [-3.8529, -3.3496] (excludes 0) | -0.0712 [-0.0738, -0.0688] (excludes 0) | 0.0033 [-0.0355, 0.0414] (overlaps 0) | -0.2656 [-0.3437, -0.1825] (excludes 0) |
| motion / nlm | -0.1486 [-0.1777, -0.1199] (excludes 0) | -0.0438 [-0.0454, -0.0423] (excludes 0) | -0.0500 [-0.0794, -0.0229] (excludes 0) | -0.0913 [-0.1413, -0.0486] (excludes 0) |
| motion / richardson_lucy / oracle | 2.9437 [2.9064, 2.9803] (excludes 0) | 0.1649 [0.1606, 0.1694] (excludes 0) | 0.0315 [-0.0071, 0.0686] (overlaps 0) | -0.1590 [-0.2242, -0.0968] (excludes 0) |
| motion / richardson_lucy / wrong | 0.2368 [0.1960, 0.2750] (excludes 0) | -0.0043 [-0.0102, 0.0011] (overlaps 0) | 0.0334 [-0.0079, 0.0751] (overlaps 0) | -0.2135 [-0.2843, -0.1420] (excludes 0) |
| motion / wiener / oracle | -0.5395 [-0.9155, -0.1603] (excludes 0) | 0.1495 [0.1397, 0.1591] (excludes 0) | 0.0382 [-0.0052, 0.0793] (overlaps 0) | -0.2656 [-0.3444, -0.1829] (excludes 0) |
| motion / wiener / wrong | -3.9439 [-4.0747, -3.8059] (excludes 0) | -0.2394 [-0.2459, -0.2328] (excludes 0) | -0.0261 [-0.0650, 0.0154] (overlaps 0) | -0.2656 [-0.3407, -0.1833] (excludes 0) |
| awgn / unsharp | -4.6074 [-4.6580, -4.5568] (excludes 0) | -0.1482 [-0.1500, -0.1463] (excludes 0) | -0.0097 [-0.0523, 0.0333] (overlaps 0) | 0.0000 [0.0000, 0.0000] (overlaps 0) |
| awgn / clahe | -7.5855 [-7.6976, -7.4740] (excludes 0) | -0.1738 [-0.1783, -0.1692] (excludes 0) | -0.0186 [-0.0548, 0.0183] (overlaps 0) | 0.0000 [0.0000, 0.0000] (overlaps 0) |
| awgn / nlm | 0.4999 [0.4293, 0.5721] (excludes 0) | 0.0282 [0.0233, 0.0337] (excludes 0) | 0.0060 [-0.0245, 0.0352] (overlaps 0) | 0.0000 [0.0000, 0.0000] (overlaps 0) |
| jpeg / unsharp | -0.9696 [-1.0023, -0.9359] (excludes 0) | -0.0260 [-0.0289, -0.0233] (excludes 0) | -0.0492 [-0.0849, -0.0149] (excludes 0) | 0.0000 [0.0000, 0.0000] (overlaps 0) |
| jpeg / clahe | -5.1382 [-5.5912, -4.6751] (excludes 0) | -0.1143 [-0.1185, -0.1101] (excludes 0) | -0.0396 [-0.0793, -0.0020] (excludes 0) | 0.0000 [0.0000, 0.0000] (overlaps 0) |
| jpeg / nlm | -0.0555 [-0.0935, -0.0167] (excludes 0) | -0.0268 [-0.0313, -0.0225] (excludes 0) | -0.0390 [-0.0752, -0.0019] (excludes 0) | 0.0000 [0.0000, 0.0000] (overlaps 0) |
| downsample / unsharp | — (null) | — (null) | -0.0411 [-0.0738, -0.0105] (excludes 0) | 0.0000 [0.0000, 0.0000] (overlaps 0) |
| downsample / clahe | — (null) | — (null) | -0.0257 [-0.0707, 0.0201] (overlaps 0) | 0.0000 [0.0000, 0.0000] (overlaps 0) |
| downsample / nlm | — (null) | — (null) | 0.0207 [-0.0139, 0.0551] (overlaps 0) | 0.0000 [0.0000, 0.0000] (overlaps 0) |

PSNR gain is not a detection gain on this set. Gaussian unsharp raises PSNR by 0.6068 and the interval excludes zero, while the paired block-IoU interval overlaps zero. Oracle Lucy on the same blur raises PSNR by 1.7582 (interval excludes zero) and lowers block IoU by 0.1086 (interval also excludes zero, in the other direction). AWGN followed by non-local means raises PSNR by 0.4999 (interval excludes zero) while the paired keypoint-IoU interval overlaps zero and block IoU stays 0. Wrong Wiener on Gaussian blur lowers PSNR by 3.2681 and raises keypoint IoU by 0.1173; both intervals exclude zero, so a worse similarity score can coincide with a higher keypoint overlap.
