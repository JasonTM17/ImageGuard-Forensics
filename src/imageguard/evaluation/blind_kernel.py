"""Experimental edge-based blur-kernel estimates for evaluation studies."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imageguard.evaluation.enhance_ops import disk_kernel, motion_kernel

_PROFILE_RADIUS = 12
_MAX_EDGE_PROFILES = 12
_CONTRAST_FLOOR = 16.0
_SHIFTS = np.arange(-1.5, 1.51, 0.5, dtype=np.float32)


def _not_run(reason: str) -> tuple[None, dict[str, Any]]:
    return None, {"status": "NOT_RUN", "reason": reason}


def _as_gray_u8(image: np.ndarray) -> np.ndarray | None:
    source = np.asarray(image)
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or min(source.shape[:2]) < 32
        or not np.issubdtype(source.dtype, np.number)
        or np.iscomplexobj(source)
    ):
        return None
    values = source.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        return None
    if values.size and float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
        values = values * 255.0
    bgr = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _edge_profile(
    gray: np.ndarray, line: np.ndarray
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = (float(value) for value in line.reshape(-1))
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length < 20.0:
        return None

    tangent = np.asarray([dx, dy], dtype=np.float64) / length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    count = min(13, max(3, int(length // 10)))
    positions = np.linspace(0.2, 0.8, count, dtype=np.float32)
    centers_x = x1 + dx * positions
    centers_y = y1 + dy * positions
    offsets = np.arange(-_PROFILE_RADIUS, _PROFILE_RADIUS + 1, dtype=np.float32)
    map_x = centers_x[:, None] + normal[0] * offsets[None, :]
    map_y = centers_y[:, None] + normal[1] * offsets[None, :]
    samples = cv2.remap(
        gray,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    profile = np.median(samples, axis=0).astype(np.float32)
    low = float(np.median(profile[:4]))
    high = float(np.median(profile[-4:]))
    if high < low:
        normal = -normal
        profile = profile[::-1].copy()
        low, high = high, low
    contrast = high - low
    if not np.isfinite(contrast) or contrast < _CONTRAST_FLOOR:
        return None

    normalized = np.clip((profile - low) / contrast, 0.0, 1.0)
    increments = np.diff(normalized)
    rising = float(np.maximum(increments, 0.0).sum(dtype=np.float64))
    falling = float(np.maximum(-increments, 0.0).sum(dtype=np.float64))
    if rising < 0.35 or falling > max(0.12, rising * 0.4):
        return None

    angle = float(np.arctan2(normal[1], normal[0]) % np.pi)
    return {
        "normal": normal.astype(np.float32),
        "angle": angle,
        "profile": normalized.astype(np.float32),
        "offsets": offsets,
        "length": length,
        "position": float(np.dot(normal, [(x1 + x2) * 0.5, (y1 + y2) * 0.5])),
    }


def _edge_profiles(gray: np.ndarray) -> list[dict[str, Any]]:
    edges = cv2.Canny(gray, 35, 105, L2gradient=True)
    minimum_length = max(20, min(gray.shape) // 10)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(12, minimum_length // 2),
        minLineLength=minimum_length,
        maxLineGap=max(2, min(gray.shape) // 96),
    )
    if lines is None:
        return []

    candidates: list[dict[str, Any]] = []
    ordered = sorted(
        (line.reshape(-1) for line in lines),
        key=lambda line: float(np.hypot(line[2] - line[0], line[3] - line[1])),
        reverse=True,
    )
    for line in ordered:
        profile = _edge_profile(gray, np.asarray(line))
        if profile is None:
            continue
        duplicate = any(
            abs((profile["angle"] - other["angle"] + np.pi / 2) % np.pi - np.pi / 2)
            < np.deg2rad(8.0)
            and abs(profile["position"] - other["position"]) < 7.0
            for other in candidates
        )
        if not duplicate:
            candidates.append(profile)

    # Keep the longest independent edge evidence from each direction first.
    candidates.sort(key=lambda item: item["length"], reverse=True)
    selected: list[dict[str, Any]] = []
    per_direction: dict[int, int] = {}
    for profile in candidates:
        direction = round(np.rad2deg(profile["angle"]) / 15.0) % 12
        if per_direction.get(direction, 0) >= 2:
            continue
        selected.append(profile)
        per_direction[direction] = per_direction.get(direction, 0) + 1
        if len(selected) >= _MAX_EDGE_PROFILES:
            break
    return selected


def _has_directional_diversity(profiles: list[dict[str, Any]]) -> bool:
    for index, first in enumerate(profiles):
        for second in profiles[index + 1 :]:
            difference = abs(first["angle"] - second["angle"])
            difference = min(difference, np.pi - difference)
            if difference >= np.deg2rad(20.0):
                return True
    return False


def _analytic_step_profile(
    kernel: np.ndarray, normal: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    height, width = kernel.shape
    rows, columns = np.indices((height, width), dtype=np.float32)
    projection = (
        (columns - width // 2) * float(normal[0])
        + (rows - height // 2) * float(normal[1])
    ).reshape(-1)
    weights = np.asarray(kernel, dtype=np.float64).reshape(-1)
    order = np.argsort(projection)
    cumulative = np.concatenate(([0.0], np.cumsum(weights[order], dtype=np.float64)))
    indexes = np.searchsorted(projection[order], offsets, side="right")
    return cumulative[indexes].astype(np.float32)


def _prepare_step_model(normal: np.ndarray, offsets: np.ndarray) -> dict[str, Any]:
    margin = _PROFILE_RADIUS + 8 + 4
    side = 2 * margin + 1
    center = side // 2
    rows, columns = np.indices((side, side), dtype=np.float32)
    signed_distance = (
        (columns - center) * float(normal[0])
        + (rows - center) * float(normal[1])
    )
    step = (signed_distance >= 0.0).astype(np.float32)
    map_x = (center + offsets * normal[0]).reshape(1, -1).astype(np.float32)
    map_y = (center + offsets * normal[1]).reshape(1, -1).astype(np.float32)
    return {"step": step, "map_x": map_x, "map_y": map_y}


def _filter2d_step_profile(
    kernel: np.ndarray, model: dict[str, Any]
) -> np.ndarray:
    filtered = cv2.filter2D(
        model["step"],
        ddepth=cv2.CV_32F,
        kernel=kernel,
        borderType=cv2.BORDER_REPLICATE,
    )
    profile = cv2.remap(
        filtered,
        model["map_x"],
        model["map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.asarray(profile, dtype=np.float32).reshape(-1)


def _profile_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    offsets = np.arange(-_PROFILE_RADIUS, _PROFILE_RADIUS + 1, dtype=np.float32)
    errors = []
    for shift in _SHIFTS:
        aligned = np.interp(
            offsets - shift,
            offsets,
            predicted,
            left=0.0,
            right=1.0,
        )
        errors.append(float(np.mean((observed - aligned) ** 2, dtype=np.float64)))
    return min(errors)


def _candidates(include_motion: bool) -> list[tuple[dict[str, Any], np.ndarray]]:
    candidates = [
        ({"family": "disk", "parameters": {"radius": radius}}, disk_kernel(radius))
        for radius in range(1, 9)
    ]
    if include_motion:
        for length in range(3, 16, 2):
            for angle in range(0, 180, 5):
                candidates.append(
                    (
                        {
                            "family": "motion",
                            "parameters": {
                                "length": length,
                                "angle_deg": float(angle),
                            },
                        },
                        motion_kernel(length, float(angle)),
                    )
                )
    return candidates


def _candidate_scores(
    profiles: list[dict[str, Any]],
    candidates: list[tuple[dict[str, Any], np.ndarray]],
    *,
    use_filter2d: bool,
) -> list[tuple[float, dict[str, Any], np.ndarray]]:
    prepared = [
        _prepare_step_model(profile["normal"], profile["offsets"])
        for profile in profiles
    ]
    scored: list[tuple[float, dict[str, Any], np.ndarray]] = []
    for record, kernel in candidates:
        losses = []
        for profile, model in zip(profiles, prepared, strict=True):
            if use_filter2d:
                predicted = _filter2d_step_profile(kernel, model)
            else:
                predicted = _analytic_step_profile(
                    kernel, profile["normal"], profile["offsets"]
                )
            losses.append(_profile_error(profile["profile"], predicted))
        score = float(np.mean(losses, dtype=np.float64))
        if np.isfinite(score):
            scored.append((score, record, kernel))
    return scored


def _valid_estimate(kernel: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    estimate = np.asarray(kernel, dtype=np.float64)
    if estimate.ndim != 2 or estimate.size == 0 or not np.all(np.isfinite(estimate)):
        return None, "estimated kernel is nonfinite"
    if np.any(estimate < 0.0):
        return None, "estimated kernel has a negative weight"
    mass = float(estimate.sum(dtype=np.float64))
    if not np.isfinite(mass) or mass <= 0.0:
        return None, "estimated kernel has zero mass"
    estimate /= mass
    if float(estimate.max()) >= 0.999:
        return None, "estimated kernel is near-delta"
    return estimate.astype(np.float32), None


def estimate_kernel(image: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate a disk or motion PSF from isolated straight edges in a BGR image.

    The estimator only receives image pixels. It does not accept or inspect the
    generating study parameters.
    """
    gray = _as_gray_u8(image)
    if gray is None:
        return _not_run("image must be a finite BGR array at least 32 pixels wide")
    profiles = _edge_profiles(gray)
    if not profiles:
        return _not_run("insufficient straight-edge evidence")

    include_motion = _has_directional_diversity(profiles)
    candidates = _candidates(include_motion)
    coarse = _candidate_scores(profiles, candidates, use_filter2d=False)
    if not coarse:
        return _not_run("candidate step fit was nonfinite")

    by_family: dict[str, list[tuple[float, dict[str, Any], np.ndarray]]] = {
        "disk": [],
        "motion": [],
    }
    for candidate in coarse:
        by_family[candidate[1]["family"]].append(candidate)
    shortlist = []
    for family_candidates in by_family.values():
        shortlist.extend(sorted(family_candidates, key=lambda item: item[0])[:5])

    refined = _candidate_scores(
        profiles,
        [(record, kernel) for _, record, kernel in shortlist],
        use_filter2d=True,
    )
    if not refined:
        return _not_run("candidate step fit was nonfinite")
    best = min(refined, key=lambda item: item[0])

    # Refine the motion angle locally after the bounded coarse search.
    if best[1]["family"] == "motion":
        length = int(best[1]["parameters"]["length"])
        angle = float(best[1]["parameters"]["angle_deg"])
        local_candidates = []
        for delta in range(-4, 5):
            local_angle = float((angle + delta) % 180.0)
            record = {
                "family": "motion",
                "parameters": {"length": length, "angle_deg": local_angle},
            }
            local_candidates.append((record, motion_kernel(length, local_angle)))
        local = _candidate_scores(profiles, local_candidates, use_filter2d=True)
        if local:
            best = min([best, *local], key=lambda item: item[0])

    estimate, reason = _valid_estimate(best[2])
    if estimate is None:
        return _not_run(reason or "invalid estimated kernel")
    return estimate, {
        "status": "RUN",
        "family": best[1]["family"],
        "parameters": best[1]["parameters"],
        "fit_error": float(best[0]),
        "edge_count": len(profiles),
    }


def _unit_kernel(kernel: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(kernel, dtype=np.float64)
    if result.ndim != 2 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite non-empty 2D kernel")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must not have negative weights")
    total = float(result.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return result / total


def kernel_error(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Return translation-aligned normalized L2 error between unit-sum kernels."""
    estimate = _unit_kernel(estimated, "estimated kernel")
    truth = _unit_kernel(ground_truth, "ground-truth kernel")
    common_height = max(estimate.shape[0], truth.shape[0])
    common_width = max(estimate.shape[1], truth.shape[1])
    max_dimension = max(common_height, common_width)
    canvas_shape = (3 * max_dimension + 1, 3 * max_dimension + 1)

    def centered(kernel: np.ndarray) -> np.ndarray:
        canvas = np.zeros(canvas_shape, dtype=np.float64)
        center_y, center_x = canvas_shape[0] // 2, canvas_shape[1] // 2
        top = center_y - kernel.shape[0] // 2
        left = center_x - kernel.shape[1] // 2
        canvas[top : top + kernel.shape[0], left : left + kernel.shape[1]] = kernel
        return canvas

    def translated(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
        result = np.zeros_like(array)
        y0 = max(0, dy)
        y1 = min(array.shape[0], array.shape[0] + dy)
        x0 = max(0, dx)
        x1 = min(array.shape[1], array.shape[1] + dx)
        result[y0:y1, x0:x1] = array[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        return result

    estimate_canvas = centered(estimate)
    truth_canvas = centered(truth)
    denominator = float(np.linalg.norm(truth_canvas))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("ground-truth kernel has zero norm")
    shift_limit_y = common_height // 2
    shift_limit_x = common_width // 2
    best = float("inf")
    for dy in range(-shift_limit_y, shift_limit_y + 1):
        for dx in range(-shift_limit_x, shift_limit_x + 1):
            difference = translated(estimate_canvas, dy, dx) - truth_canvas
            score = float(np.linalg.norm(difference) / denominator)
            best = min(best, score)
    if not np.isfinite(best):
        raise ValueError("kernel error is nonfinite")
    return best
