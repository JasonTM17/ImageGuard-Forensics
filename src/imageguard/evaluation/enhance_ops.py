"""Classical image degradation and restoration operators for evaluation studies.

These operators produce controlled image variants. Their outputs are method
inputs, not evidence that an image has been restored to an original state.
"""

from __future__ import annotations

import cv2
import numpy as np


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _real(value: float, name: str, minimum: float | None = None) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _image_copy(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim not in (2, 3) or source.size == 0:
        raise ValueError("image must be a non-empty 2D or 3D array")
    return source.copy()


def _uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _normalized_kernel(kernel: np.ndarray) -> np.ndarray:
    result = np.asarray(kernel, dtype=np.float32)
    if result.ndim != 2 or result.size == 0:
        raise ValueError("kernel must be a non-empty 2D array")
    total = float(result.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("kernel must have a finite positive sum")
    return result / total


def gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    """Return a normalized square Gaussian kernel with odd dimensions."""
    size = _integer(ksize, "ksize")
    if size % 2 == 0:
        raise ValueError("ksize must be odd")
    spread = _real(sigma, "sigma", minimum=np.finfo(float).tiny)
    vector = cv2.getGaussianKernel(size, spread, cv2.CV_64F)
    return _normalized_kernel(vector @ vector.T)


def disk_kernel(radius: int) -> np.ndarray:
    """Return a normalized filled elliptical kernel of the requested radius."""
    r = _integer(radius, "radius", minimum=0)
    size = 2 * r + 1
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return _normalized_kernel(ellipse)


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Return a normalized, rasterized straight-line motion kernel."""
    size = _integer(length, "length")
    angle = _real(angle_deg, "angle_deg")
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = (size - 1) / 2.0
    half = center
    radians = np.deg2rad(angle)
    dx = half * float(np.cos(radians))
    dy = half * float(np.sin(radians))
    start = (round(center - dx), round(center - dy))
    end = (round(center + dx), round(center + dy))
    cv2.line(kernel, start, end, color=1, thickness=1, lineType=cv2.LINE_8)
    if not np.any(kernel):
        kernel[round(center), round(center)] = 1
    return _normalized_kernel(kernel)


def _metadata(kind: str, **parameters: object) -> dict:
    return {"kind": kind, **parameters}


def _awgn_generator(
    rng: np.random.Generator | int,
    declared_seed: object | None,
) -> tuple[np.random.Generator, int | list[int] | None]:
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
        seed = int(rng)
        if declared_seed is not None and int(declared_seed) != seed:
            raise ValueError("params seed must match the integer rng seed")
        return np.random.default_rng(seed), seed
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an integer seed or numpy.random.Generator")

    seed = declared_seed
    if seed is None:
        seed_sequence = getattr(rng.bit_generator, "_seed_seq", None)
        seed = getattr(seed_sequence, "entropy", None)
    if isinstance(seed, np.ndarray):
        seed = seed.tolist()
    elif isinstance(seed, np.integer):
        seed = int(seed)
    elif isinstance(seed, (list, tuple)):
        seed = [int(part) for part in seed]
    elif seed is not None:
        seed = int(seed)
    return rng, seed


def degrade(
    image: np.ndarray,
    kind: str,
    params: dict,
    rng: np.random.Generator | int | None,
) -> tuple[np.ndarray, dict]:
    """Apply one controlled degradation and return its effective parameters."""
    source = _image_copy(image)
    if kind == "gaussian":
        size = _integer(params["ksize"], "ksize")
        if size % 2 == 0:
            raise ValueError("ksize must be odd")
        sigma = _real(params["sigma"], "sigma", minimum=np.finfo(float).tiny)
        kernel = gaussian_kernel(size, sigma)
        output = cv2.filter2D(source, -1, kernel)
        return output, _metadata(
            kind, ksize=size, sigma=sigma, kernel=kernel.tolist()
        )

    if kind == "disk":
        radius = _integer(params["radius"], "radius", minimum=0)
        kernel = disk_kernel(radius)
        output = cv2.filter2D(source, -1, kernel)
        return output, _metadata(kind, radius=radius, kernel=kernel.tolist())

    if kind == "motion":
        length = _integer(params["length"], "length")
        angle = _real(params["angle_deg"], "angle_deg")
        kernel = motion_kernel(length, angle)
        output = cv2.filter2D(source, -1, kernel)
        return output, _metadata(
            kind, length=length, angle_deg=angle, kernel=kernel.tolist()
        )

    if kind == "awgn":
        sigma = _real(params["sigma"], "sigma", minimum=0.0)
        generator, seed = _awgn_generator(rng, params.get("seed"))
        noise = generator.normal(0.0, sigma, size=source.shape)
        output = _uint8(source.astype(np.float64) + noise)
        return output, _metadata(kind, sigma=sigma, seed=seed)

    if kind == "jpeg":
        quality = _integer(params["quality"], "quality", minimum=0)
        if quality > 100:
            raise ValueError("quality must be in [0, 100]")
        ok, encoded = cv2.imencode(
            ".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            raise ValueError("OpenCV could not encode the image as JPEG")
        output = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if output is None:
            raise ValueError("OpenCV could not decode the JPEG image")
        return output, _metadata(kind, quality=quality)

    if kind == "downsample":
        factor = _integer(params["factor"], "factor")
        height, width = source.shape[:2]
        target = (width // factor, height // factor)
        if target[0] == 0 or target[1] == 0:
            raise ValueError("factor must leave a non-zero output dimension")
        output = cv2.resize(source, target, interpolation=cv2.INTER_AREA)
        return output, _metadata(kind, factor=factor)

    raise ValueError(f"unsupported degradation kind: {kind}")


def _kernel_from_params(params: dict) -> np.ndarray:
    if "kernel" not in params:
        raise ValueError("params must include a kernel")
    return _normalized_kernel(np.asarray(params["kernel"], dtype=np.float32))


def _preserve_wrong_delta(params: dict, metadata: dict) -> None:
    for name in ("wrong_sigma", "wrong_radius", "wrong_angle_deg"):
        if name in params:
            metadata[name] = params[name]


def _richardson_lucy(
    source: np.ndarray, kernel: np.ndarray, iterations: int
) -> np.ndarray:
    observed = source.astype(np.float32)
    estimate = observed.copy()
    flipped = cv2.flip(kernel, -1)
    epsilon = np.finfo(np.float32).eps
    for _ in range(iterations):
        blurred = cv2.filter2D(estimate, -1, kernel, borderType=cv2.BORDER_REFLECT)
        ratio = observed / np.maximum(blurred, epsilon)
        correction = cv2.filter2D(
            ratio, -1, flipped, borderType=cv2.BORDER_REFLECT
        )
        estimate *= correction
    return _uint8(estimate)


def _wiener_channel(
    channel: np.ndarray, kernel: np.ndarray, nsr: float
) -> np.ndarray:
    height, width = channel.shape
    padded = np.zeros((height, width), dtype=np.float64)
    for row in range(kernel.shape[0]):
        for column in range(kernel.shape[1]):
            padded[row % height, column % width] += float(kernel[row, column])
    padded = np.roll(
        padded,
        shift=(-(kernel.shape[0] // 2), -(kernel.shape[1] // 2)),
        axis=(0, 1),
    )
    transfer = np.fft.fft2(padded)
    power = np.abs(transfer) ** 2
    denominator = np.maximum(power + nsr, np.finfo(np.float64).eps)
    estimate = np.fft.ifft2(
        np.fft.fft2(channel.astype(np.float64))
        * np.conj(transfer)
        / denominator
    ).real
    return estimate


def _wiener(source: np.ndarray, kernel: np.ndarray, nsr: float) -> np.ndarray:
    if source.ndim == 2:
        restored = _wiener_channel(source, kernel, nsr)
    else:
        restored = np.stack(
            [
                _wiener_channel(source[:, :, channel], kernel, nsr)
                for channel in range(source.shape[2])
            ],
            axis=2,
        )
    return _uint8(restored)


def restore(image: np.ndarray, kind: str, params: dict) -> tuple[np.ndarray, dict]:
    """Apply a classical restoration operator and record its effective inputs."""
    source = _image_copy(image)

    if kind == "unsharp":
        size = _integer(params["ksize"], "ksize")
        if size % 2 == 0:
            raise ValueError("ksize must be odd")
        sigma = _real(params["sigma"], "sigma", minimum=np.finfo(float).tiny)
        amount = _real(params["amount"], "amount", minimum=0.0)
        blurred = cv2.GaussianBlur(source, (size, size), sigmaX=sigma)
        restored = source.astype(np.float64) + amount * (
            source.astype(np.float64) - blurred.astype(np.float64)
        )
        output = _uint8(restored)
        return output, _metadata(
            kind, ksize=size, sigma=sigma, amount=amount
        )

    if kind == "clahe":
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("clahe expects a three-channel BGR image")
        clip_limit = _real(params["clip_limit"], "clip_limit", minimum=0.0)
        tile_value = params["tile_grid_size"]
        if len(tile_value) != 2:
            raise ValueError("tile_grid_size must have two values")
        tile = (
            _integer(tile_value[0], "tile_grid_size[0]"),
            _integer(tile_value[1], "tile_grid_size[1]"),
        )
        lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
        adjusted = clahe.apply(lightness)
        output = cv2.cvtColor(
            cv2.merge((adjusted, a_channel, b_channel)), cv2.COLOR_LAB2BGR
        )
        return output, _metadata(
            kind, clip_limit=clip_limit, tile_grid_size=list(tile)
        )

    if kind == "nlm":
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("nlm expects a three-channel BGR image")
        h = _real(params["h"], "h", minimum=0.0)
        h_color = _real(params["hColor"], "hColor", minimum=0.0)
        template_size = _integer(params["templateWindowSize"], "templateWindowSize")
        search_size = _integer(params["searchWindowSize"], "searchWindowSize")
        if template_size % 2 == 0 or search_size % 2 == 0:
            raise ValueError("NLM window sizes must be odd")
        output = cv2.fastNlMeansDenoisingColored(
            source,
            None,
            h=h,
            hColor=h_color,
            templateWindowSize=template_size,
            searchWindowSize=search_size,
        )
        return output, _metadata(
            kind,
            h=h,
            hColor=h_color,
            templateWindowSize=template_size,
            searchWindowSize=search_size,
        )

    if kind in ("richardson_lucy", "wiener"):
        role = params["kernel_role"]
        if role not in ("oracle", "wrong", "estimated"):
            raise ValueError(
                "kernel_role must be 'oracle', 'wrong', or 'estimated'"
            )
        kernel = _kernel_from_params(params)
        kernel_record = kernel.tolist()
        if kind == "richardson_lucy":
            iterations = _integer(params["iterations"], "iterations")
            output = _richardson_lucy(source, kernel, iterations)
            metadata = _metadata(
                kind,
                iterations=iterations,
                kernel=kernel_record,
                kernel_role=role,
            )
        else:
            nsr = _real(params["nsr"], "nsr", minimum=0.0)
            output = _wiener(source, kernel, nsr)
            metadata = _metadata(
                kind, nsr=nsr, kernel=kernel_record, kernel_role=role
            )
        _preserve_wrong_delta(params, metadata)
        return output, metadata

    raise ValueError(f"unsupported restoration kind: {kind}")


def wrong_kernel(kind: str, params: dict) -> np.ndarray:
    """Build a distinct blur kernel and record its changed parameter in params.

    ``params`` may be the metadata dict returned by :func:`degrade`; the
    selected ``wrong_*`` value is added to that dict for later reporting.
    """
    if not isinstance(params, dict):
        raise TypeError("params must be a dict so the wrong parameter is recorded")

    if kind == "gaussian":
        size = _integer(params["ksize"], "ksize")
        sigma = _real(params["sigma"], "sigma", minimum=np.finfo(float).tiny)
        actual = np.asarray(
            params.get("kernel", gaussian_kernel(size, sigma)), dtype=np.float32
        )
        wrong_sigma = sigma + max(0.5, abs(sigma) * 0.5)
        candidate = gaussian_kernel(size, wrong_sigma)
        if np.array_equal(candidate, actual):
            raise ValueError("ksize=1 cannot represent a different Gaussian kernel")
        params["wrong_sigma"] = wrong_sigma
        return candidate

    if kind == "disk":
        radius = _integer(params["radius"], "radius", minimum=0)
        actual = np.asarray(
            params.get("kernel", disk_kernel(radius)), dtype=np.float32
        )
        wrong_radius = radius + 1
        candidate = disk_kernel(wrong_radius)
        if np.array_equal(candidate, actual):
            raise ValueError("the selected radius cannot represent a different disk")
        params["wrong_radius"] = wrong_radius
        return candidate

    if kind == "motion":
        length = _integer(params["length"], "length")
        angle = _real(params["angle_deg"], "angle_deg")
        actual = np.asarray(
            params.get("kernel", motion_kernel(length, angle)), dtype=np.float32
        )
        for offset in range(1, 181):
            wrong_angle = float((angle + offset) % 180.0)
            candidate = motion_kernel(length, wrong_angle)
            if not np.array_equal(candidate, actual):
                params["wrong_angle_deg"] = wrong_angle
                return candidate
        raise ValueError("length=1 cannot represent a different motion kernel")

    raise ValueError(f"unsupported blur kind: {kind}")
