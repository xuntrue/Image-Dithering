import numpy as np

KERNELS = {
    "floyd-steinberg": (16, [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]),
    "atkinson": (8, [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)]),
    "jarvis": (48, [(1, 0, 7), (2, 0, 5),
                    (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
                    (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)]),
    "sierra": (32, [(1, 0, 5), (2, 0, 3),
                    (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2),
                    (-1, 2, 2), (0, 2, 3), (1, 2, 2)]),
}

METHODS = ["threshold", "random", "bayer", *KERNELS]

def _quantize(values: np.ndarray, levels: int) -> np.ndarray:
    """ Snap values (0-255) to the nearest of `levels` evenly spaced values """
    step = 255.0 / (levels - 1)
    return np.clip(np.round(values / step) * step, 0, 255)

def threshold(arr: np.ndarray, levels: int = 2) -> np.ndarray:
    """No dithering: just quantise. Useful as a baseline to compare against """
    return _quantize(arr.astype(np.float32), levels).astype(np.uint8)

def random_noise(arr: np.ndarray, levels: int = 2, amount: float = 1.0,
                 seed: int | None = None) -> np.ndarray:
    """ Random (white-noise) dithering: add random noise to every pixel, then quantise """
    h, w, _ = arr.shape
    rng = np.random.default_rng(seed)
    step = 255.0 / (levels - 1)
    noise = rng.uniform(-0.5, 0.5, size=(h, w, 1)) * amount * step
    return _quantize(arr.astype(np.float32) + noise, levels).astype(np.uint8)

def bayer_matrix(n: int) -> np.ndarray:
    """ Build an n x n Bayer index matrix (n must be a power of two) """
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2],
                      [4 * m + 3, 4 * m + 1]])
    return m

def ordered(arr: np.ndarray, levels: int = 2, size: int = 4) -> np.ndarray:
    """ Ordered dithering with a Bayer matrix: a fixed, repeating threshold pattern """
    h, w, _ = arr.shape
    m = bayer_matrix(size)
    thresholds = (m + 0.5) / m.size - 0.5   # range about -0.5 .. +0.5
    tiled = np.tile(thresholds, (h // size + 1, w // size + 1))[:h, :w, None]
    step = 255.0 / (levels - 1)
    return _quantize(arr.astype(np.float32) + tiled * step, levels).astype(np.uint8)


def error_diffusion(arr: np.ndarray, kernel: str = "floyd-steinberg",
                    levels: int = 2, serpentine: bool = True) -> np.ndarray:
    """ Error-diffusion dithering: push each pixel's rounding error onto neighbours.

    serpentine=True alternates scan direction each row, which avoids the
    diagonal "worm" artefacts of always scanning left-to-right.
    """
    divisor, taps = KERNELS[kernel]
    out = arr.astype(np.float32).copy()
    h, w, _ = out.shape

    for y in range(h):
        left_to_right = not (serpentine and y % 2)
        xs = range(w) if left_to_right else range(w - 1, -1, -1)
        for x in xs:
            old = out[y, x].copy()
            new = _quantize(old, levels)
            out[y, x] = new
            err = old - new
            for dx, dy, weight in taps:
                nx = x + dx if left_to_right else x - dx
                ny = y + dy
                if 0 <= nx < w and ny < h:
                    out[ny, nx] += err * (weight / divisor)

    return np.clip(out, 0, 255).astype(np.uint8)


def apply(arr: np.ndarray, method: str, levels: int = 2, bayer_size: int = 4,
          seed: int | None = None) -> np.ndarray:
    if method == "threshold":
        return threshold(arr, levels)
    if method == "random":
        return random_noise(arr, levels, seed=seed)
    if method == "bayer":
        return ordered(arr, levels, bayer_size)
    if method in KERNELS:
        return error_diffusion(arr, method, levels)
    raise ValueError(f"Unknown method {method!r}. Choose from {METHODS}")