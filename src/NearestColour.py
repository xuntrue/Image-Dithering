import numpy as np
import random

from PIL import Image

import dither
import ImageLoader, ImageAnalyser
from pixelize import pixelize

IMAGE_NAMES = "Rock.png" # None = every image in data/images/
MAX_SIZE = 10_000
METHODS = ["floyd-steinberg", "atkinson", "jarvis", "sierra"]
METHOD = "floyd-steinberg"
LEVELS = 5
COLOR = True

def load_image_paths() -> list[Path]:
    paths = ImageLoader.list_images()
    if IMAGE_NAMES is not None:
        paths = [p for p in paths if p.name in IMAGE_NAMES]
    return paths

def analyse_image(img: Image.Image):
    original = img
        
    stages = [original, nearest_colour(original, 2, COLOR)]
    #print(f"Images: {len(stages)} colours")

    #stages = [original, colour]
    names = ["Original", f"Colour ({2} levels)"]
    ImageAnalyser.compare(*stages, titles=names)

def build_palette(levels: int, color: bool = True) -> np.ndarray:
    """ Every colour available with `levels` values per channel, shape (P, 3).

    color=True : levels**3 colours (every combination of R, G, B levels)
    color=False: `levels` grays (R = G = B)
    """
    values = np.linspace(0, 255, levels)   # e.g. levels=3 -> 0, 127.5, 255
    if color:
        r, g, b = np.meshgrid(values, values, values, indexing="ij")
        return np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)
    return np.stack([values, values, values], axis=1)


def nearest_colour(img: Image.Image, levels: int, color: bool = True, chunk_size: int = 20_000) -> Image.Image:
    """Replace every pixel with the closest palette colour (Euclidean distance) which keeps memory use small on large images """
    if levels < 2:
        raise ValueError("levels must be at least 2")

    palette = build_palette(levels, color) # (P, 3)

    rgb = img.convert("RGB")
    if not color:
        rgb = rgb.convert("L").convert("RGB") # compare brightness only
    width, height = rgb.size
    pixels = np.asarray(rgb, dtype=np.float32).reshape(-1, 3) # (N, 3), one row per pixel

    result = np.empty_like(pixels)
    for start in range(0, len(pixels), chunk_size):
        block = pixels[start:start + chunk_size]                  # (n, 3)
        # Squared distance from every pixel in the block to every palette colour: (n, P)
        dist_sq = ((block[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        result[start:start + chunk_size] = palette[dist_sq.argmin(axis=1)]

    out = Image.fromarray(np.round(result).astype(np.uint8).reshape(height, width, 3))
    return out if color else out.convert("L")

def main() -> None:
    paths = load_image_paths()
    if not paths:
        return

    path = random.choice(paths)
    original = pixelize(ImageLoader.load_image(path), MAX_SIZE) # small image keeps this quick
    if not COLOR:
        original = original.convert("L").convert("RGB")


    nearest = nearest_colour(original, LEVELS, COLOR)
    #ImageLoader.show([original, nearest], titles=["Original", "Nearest"])
    # Test 1: on an evenly spaced grid, "nearest colour" is the same as rounding each channel
    """
    nearest = nearest_colour(original, LEVELS, COLOR)
    rounded = dither.threshold(np.asarray(original), LEVELS)
    diff = np.abs(np.asarray(nearest.convert("RGB")).astype(int) - rounded.astype(int)).max()
    assert diff <= 1, f"nearest-colour disagrees with per-channel rounding (max diff {diff})"
    print(f"{path.name}: nearest-colour matches per-channel rounding (max difference {diff})")
    print(f"Palette size: {len(build_palette(LEVELS, COLOR))} colours")

    titles=[f"Original ({original.size[0]}x{original.size[1]})", f"Nearest ({nearest.size[0]}x{nearest.size[1]})"]

    ImageLoader.show(original, nearest, titles=titles, caption=None)#f"MAX_SIZE={MAX_SIZE} LEVELS={LEVELS}")
    """
    #analyse_image(original)

    if True:
        out_path = f"{path.stem}.png"
        nearest.save(out_path)
        print(f"{path.name}: {original.size[0]}x{original.size[1]} -> {nearest.size[0]}x{nearest.size[1]}, saved {out_path}")

if __name__ == "__main__":
    main()