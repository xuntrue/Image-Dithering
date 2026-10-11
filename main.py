from pathlib import Path

import numpy as np
from PIL import Image

from src import dither, ImageLoader
from src.pixelize import pixelize

IMAGE_NAMES = None          # None = every image in data/images, or e.g. ["photo.jpg"]
MAX_SIZE = 160              # longest side after pixelizing
METHODS = ["floyd-steinberg", "atkinson", "jarvis", "sierra"]
METHOD = "floyd-steinberg"    
LEVELS = 8                  # levels per channel (2 = black/white, or 8 colours with COLOR)
COLOR = True               # False = grayscale, True = dither R, G, B separately
BAYER_SIZE = 4              # only used when METHOD == "bayer"
UPSCALE = 4                 # enlarge result so dither pixels are visible (1 = OFF)

SHOW_RESULTS = True
OUTPUT_DIR = Path("output")

def dither_image(img: Image.Image) -> Image.Image:
    # Step 2: pixelize
    small = pixelize(img, MAX_SIZE)
    if not COLOR:
        small = small.convert("L").convert("RGB")   # grayscale, kept as 3 channels

    # Step 3: dither
    result = dither.apply(np.asarray(small), METHOD, LEVELS, BAYER_SIZE)
    out = Image.fromarray(result)
    if not COLOR:
        out = out.convert("L")
    if UPSCALE > 1:
        out = out.resize((out.width * UPSCALE, out.height * UPSCALE), Image.Resampling.NEAREST)
    return out

def main() -> None:
    # Step 1: load images
    paths = ImageLoader.list_images()
    if IMAGE_NAMES is not None:
        paths = [p for p in paths if p.name in IMAGE_NAMES]
    if not paths:
        print("ERROR: No images found.")
        return

    # Steps 2 & 3: pixelize an RGB image, then dither it
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in paths:
        original = ImageLoader.load_image(path)
        dithered = dither_image(original)

        out_path = OUTPUT_DIR / f"{path.stem}_{METHOD}.png"
        dithered.save(out_path)
        print(f"{path.name}: {original.size[0]}x{original.size[1]} -> {dithered.size[0]}x{dithered.size[1]}, saved {out_path}")

        if SHOW_RESULTS:
            ImageLoader.show(original, dithered, titles=["Original", f"{METHOD} dithered"])

if __name__ == "__main__":
    main()
