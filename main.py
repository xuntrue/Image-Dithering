from pathlib import Path

import numpy as np
from PIL import Image

from src import dither, ImageLoader, ImageAnalyser
from src.pixelize import pixelize

IMAGE_NAMES = "Sphere.jpg" # None = every image in data/images/
MAX_SIZE = 200              # longest side after pixelizing
METHODS = ["floyd-steinberg", "atkinson", "jarvis", "sierra"]
METHOD = "floyd-steinberg"   
LEVELS = 4                  # Distinct brightness values
COLOR = True                # False = grayscale, True = dither R, G, B separately
BAYER_SIZE = 4              # only used when METHOD == "bayer"
UPSCALE = 1                # enlarge result so dither pixels are visible (1 = OFF)

SHOW_RESULTS = True
OUTPUT_DIR = Path("output")

def load_image_paths() -> list[Path]:
    paths = ImageLoader.list_images()
    if IMAGE_NAMES is not None:
        paths = [p for p in paths if p.name in IMAGE_NAMES]
    return paths

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

def analyse_image(img: Image.Image):
    original = img
    small = pixelize(original, 160)
    gray_small = small.convert("L").convert("RGB")

    bw = Image.fromarray(dither.apply(np.asarray(gray_small), "floyd-steinberg", 2)).convert("L")
    colour = Image.fromarray(dither.apply(np.asarray(small), "floyd-steinberg", 2))

    stages = [original, small, bw, colour]
    names = ["Original", "Pixelized (160px)", "Dithered B/W (2 levels)", "Dithered colour (2 levels)"]
    ImageAnalyser.compare(*stages, titles=names)

def main() -> None:
    # Step 1: load image(s)
    paths = load_image_paths()
    if not paths:
        return

    # Steps 2 & 3: pixelize an RGB image, then dither it
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in paths:
        print(f"Loading {path.name}. . .\n")
        original = ImageLoader.load_image(path)
        
        print(f"Analyzing {path.name}. . .\n")
        analyse_image(original)

        """
        print(f"\nDithering {path.name}. . .\n")
        dithered = dither_image(original)

        out_path = OUTPUT_DIR / f"{path.stem}_{METHOD}.png"
        dithered.save(out_path)
        print(f"{path.name}: {original.size[0]}x{original.size[1]} -> {dithered.size[0]}x{dithered.size[1]}, saved {out_path}")

        if SHOW_RESULTS:
            ImageLoader.show(original, dithered, titles=["Original", f"Dithered"])
        """
if __name__ == "__main__":
    main()