import io
import matplotlib.pyplot as plt
import random

from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "images"

def list_images(directory: Path | str = DEFAULT_IMAGE_DIR) -> list[Path]:
    """ Return every supported image file in `directory`, sorted by name """
    directory = Path(directory)
    return sorted( # Sort alphabetically
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

def load_image(path: Path | str, svg_width: int = 1024) -> Image.Image:
    """ Load an image file and return it as an RGB `Image.Image` """
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        path = DEFAULT_IMAGE_DIR / path # allow bare file names
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    img = _load_svg(path, svg_width) if ext == ".svg" else Image.open(path)
    img = ImageOps.exif_transpose(img) # respect phone-camera rotation

    return _to_rgb(img)

def show(*images: Image.Image, titles: list[str] | None = None) -> None:
    """Display one or more images side by side with matplotlib """
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5), squeeze=False)
    for i, (ax, img) in enumerate(zip(axes[0], images)):
        ax.imshow(img, cmap="gray" if img.mode in ("L", "1") else None,
                  interpolation="nearest")     # nearest keeps dither pixels crisp
        ax.set_title(titles[i] if titles else f"Image {i + 1}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
  
def show_zoom(original: Image.Image, dithered: Image.Image, region: tuple[int, int, int] | None = None, titles=("Original", "Dithered")) -> None:
    """
    Top row: both images with a red square. Bottom row: that square zoomed in.
    region = (x, y, size): top-left corner and side length in pixels. None = a square in the centre. Both images must be the same size.
    """
    if original.size != dithered.size: # We're not interested in pixelized images
        raise ValueError(f"Images must be the same size, got {original.size} and {dithered.size}.")

    w, h = original.size
    if region is None:
        size = max(8, min(w, h) // 10)
        region = ((w - size) // 2, (h - size) // 2, size)
    x, y, size = region
    box = (x, y, x + size, y + size)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), squeeze=False)
    for col, (img, title) in enumerate(zip((original, dithered), titles)):
        gray = img.mode in ("L", "1")
        kwargs = dict(cmap="gray", vmin=0, vmax=255) if gray else {}

        ax = axes[0][col] # full image + red square
        ax.imshow(img, interpolation="antialiased", **kwargs)
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), size, size,
                               edgecolor="red", facecolor="none", linewidth=2))
        ax.set_title(f"{title}")#({w}x{h})")
        ax.axis("off")

        ax = axes[1][col] # zoomed crop
        ax.imshow(img.crop(box), interpolation="nearest", **kwargs)
        #ax.set_title(f"Zoom: {size}x{size} px")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("red"); spine.set_linewidth(3)

    plt.tight_layout()
    plt.show()

# --- Helper function(s) ---
def _to_rgb(img: Image.Image) -> Image.Image:
    """ Convert any mode to RGB, flattening transparency onto white """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")

if __name__ == "__main__":
    images = list_images()
    print("Images found:")
    for p in images:
        print(f" {str(p.name)}")
    print()

    target = images[5] if images else None # Attempt to load image
    if not target:
        print("ERROR: No images found.")
        exit(1)

    image = load_image(target)
    print(f"Successfully loaded {target}: {image.size[0]}x{image.size[1]}, mode={image.mode}")
    #show(image, titles=[str(Path(target).name)])

    print("\nShowing first 3 images:")
    first_three = images[:3]
    loaded = [load_image(p) for p in first_three]
    show(*loaded, titles=[p.name for p in first_three])
    print("\nSuccessfully displayed first 3 images")

    print("\nShowing 3 random images:")
    picks = random.choices(images, k=3) # 3 random picks, duplicates allowed
    loaded = [load_image(p) for p in picks]
    show(*loaded, titles=[p.name for p in picks])
    print("\nSuccessfully displayed 3 random images")