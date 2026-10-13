import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image

def _rgb_array(img: Image.Image) -> np.ndarray:
    """ Any PIL image -> (H, W, 3) uint8 array """
    return np.asarray(img.convert("RGB"))

def count_unique_colours(img: Image.Image) -> int:
    """ Number of distinct (R, G, B) colours that actually appear in the image """
    arr = _rgb_array(img).astype(np.uint32)
    packed = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]  # one int per colour
    return int(np.unique(packed).size)

def is_grayscale(img: Image.Image) -> bool:
    """ True if R == G == B for every pixel """
    arr = _rgb_array(img)
    return bool(np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]))

def channel_histograms(img: Image.Image) -> dict[str, np.ndarray]:
    """ Percent of pixels at each intensity 0-255, for each of R, G and B """
    arr = _rgb_array(img)
    total = arr.shape[0] * arr.shape[1]
    return {
        name: np.bincount(arr[..., i].ravel(), minlength=256) / total * 100 for i, name in enumerate("RGB")
    }

def describe(img: Image.Image) -> dict:
    """ Summary numbers for one image. Brightness uses the standard luma formula."""
    gray = np.asarray(img.convert("L"))
    return {
        "width": img.size[0],
        "height": img.size[1],
        "pixels": img.size[0] * img.size[1],
        "unique_colours": count_unique_colours(img),
        "unique_gray_levels": int(np.unique(gray).size),
        "mean_brightness": float(gray.mean()),
        "std_brightness": float(gray.std()),
        "min_brightness": int(gray.min()),
        "max_brightness": int(gray.max()),
    }

def print_stats(*images: Image.Image, titles: list[str] | None = None) -> None:
    """ Print a comparison table of summary numbers """
    titles = titles or [f"Image {i + 1}" for i in range(len(images))]
    print(f"{'Stage':<28}{'Size':>11}{'Colours':>11}{'Gray lvls':>11}{'Mean':>8}{'Std':>8}{'Min':>6}{'Max':>6}")
    for img, t in zip(images, titles):
        s = describe(img)
        size = f"{s['width']}x{s['height']}"
        print(f"{t:<28}{size:>11}{s['unique_colours']:>11,}{s['unique_gray_levels']:>11}"
              f"{s['mean_brightness']:>8.2f}{s['std_brightness']:>8.2f}"
              f"{s['min_brightness']:>6}{s['max_brightness']:>6}")

def plot_analysis(*images: Image.Image, titles: list[str] | None = None) -> None:
    """
    One column per image: the picture on top, its intensity histogram below.
    Grayscale images get a single gray histogram; 
    Colour images get one line per channel.
    """
    n = len(images)
    titles = titles or [f"Image {i + 1}" for i in range(n)]
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7), squeeze=False)
    x = np.arange(256)

    for col, (img, title) in enumerate(zip(images, titles)):
        stats = describe(img)
        gray = is_grayscale(img)

        # top row: the image
        ax_img = axes[0][col]
        ax_img.imshow(img, cmap="gray" if gray else None, interpolation="nearest")
        ax_img.set_title(f"{title}\n{stats['width']}x{stats['height']}, "
                         f"{stats['unique_colours']:,} colours", fontsize=10)
        ax_img.axis("off")

        # bottom row: histogram
        ax = axes[1][col]
        hists = channel_histograms(img)
        few_values = max(np.count_nonzero(h) for h in hists.values()) <= 32
        if gray:
            if few_values:
                ax.bar(x, hists["R"], width=4, color="gray")
            else:
                ax.fill_between(x, hists["R"], step="mid", color="gray", alpha=0.8)
        else:
            for k, (name, colour) in enumerate(zip("RGB", ("red", "green", "blue"))):
                if few_values:
                    ax.bar(x + (k - 1) * 4, hists[name], width=4, color=colour, label=name)
                else:
                    ax.plot(x, hists[name], color=colour, alpha=0.7, linewidth=1, label=name)
            ax.legend(loc="center right" if few_values else "upper right", fontsize=8)
        ax.axvline(stats["mean_brightness"], color="black", linestyle="--", linewidth=1)
        ax.set_xlim(-10, 265)
        ax.set_xlabel("Intensity (0 = black, 255 = white)")
        ax.set_ylabel("% of pixels")
        ax.set_title(f"mean brightness {stats['mean_brightness']:.0f}  (dashed)", fontsize=9)

        if True:  # toggle gridlines on/off
            ax.set_axisbelow(True)    # draw gridlines behind the bars and lines
            ax.minorticks_on()        # minor gridlines need minor ticks to exist
            ax.grid(which="major", axis="both", linestyle="-", alpha=0.5)
            ax.grid(which="minor", axis="both", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_colour_counts(*images: Image.Image, titles: list[str] | None = None) -> None:
    """ Bar chart of unique colours per image (log scale: the counts differ hugely) """
    titles = titles or [f"Image {i + 1}" for i in range(len(images))]
    counts = [count_unique_colours(img) for img in images]

    fig, ax = plt.subplots(figsize=(1.8 * len(images) + 3, 5))
    bars = ax.bar(range(len(images)), counts, color="steelblue")
    ax.set_yscale("log")
    ax.set_xticks(range(len(images)))
    ax.set_xticklabels(titles, rotation=20, ha="right")
    ax.set_ylabel("Unique colours (log scale)")
    ax.set_title("Colours remaining at each stage")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(top=max(counts) * 3)
    if True:  # toggle gridlines on/off
        ax.set_axisbelow(True)
        ax.grid(which="major", axis="y", linestyle="-", alpha=0.5)
        ax.grid(which="minor", axis="y", linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare(*images: Image.Image, titles: list[str] | None = None) -> None:
    """Print the stats table, then show the histogram figure and the colour-count chart."""
    print_stats(*images, titles=titles)
    plot_analysis(*images, titles=titles)
    plot_colour_counts(*images, titles=titles)