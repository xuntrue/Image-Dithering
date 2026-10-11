import random

from PIL import Image

def pixelize(img: Image.Image, max_size: int = 160) -> Image.Image:
    """ Shrink an image so its longest side is at most `max_size` pixels  """
    w, h = img.size
    scale = max_size / max(w, h) # Calc scale factor to shrink img
    if scale >= 1:
        return img.copy()
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)

if __name__ == "__main__":
    import ImageLoader

    wide = Image.new("RGB", (1000, 500))
    tall = Image.new("RGB", (500, 1000))
    tiny = Image.new("RGB", (50, 30))
    sliver = Image.new("RGB", (1000, 1))

    assert pixelize(wide, 100).size == (100, 50), "wide image should keep 2:1 ratio"
    assert pixelize(tall, 100).size == (50, 100), "tall image should keep 1:2 ratio"
    assert pixelize(tiny, 100).size == (50, 30), "small image should not be enlarged"
    assert pixelize(tiny, 100) is not tiny, "should return a copy, not the original"
    assert pixelize(sliver, 100).size == (100, 1), "height should never drop below 1px"
    print("All size tests passed")

    images = ImageLoader.list_images()
    if not images:
        print("ERROR: No images found.")
        exit(1)

    path = random.choice(images)
    original = ImageLoader.load_image(path)
    sizes = [160, 80, 40]
    results = [pixelize(original, s) for s in sizes]

    print(f"\nPixelizing {path.name}: {original.size[0]}x{original.size[1]}")
    for s, r in zip(sizes, results):
        print(f" max_size={s} -> {r.size[0]}x{r.size[1]}")

    titles = [f"Original {original.size[0]}x{original.size[1]}"]
    titles += [f"max_size={s} ({r.size[0]}x{r.size[1]})" for s, r in zip(sizes, results)]
    ImageLoader.show(original, *results, titles=titles)