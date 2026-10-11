# Image Dithering

Turns an image into a dithered version in two steps:

1. **penalize.py**: shrinks the image (default max side 160px) so the dithering pattern is visible.
2. **dither.py**: quantises it using one of several algorithms.

## Setup
```
pip install -r requirements.txt
```

## Usage
```
python main.py photo.jpg                          # Floyd-Steinberg, black & white
python main.py photo.jpg -m atkinson -s 200       # Atkinson at 200px
python main.py photo.jpg -m bayer --bayer-size 8  # ordered dithering
python main.py photo.jpg --color -l 2             # 8-colour RGB dither
python main.py photo.jpg -m threshold             # no dithering (baseline)
```

Methods: `threshold`, `bayer`, `floyd-steinberg`, `atkinson`, `jarvis`, `sierra`.

Output is enlarged with nearest-neighbour (`--upscale`, default 4) so the
individual dithered pixels stay crisp.
