"""
Draw a 4×12 grid overlay on the metallic rack image.

Usage:
    python src/draw_rack_grid.py

Output:
    sample_data/WIN_20260421_13_37_56_Pro_grid.jpg
"""

from pathlib import Path

import cv2
import numpy as np

ROWS = 4
COLS = 12

# Approximate rack bounding box in normalised [0,1] coords (1920×1080 image).
# Excludes the robot arm on the right and background shelving on the left.
RACK = dict(x0=0.26, x1=0.77, y0=0.04, y1=0.93)

LINE_COLOR = (0, 220, 0)   # green
LINE_THICKNESS = 3

LABEL_COLOR = (255, 255, 255)       # white text
LABEL_OUTLINE_COLOR = (0, 0, 0)    # black outline for legibility
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.38
FONT_THICKNESS = 1


def _pixel_boundaries(img_dim: int, norm_start: float, norm_end: float, n: int):
    """Return pixel positions for n+1 evenly spaced grid boundaries."""
    return [int(norm_start * img_dim + (norm_end - norm_start) * img_dim * i / n) for i in range(n + 1)]


def draw_rack_grid(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    xs = _pixel_boundaries(w, RACK["x0"], RACK["x1"], COLS)
    ys = _pixel_boundaries(h, RACK["y0"], RACK["y1"], ROWS)

    out = img.copy()

    # Draw all vertical lines (including outer borders)
    for x in xs:
        cv2.line(out, (x, ys[0]), (x, ys[-1]), LINE_COLOR, LINE_THICKNESS)

    # Draw all horizontal lines (including outer borders)
    for y in ys:
        cv2.line(out, (xs[0], y), (xs[-1], y), LINE_COLOR, LINE_THICKNESS)

    # Label each cell (row, col)
    for r in range(ROWS):
        for c in range(COLS):
            cx = (xs[c] + xs[c + 1]) // 2
            cy = (ys[r] + ys[r + 1]) // 2
            label = f"({r},{c})"
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            tx, ty = cx - tw // 2, cy + th // 2
            # Dark outline
            cv2.putText(out, label, (tx, ty), FONT, FONT_SCALE, LABEL_OUTLINE_COLOR, FONT_THICKNESS + 2)
            # White text
            cv2.putText(out, label, (tx, ty), FONT, FONT_SCALE, LABEL_COLOR, FONT_THICKNESS)

    return out


def main():
    src = Path("sample_data/WIN_20260421_13_37_56_Pro.jpg")
    dst = src.with_name(src.stem + "_grid.jpg")

    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {src}")

    result = draw_rack_grid(img)
    cv2.imwrite(str(dst), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {dst}  ({result.shape[1]}×{result.shape[0]})")


if __name__ == "__main__":
    main()
