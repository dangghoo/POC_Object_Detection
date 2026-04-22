"""
Apply a saved rack grid to one or more new images.

The grid was drawn on a reference image (source_size in the JSON).
Each line's endpoints are scaled proportionally to the target image size,
then drawn with labels.

Usage:
    python src/apply_grid.py sample_data/WIN_20260421_13_37_56_Pro_grid.json \
        rack1.jpg rack2.jpg rack3.jpg

Output: <original_name>_grid.jpg  next to each input file.
"""

import json
import sys
from pathlib import Path

import cv2


def _line_isect(ax, ay, bx, by, cx, cy, dx, dy):
    d = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(d) < 1e-9:
        return ((ax + bx) / 2, (ay + by) / 2)
    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / d
    return (ax + t * (bx - ax), ay + t * (by - ay))


def apply_grid(img, vlines, hlines, rows, cols):
    out = img.copy()

    for vl in vlines:
        cv2.line(out,
                 (int(round(vl["tx"])), int(round(vl["ty"]))),
                 (int(round(vl["bx"])), int(round(vl["by"]))),
                 (0, 220, 0), 2)

    for hl in hlines:
        cv2.line(out,
                 (int(round(hl["lx"])), int(round(hl["ly"]))),
                 (int(round(hl["rx"])), int(round(hl["ry"]))),
                 (0, 220, 0), 2)

    grid = [
        [_line_isect(hl["lx"], hl["ly"], hl["rx"], hl["ry"],
                     vl["tx"], vl["ty"], vl["bx"], vl["by"])
         for vl in vlines]
        for hl in hlines
    ]

    for r in range(rows):
        for c in range(cols):
            cx = int((grid[r][c][0] + grid[r][c+1][0] +
                      grid[r+1][c][0] + grid[r+1][c+1][0]) / 4)
            cy = int((grid[r][c][1] + grid[r][c+1][1] +
                      grid[r+1][c][1] + grid[r+1][c+1][1]) / 4)
            label = f"({r},{c})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(out, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(out, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return out


def scale_lines(vlines, hlines, src_w, src_h, dst_w, dst_h):
    sx, sy = dst_w / src_w, dst_h / src_h

    def sv(vl):
        return {
            "tx": vl["tx"] * sx, "ty": vl["ty"] * sy,
            "bx": vl["bx"] * sx, "by": vl["by"] * sy,
        }

    def sh(hl):
        return {
            "lx": hl["lx"] * sx, "ly": hl["ly"] * sy,
            "rx": hl["rx"] * sx, "ry": hl["ry"] * sy,
        }

    return [sv(v) for v in vlines], [sh(h) for h in hlines]


def main(args):
    if len(args) < 2:
        print("Usage: apply_grid.py <grid.json> <image1> [image2 ...]")
        sys.exit(1)

    grid_path = Path(args[0])
    image_paths = [Path(a) for a in args[1:]]

    grid = json.loads(grid_path.read_text())
    src_w = grid["source_size"]["width"]
    src_h = grid["source_size"]["height"]
    rows  = grid["rows"]
    cols  = grid["cols"]
    vlines = grid["vlines"]
    hlines = grid["hlines"]

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP (cannot read): {img_path}")
            continue

        dst_h, dst_w = img.shape[:2]
        scaled_v, scaled_h = scale_lines(vlines, hlines, src_w, src_h, dst_w, dst_h)
        result = apply_grid(img, scaled_v, scaled_h, rows, cols)

        out_path = img_path.with_name(img_path.stem + "_grid" + img_path.suffix)
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
