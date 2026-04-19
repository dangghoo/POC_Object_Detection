"""
Annotation helper: visualises YOLO bounding boxes drawn over the original images.

Usage:
    # View a single image with its annotation
    python annotate.py --image sample_data/top_left.png \
                       --label annotations/top_left.txt

    # Verify all annotations in a directory (saves overlay images to out_dir)
    python annotate.py --src_images sample_data/ \
                       --src_labels annotations/ \
                       --out_dir annotation_preview/

Annotation format expected in each .txt file (YOLO normalised):
    <class_id> <x_center> <y_center> <width> <height>

Image dimensions: 2016 x 2118 px
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


# Grid boundary constants (in normalised [0,1] coords, calibrated for 2016x2118 images)
# Adjust these after measuring actual shelf grid lines from the images.
GRID_COLS_NORM = [0.0, 0.3914, 0.6741, 1.0]  # calibrated from mean-image brightness peaks
GRID_ROWS_NORM = [0.0, 0.3687, 0.6242, 1.0]

CELL_NAMES = [
    "top_left",    "top_center",    "top_right",
    "middle_left", "middle_center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
]


def draw_grid(img, cols_norm, rows_norm, color=(80, 80, 255), thickness=6):
    h, w = img.shape[:2]
    for xn in cols_norm[1:-1]:
        x = int(xn * w)
        cv2.line(img, (x, 0), (x, h), color, thickness)
    for yn in rows_norm[1:-1]:
        y = int(yn * h)
        cv2.line(img, (0, y), (w, y), color, thickness)
    return img


def draw_boxes(img, label_path, class_colors=None):
    if class_colors is None:
        class_colors = {0: (0, 200, 80)}

    h, w = img.shape[:2]
    if not label_path.exists():
        return img

    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = class_colors.get(cls, (0, 200, 80))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"cls:{cls} ({xc:.2f},{yc:.2f})"
        cv2.putText(img, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


def overlay_cell_labels(img, cols_norm, rows_norm):
    h, w = img.shape[:2]
    for row in range(3):
        for col in range(3):
            cx = int(((cols_norm[col] + cols_norm[col + 1]) / 2) * w)
            cy = int(((rows_norm[row] + rows_norm[row + 1]) / 2) * h)
            cell_id = row * 3 + col
            cv2.putText(img, f"{cell_id}", (cx - 20, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 3)
    return img


def process_single(image_path: Path, label_path: Path, show: bool = True, save_path: Path = None):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot read image: {image_path}")
        return

    display = img.copy()
    draw_grid(display, GRID_COLS_NORM, GRID_ROWS_NORM)
    overlay_cell_labels(display, GRID_COLS_NORM, GRID_ROWS_NORM)
    draw_boxes(display, label_path)

    # Downscale for display (images are 2016x2118)
    scale = 0.35
    small = cv2.resize(display, (int(display.shape[1] * scale), int(display.shape[0] * scale)))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), small)
        print(f"Saved preview: {save_path}")

    if show:
        cv2.imshow(image_path.name, small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualise YOLO annotations on shelf images")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--label", help="Single label path (YOLO .txt)")
    parser.add_argument("--src_images", help="Directory of images for batch preview")
    parser.add_argument("--src_labels", help="Directory of YOLO .txt labels")
    parser.add_argument("--out_dir", default="annotation_preview", help="Directory to save overlay images")
    parser.add_argument("--no_show", action="store_true", help="Don't open display window (save only)")
    args = parser.parse_args()

    if args.image:
        img_path = Path(args.image)
        if args.label:
            lbl_path = Path(args.label)
        else:
            # Search for UUID-prefixed label (e.g. "ef2e3205-top_left.txt")
            lbl_dir = Path("annotations")
            matches = list(lbl_dir.glob(f"*{img_path.stem}.txt"))
            lbl_path = matches[0] if matches else lbl_dir / (img_path.stem + ".txt")
        save_path = Path(args.out_dir) / (img_path.stem + "_preview.png") if args.no_show else None
        process_single(img_path, lbl_path, show=not args.no_show, save_path=save_path)

    elif args.src_images:
        src_images = Path(args.src_images)
        src_labels = Path(args.src_labels) if args.src_labels else Path("annotations")
        out_dir = Path(args.out_dir)

        image_files = sorted(src_images.glob("*.png")) + sorted(src_images.glob("*.jpg"))
        print(f"Found {len(image_files)} images in {src_images}")

        # Build a lookup: image stem → label file (handles UUID-prefixed names from Label Studio)
        label_lookup = {}
        for txt in src_labels.glob("*.txt"):
            # Matches both "top_left.txt" and "ef2e3205-top_left.txt"
            stem = txt.stem.split("-", 1)[-1] if "-" in txt.stem else txt.stem
            label_lookup[stem] = txt

        for img_path in image_files:
            lbl_path = label_lookup.get(img_path.stem, src_labels / (img_path.stem + ".txt"))
            save_path = out_dir / (img_path.stem + "_preview.png")
            process_single(img_path, lbl_path, show=not args.no_show, save_path=save_path)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
