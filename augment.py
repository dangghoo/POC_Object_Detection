"""
Offline augmentation script.

Usage:
    python augment.py --src_images sample_data/ --src_labels annotations/ \
                      --out_dir dataset/ --n_aug 25 --val_split 0.3

Reads YOLO .txt annotations from --src_labels (same stem as image files).
Generates augmented images + recalculated labels into dataset/images/{train,val}
and dataset/labels/{train,val}.

Missing.png (empty label file) is included as a negative example.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


# Augmentation pipeline — no flips/large rotations (fixed camera, asymmetric shelf)
def build_transform():
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.20, p=0.8),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.7),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=3,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.5,
        ),
    )


def read_yolo_label(label_path):
    """Returns list of (class_id, x_c, y_c, w, h) tuples, or [] if file is empty/missing."""
    if not label_path.exists():
        return []
    lines = label_path.read_text().strip().splitlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls, xc, yc, w, h))
    return boxes


def write_yolo_label(label_path, boxes):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cls, xc, yc, w, h in boxes]
    label_path.write_text("\n".join(lines))


def augment_and_save(image_path, label_path, out_img_dir, out_lbl_dir, n_aug, transform, stem_prefix=""):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = read_yolo_label(label_path)

    # Save original (copy as-is)
    orig_stem = stem_prefix + image_path.stem + "_orig"
    out_img = out_img_dir / (orig_stem + ".png")
    out_lbl = out_lbl_dir / (orig_stem + ".txt")
    shutil.copy(image_path, out_img)
    write_yolo_label(out_lbl, boxes)

    # Generate augmented variants
    for i in range(n_aug):
        if boxes:
            yolo_boxes = [(xc, yc, w, h) for _, xc, yc, w, h in boxes]
            class_labels = [cls for cls, *_ in boxes]
            try:
                result = transform(image=img_rgb, bboxes=yolo_boxes, class_labels=class_labels)
            except Exception:
                continue
            aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            aug_boxes = [
                (cls, xc, yc, w, h)
                for cls, (xc, yc, w, h) in zip(result["class_labels"], result["bboxes"])
            ]
        else:
            # Negative image — augment without bbox logic
            result = transform(image=img_rgb, bboxes=[], class_labels=[])
            aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            aug_boxes = []

        aug_stem = stem_prefix + image_path.stem + f"_aug{i:03d}"
        cv2.imwrite(str(out_img_dir / (aug_stem + ".png")), aug_img)
        write_yolo_label(out_lbl_dir / (aug_stem + ".txt"), aug_boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_images", default="sample_data", help="Directory with source images")
    parser.add_argument("--src_labels", default="annotations", help="Directory with YOLO .txt labels")
    parser.add_argument("--out_dir", default="dataset", help="Output dataset directory")
    parser.add_argument("--n_aug", type=int, default=25, help="Augmented variants per image")
    parser.add_argument("--val_split", type=float, default=0.3, help="Fraction of images for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    out_dir = Path(args.out_dir)

    image_files = sorted(src_images.glob("*.png")) + sorted(src_images.glob("*.jpg"))
    if not image_files:
        print(f"No images found in {src_images}")
        return

    # Build label lookup — handles UUID-prefixed names exported by Label Studio
    # e.g. "ef2e3205-top_left.txt" matches image stem "top_left"
    label_lookup = {}
    for txt in src_labels.glob("*.txt"):
        stem = txt.stem.split("-", 1)[-1] if "-" in txt.stem else txt.stem
        label_lookup[stem] = txt

    # Split into train/val before augmentation
    random.shuffle(image_files)
    n_val = max(1, int(len(image_files) * args.val_split))
    val_files = set(image_files[:n_val])
    train_files = set(image_files[n_val:])

    transform = build_transform()

    for split, files in [("train", train_files), ("val", val_files)]:
        out_img_dir = out_dir / "images" / split
        out_lbl_dir = out_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            lbl_path = label_lookup.get(img_path.stem, src_labels / (img_path.stem + ".txt"))
            n = args.n_aug if split == "train" else max(3, args.n_aug // 5)
            augment_and_save(img_path, lbl_path, out_img_dir, out_lbl_dir, n, transform)
            print(f"[{split}] {img_path.name} → {n + 1} images")

    print(f"\nDone. Dataset written to {out_dir}/")
    _count(out_dir)


def _count(out_dir):
    for split in ("train", "val"):
        imgs = list((out_dir / "images" / split).glob("*.png"))
        print(f"  {split}: {len(imgs)} images")


if __name__ == "__main__":
    main()
