"""
Annotation preview: draws the rack grid + YOLO bounding boxes on images.

Grid geometry is loaded from a JSON file produced by draw_rack_grid_interactive.py
and scaled proportionally to each target image.

Usage:
    # Batch preview (save to out_dir, no display)
    uv run python src/annotate.py \
        --src_images sample_data_single/ \
        --src_labels annotations/ \
        --grid sample_data_single/rack_grids.json \
        --out_dir annotation_preview/ \
        --no_show

    # Single image
    uv run python src/annotate.py \
        --image sample_data_single/WIN_20260421_13_37_56_Pro.jpg \
        --grid sample_data_single/rack_grids.json

Annotation format expected in each .txt (YOLO normalised):
    <class_id> <x_center> <y_center> <width> <height>
"""

import argparse
import json
from pathlib import Path

import cv2


DEFAULT_GRID = "sample_data_single/rack_grids.json"


# ── grid helpers ─────────────────────────────────────────────────────────────

def load_grid(path):
    d = json.loads(Path(path).read_text())
    return (d["vlines"], d["hlines"],
            d["source_size"]["width"], d["source_size"]["height"],
            d["rows"], d["cols"])


def scale_lines(vlines, hlines, src_w, src_h, dst_w, dst_h):
    sx, sy = dst_w / src_w, dst_h / src_h

    def sv(vl):
        return {"tx": vl["tx"]*sx, "ty": vl["ty"]*sy,
                "bx": vl["bx"]*sx, "by": vl["by"]*sy}

    def sh(hl):
        return {"lx": hl["lx"]*sx, "ly": hl["ly"]*sy,
                "rx": hl["rx"]*sx, "ry": hl["ry"]*sy}

    return [sv(v) for v in vlines], [sh(h) for h in hlines]


def _isect(hl, vl):
    ax, ay, bx, by = hl["lx"], hl["ly"], hl["rx"], hl["ry"]
    cx, cy, dx, dy = vl["tx"], vl["ty"], vl["bx"], vl["by"]
    d = (ax-bx)*(cy-dy) - (ay-by)*(cx-dx)
    if abs(d) < 1e-9:
        return (ax+bx)/2, (ay+by)/2
    t = ((ax-cx)*(cy-dy) - (ay-cy)*(cx-dx)) / d
    return ax + t*(bx-ax), ay + t*(by-ay)


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_grid_lines(img, vlines, hlines):
    for vl in vlines:
        cv2.line(img, (int(round(vl["tx"])), int(round(vl["ty"]))),
                      (int(round(vl["bx"])), int(round(vl["by"]))),
                      (80, 80, 255), 2)
    for hl in hlines:
        cv2.line(img, (int(round(hl["lx"])), int(round(hl["ly"]))),
                      (int(round(hl["rx"])), int(round(hl["ry"]))),
                      (80, 80, 255), 2)


def draw_cell_labels(img, vlines, hlines, rows, cols):
    grid = [[_isect(hlines[r], vlines[c])
             for c in range(cols + 1)]
            for r in range(rows + 1)]

    for r in range(rows):
        for c in range(cols):
            cx = int((grid[r][c][0]+grid[r][c+1][0]+
                      grid[r+1][c][0]+grid[r+1][c+1][0])/4)
            cy = int((grid[r][c][1]+grid[r][c+1][1]+
                      grid[r+1][c][1]+grid[r+1][c+1][1])/4)
            label = f"({r},{c})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(img, label, (cx-tw//2, cy+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(img, label, (cx-tw//2, cy+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)


def draw_boxes(img, label_path: Path):
    if not label_path.exists():
        return
    h, w = img.shape[:2]
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 80), 3)
        cv2.putText(img, f"cls:{cls}", (x1, max(y1-8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 80), 2)


# ── core ─────────────────────────────────────────────────────────────────────

def process(image_path: Path, label_path: Path, grid_path: str,
            show: bool, save_path: Path | None):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  SKIP (cannot read): {image_path}")
        return

    vlines_ref, hlines_ref, src_w, src_h, rows, cols = load_grid(grid_path)
    h, w = img.shape[:2]
    vlines, hlines = scale_lines(vlines_ref, hlines_ref, src_w, src_h, w, h)

    out = img.copy()
    draw_grid_lines(out, vlines, hlines)
    draw_cell_labels(out, vlines, hlines, rows, cols)
    draw_boxes(out, label_path)

    scale = min(1.0, 1200 / w)
    display = cv2.resize(out, (int(w*scale), int(h*scale)))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), display)
        print(f"  Saved → {save_path}")

    if show:
        cv2.imshow(image_path.name, display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise rack grid + YOLO annotations")
    parser.add_argument("--image",       help="Single image path")
    parser.add_argument("--label",       help="Single YOLO .txt label path")
    parser.add_argument("--src_images",  help="Directory of images for batch preview")
    parser.add_argument("--src_labels",  help="Directory of YOLO .txt labels")
    parser.add_argument("--grid",        default=DEFAULT_GRID, help="Path to rack_grids.json")
    parser.add_argument("--out_dir",     default="annotation_preview")
    parser.add_argument("--no_show",     action="store_true")
    args = parser.parse_args()

    if args.image:
        img_path = Path(args.image)
        lbl_path = Path(args.label) if args.label else Path("annotations") / (img_path.stem + ".txt")
        save_path = Path(args.out_dir) / (img_path.stem + "_preview.jpg") if args.no_show else None
        process(img_path, lbl_path, args.grid, show=not args.no_show, save_path=save_path)

    elif args.src_images:
        src_images = Path(args.src_images)
        src_labels = Path(args.src_labels) if args.src_labels else Path("annotations")
        out_dir = Path(args.out_dir)

        image_files = sorted(src_images.glob("*.jpg")) + sorted(src_images.glob("*.png"))
        image_files = [f for f in image_files if "grid" not in f.stem]
        print(f"Found {len(image_files)} images in {src_images}")

        label_lookup = {}
        if src_labels.exists():
            for txt in src_labels.glob("*.txt"):
                stem = txt.stem.split("-", 1)[-1] if "-" in txt.stem else txt.stem
                label_lookup[stem] = txt

        for img_path in image_files:
            lbl_path = label_lookup.get(img_path.stem, src_labels / (img_path.stem + ".txt"))
            save_path = out_dir / (img_path.stem + "_preview.jpg")
            process(img_path, lbl_path, args.grid, show=not args.no_show, save_path=save_path)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
