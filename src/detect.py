"""
Detect objects and map each detection to a rack grid cell.

The grid is loaded from a JSON file produced by draw_rack_grid_interactive.py.
Lines are scaled proportionally when the target image differs in size from the
reference image used to draw the grid.

Usage (CLI):
    python src/detect.py --weights runs/train/.../best.pt \
                         --grid sample_data/WIN_20260421_13_37_56_Pro_grid.json \
                         --image sample_data/top_left.png

    # visualise:
    python src/detect.py ... --viz

Usage (programmatic):
    from detect import Detector
    det = Detector("runs/.../best.pt", "sample_data/..._grid.json")
    result = det.detect("sample_data/top_left.png")
    # {"detections": [{"conf": 0.91, "bbox": [x1,y1,x2,y2], "cell": [2, 1]}]}

Cell coordinates are (row, col), 0-indexed, matching the grid drawn
by draw_rack_grid_interactive.py  (12 rows × 4 cols by default).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ── grid geometry helpers ────────────────────────────────────────────────────

def _x_at_y(vl, py):
    """X coordinate on a vertical line at a given y."""
    ty, by = vl["ty"], vl["by"]
    if abs(by - ty) < 1e-9:
        return (vl["tx"] + vl["bx"]) / 2
    t = (py - ty) / (by - ty)
    return vl["tx"] + t * (vl["bx"] - vl["tx"])


def _y_at_x(hl, px):
    """Y coordinate on a horizontal line at a given x."""
    lx, rx = hl["lx"], hl["rx"]
    if abs(rx - lx) < 1e-9:
        return (hl["ly"] + hl["ry"]) / 2
    t = (px - lx) / (rx - lx)
    return hl["ly"] + t * (hl["ry"] - hl["ly"])


def find_cell(px, py, vlines, hlines):
    """
    Return (row, col) for pixel point (px, py) given the grid lines.
    Clamps to grid boundaries if the point falls outside.
    """
    n_cols = len(vlines) - 1
    n_rows = len(hlines) - 1

    xs = [_x_at_y(vl, py) for vl in vlines]
    col = n_cols - 1
    for c in range(n_cols):
        if px <= xs[c + 1]:
            col = c
            break

    ys = [_y_at_x(hl, px) for hl in hlines]
    row = n_rows - 1
    for r in range(n_rows):
        if py <= ys[r + 1]:
            row = r
            break

    return row, col


def scale_lines(vlines, hlines, src_w, src_h, dst_w, dst_h):
    sx, sy = dst_w / src_w, dst_h / src_h

    def sv(vl):
        return {"tx": vl["tx"]*sx, "ty": vl["ty"]*sy,
                "bx": vl["bx"]*sx, "by": vl["by"]*sy}

    def sh(hl):
        return {"lx": hl["lx"]*sx, "ly": hl["ly"]*sy,
                "rx": hl["rx"]*sx, "ry": hl["ry"]*sy}

    return [sv(v) for v in vlines], [sh(h) for h in hlines]


def load_grid(grid_json_path):
    data = json.loads(Path(grid_json_path).read_text())
    return (data["vlines"], data["hlines"],
            data["source_size"]["width"], data["source_size"]["height"],
            data["rows"], data["cols"])


# ── detector ─────────────────────────────────────────────────────────────────

class Detector:
    def __init__(self, weights: str, grid_json: str, conf_threshold: float = 0.5):
        self.model = YOLO(weights)
        self.conf = conf_threshold
        self.vlines_ref, self.hlines_ref, self.src_w, self.src_h, \
            self.rows, self.cols = load_grid(grid_json)

    def _grid_for(self, img_w, img_h):
        return scale_lines(self.vlines_ref, self.hlines_ref,
                           self.src_w, self.src_h, img_w, img_h)

    def detect(self, image_path: str) -> dict:
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        vlines, hlines = self._grid_for(w, h)

        results = self.model.predict(source=image_path, conf=self.conf, verbose=False)
        boxes = results[0].boxes

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            px, py = (x1 + x2) / 2, (y1 + y2) / 2
            row, col = find_cell(px, py, vlines, hlines)
            detections.append({
                "conf": round(float(boxes.conf[i]), 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "cell": [row, col],
            })

        detections.sort(key=lambda d: d["conf"], reverse=True)
        return {"detections": detections}

    def detect_flat(self, image_path: str) -> dict:
        """Return {has_bowl, cell_id} using the highest-confidence detection."""
        result = self.detect(image_path)
        if not result["detections"]:
            return {"has_bowl": 0, "cell_id": None}
        best = result["detections"][0]
        row, col = best["cell"]
        cell_id = row * self.cols + col
        return {"has_bowl": 1, "cell_id": cell_id}

    def detect_with_viz(self, image_path: str) -> tuple[dict, np.ndarray]:
        result = self.detect(image_path)
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        vlines, hlines = self._grid_for(w, h)

        # Draw grid lines
        for vl in vlines:
            cv2.line(img, (int(vl["tx"]), int(vl["ty"])),
                          (int(vl["bx"]), int(vl["by"])), (0, 220, 0), 2)
        for hl in hlines:
            cv2.line(img, (int(hl["lx"]), int(hl["ly"])),
                          (int(hl["rx"]), int(hl["ry"])), (0, 220, 0), 2)

        # Cell labels at each cell centre
        def _isect(hl, vl):
            ax, ay, bx, by = hl["lx"], hl["ly"], hl["rx"], hl["ry"]
            cx, cy, dx, dy = vl["tx"], vl["ty"], vl["bx"], vl["by"]
            d = (ax-bx)*(cy-dy) - (ay-by)*(cx-dx)
            if abs(d) < 1e-9:
                return (ax+bx)/2, (ay+by)/2
            t = ((ax-cx)*(cy-dy) - (ay-cy)*(cx-dx)) / d
            return ax + t*(bx-ax), ay + t*(by-ay)

        grid = [[_isect(hlines[r], vlines[c])
                 for c in range(self.cols + 1)]
                for r in range(self.rows + 1)]

        hit_cells = {(d["cell"][0], d["cell"][1]) for d in result["detections"]}

        for r in range(self.rows):
            for c in range(self.cols):
                cx = int((grid[r][c][0]+grid[r][c+1][0]+
                          grid[r+1][c][0]+grid[r+1][c+1][0])/4)
                cy = int((grid[r][c][1]+grid[r][c+1][1]+
                          grid[r+1][c][1]+grid[r+1][c+1][1])/4)
                label = f"({r},{c})"
                color = (0, 200, 80) if (r, c) in hit_cells else (255, 255, 255)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.putText(img, label, (cx-tw//2, cy+th//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(img, label, (cx-tw//2, cy+th//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw detection boxes
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            r, c = det["cell"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 80), 3)
            lbl = f"({r},{c}) {det['conf']:.2f}"
            cv2.putText(img, lbl, (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, lbl, (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2)

        return result, img


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect objects and map to rack grid cell")
    parser.add_argument("--weights", required=True, help="Path to YOLOv8 .pt weights")
    parser.add_argument("--grid",    required=True, help="Path to grid JSON from draw_rack_grid_interactive.py")
    parser.add_argument("--image",   required=True, help="Path to input image")
    parser.add_argument("--conf",    type=float, default=0.5)
    parser.add_argument("--viz",     action="store_true", help="Save annotated image as <stem>_result.jpg")
    args = parser.parse_args()

    detector = Detector(args.weights, args.grid, conf_threshold=args.conf)

    if args.viz:
        result, annotated = detector.detect_with_viz(args.image)
        out = Path(args.image).with_name(Path(args.image).stem + "_result.jpg")
        cv2.imwrite(str(out), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Annotated image → {out}")
    else:
        result = detector.detect(args.image)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
