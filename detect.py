"""
Inference module: detect pineapple cup and map to 3x3 grid cell.

Usage (CLI):
    python detect.py --weights runs/train/pineapple_cup_v1/weights/best.pt \
                     --image sample_data/top_left.png

Usage (programmatic):
    from detect import Detector
    det = Detector("runs/train/pineapple_cup_v1/weights/best.pt")
    result = det.detect("sample_data/top_left.png")
    # {"has_cup": 1, "cell_id": 0}

Output format:
    {"has_cup": 0 or 1, "cell_id": null or int 0-8}

Cell ID layout (row-major, 0-indexed):
    | 0 | 1 | 2 |
    | 3 | 4 | 5 |
    | 6 | 7 | 8 |
"""

import argparse
import json
from bisect import bisect
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Grid boundary calibration
#
# Source images are 2016 x 2118 px.  These normalised boundaries define the
# 3x3 shelf grid and must be calibrated against the actual shelf grid lines.
# Values below are approximate — run annotate.py to verify visually and adjust.
#
# CALIBRATION INSTRUCTIONS:
#   1. Open a sample image in an image editor or run:
#        python annotate.py --image sample_data/top_left.png --label annotations/top_left.txt
#   2. Identify the pixel x-coordinates of the two vertical dividers and the
#      two horizontal dividers of the shelf.
#   3. Divide by image width (2016) for GRID_COLS_NORM and by image height (2118)
#      for GRID_ROWS_NORM.
#   4. Update the constants below.
# ---------------------------------------------------------------------------
GRID_COLS_NORM = [0.0, 0.3914, 0.6741, 1.0]  # calibrated from mean-image brightness peaks
GRID_ROWS_NORM = [0.0, 0.3687, 0.6242, 1.0]


def pixel_to_cell(cx_norm: float, cy_norm: float) -> int:
    """Map normalised (cx, cy) ∈ [0,1]² to cell_id 0–8."""
    col = bisect(GRID_COLS_NORM, cx_norm) - 1
    row = bisect(GRID_ROWS_NORM, cy_norm) - 1
    col = max(0, min(2, col))
    row = max(0, min(2, row))
    return row * 3 + col


class Detector:
    def __init__(self, weights: str, conf_threshold: float = 0.5):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold

    def detect(self, image_path: str) -> dict:
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False,
        )
        result = results[0]

        if len(result.boxes) == 0:
            return {"has_cup": 0, "cell_id": None}

        # Take the highest-confidence detection
        boxes = result.boxes
        best_idx = int(boxes.conf.argmax())
        box = boxes.xyxyn[best_idx]  # normalised [x1, y1, x2, y2]
        x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()

        cx_norm = (x1 + x2) / 2
        cy_norm = (y1 + y2) / 2
        cell_id = pixel_to_cell(cx_norm, cy_norm)

        return {"has_cup": 1, "cell_id": cell_id}

    def detect_with_viz(self, image_path: str) -> tuple[dict, np.ndarray]:
        """Returns detection result and annotated image (BGR numpy array)."""
        result_dict = self.detect(image_path)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        # Draw grid lines
        for xn in GRID_COLS_NORM[1:-1]:
            cv2.line(img, (int(xn * w), 0), (int(xn * w), h), (180, 180, 180), 3)
        for yn in GRID_ROWS_NORM[1:-1]:
            cv2.line(img, (0, int(yn * h)), (w, int(yn * h)), (180, 180, 180), 3)

        # Draw cell IDs
        for row in range(3):
            for col in range(3):
                cx = int(((GRID_COLS_NORM[col] + GRID_COLS_NORM[col + 1]) / 2) * w)
                cy = int(((GRID_ROWS_NORM[row] + GRID_ROWS_NORM[row + 1]) / 2) * h)
                cell_id = row * 3 + col
                color = (0, 200, 80) if result_dict.get("cell_id") == cell_id else (140, 140, 140)
                cv2.putText(img, str(cell_id), (cx - 25, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

        # Draw detection box if present
        if result_dict["has_cup"]:
            results = self.model.predict(source=image_path, conf=self.conf_threshold, verbose=False)
            boxes = results[0].boxes
            best_idx = int(boxes.conf.argmax())
            box = boxes.xyxy[best_idx]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 80), 4)
            label = f"cell {result_dict['cell_id']}  conf:{boxes.conf[best_idx]:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 12, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 80), 3)

        status = f"has_cup={result_dict['has_cup']}  cell_id={result_dict['cell_id']}"
        cv2.putText(img, status, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 255), 3)

        return result_dict, img


def main():
    parser = argparse.ArgumentParser(description="Detect pineapple cup and map to grid cell")
    parser.add_argument("--weights", required=True, help="Path to trained YOLOv8 .pt weights")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--viz", action="store_true", help="Save annotated image as <stem>_result.png")
    args = parser.parse_args()

    detector = Detector(args.weights, conf_threshold=args.conf)

    if args.viz:
        result, annotated = detector.detect_with_viz(args.image)
        scale = 0.35
        small = cv2.resize(annotated, (int(annotated.shape[1] * scale), int(annotated.shape[0] * scale)))
        out_path = Path(args.image).stem + "_result.png"
        cv2.imwrite(out_path, small)
        print(f"Annotated image saved to {out_path}")
    else:
        result = detector.detect(args.image)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
