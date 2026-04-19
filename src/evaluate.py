"""
End-to-end evaluation on the 10 sample images.

Usage:
    python evaluate.py --weights runs/train/pineapple_cup_v1/weights/best.pt
    python evaluate.py --weights runs/train/pineapple_cup_v1/weights/best.pt \
                       --data sample_data/ --save_viz

Expected results (ground truth derived from file names):
    top_left.png      → {has_cup: 1, cell_id: 0}
    top_center.png    → {has_cup: 1, cell_id: 1}
    top_right.png     → {has_cup: 1, cell_id: 2}
    middle_left.png   → {has_cup: 1, cell_id: 3}
    middle_center.png → {has_cup: 1, cell_id: 4}
    middle_right.png  → {has_cup: 1, cell_id: 5}
    bottom_left.png   → {has_cup: 1, cell_id: 6}
    bottom_center.png → {has_cup: 1, cell_id: 7}
    bottom_right.png  → {has_cup: 1, cell_id: 8}
    missing.png       → {has_cup: 0, cell_id: null}
"""

import argparse
import json
from pathlib import Path

import cv2

from detect import Detector


# Ground-truth mapping derived from file names
GROUND_TRUTH = {
    "top_left":      {"has_cup": 1, "cell_id": 0},
    "top_center":    {"has_cup": 1, "cell_id": 1},
    "top_right":     {"has_cup": 1, "cell_id": 2},
    "middle_left":   {"has_cup": 1, "cell_id": 3},
    "middle_center": {"has_cup": 1, "cell_id": 4},
    "middle_right":  {"has_cup": 1, "cell_id": 5},
    "bottom_left":   {"has_cup": 1, "cell_id": 6},
    "bottom_center": {"has_cup": 1, "cell_id": 7},
    "bottom_right":  {"has_cup": 1, "cell_id": 8},
    "missing":       {"has_cup": 0, "cell_id": None},
}


def evaluate(weights: str, data_dir: str = "sample_data", conf: float = 0.5, save_viz: bool = False):
    detector = Detector(weights, conf_threshold=conf)
    data_path = Path(data_dir)

    image_files = sorted(data_path.glob("*.png")) + sorted(data_path.glob("*.jpg"))
    if not image_files:
        print(f"No images found in {data_dir}")
        return

    viz_dir = Path("eval_viz")
    if save_viz:
        viz_dir.mkdir(exist_ok=True)

    total = 0
    presence_correct = 0
    cell_correct = 0
    cell_total = 0  # only images where has_cup=1 in ground truth

    print(f"\n{'Image':<22} {'GT':^18} {'Pred':^18} {'Presence':^10} {'Cell':^8}")
    print("-" * 80)

    for img_path in image_files:
        stem = img_path.stem
        gt = GROUND_TRUTH.get(stem)
        if gt is None:
            print(f"{stem:<22} [no ground truth — skipping]")
            continue

        if save_viz:
            pred, annotated = detector.detect_with_viz(str(img_path))
            scale = 0.35
            small = cv2.resize(annotated, (int(annotated.shape[1] * scale), int(annotated.shape[0] * scale)))
            cv2.imwrite(str(viz_dir / f"{stem}_eval.png"), small)
        else:
            pred = detector.detect(str(img_path))

        total += 1

        pres_ok = pred["has_cup"] == gt["has_cup"]
        if pres_ok:
            presence_correct += 1

        cell_ok_str = "N/A"
        if gt["has_cup"] == 1:
            cell_total += 1
            cell_ok = pred["cell_id"] == gt["cell_id"]
            if cell_ok:
                cell_correct += 1
            cell_ok_str = "✓" if cell_ok else f"✗ (got {pred['cell_id']})"

        gt_str = f"has={gt['has_cup']} cell={gt['cell_id']}"
        pred_str = f"has={pred['has_cup']} cell={pred['cell_id']}"
        pres_str = "✓" if pres_ok else "✗"

        print(f"{stem:<22} {gt_str:^18} {pred_str:^18} {pres_str:^10} {cell_ok_str:^8}")

    print("-" * 80)
    print(f"\nPresence accuracy : {presence_correct}/{total} ({100*presence_correct/total:.1f}%)")
    if cell_total > 0:
        print(f"Cell accuracy     : {cell_correct}/{cell_total} ({100*cell_correct/cell_total:.1f}%)")
    end2end = sum(
        1 for img_path in image_files
        if GROUND_TRUTH.get(img_path.stem) is not None
        and _end2end_correct(detector, img_path, GROUND_TRUTH[img_path.stem])
    )
    print(f"End-to-end accuracy: {end2end}/{total} ({100*end2end/total:.1f}%)")

    if save_viz:
        print(f"\nAnnotated images saved to {viz_dir}/")

    return {
        "presence_accuracy": presence_correct / total if total else 0,
        "cell_accuracy": cell_correct / cell_total if cell_total else 0,
        "end_to_end_accuracy": end2end / total if total else 0,
    }


def _end2end_correct(detector: Detector, img_path: Path, gt: dict) -> bool:
    pred = detector.detect(str(img_path))
    return pred["has_cup"] == gt["has_cup"] and pred["cell_id"] == gt["cell_id"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate pineapple cup detector on sample images")
    parser.add_argument("--weights", required=True, help="Path to trained YOLOv8 .pt weights")
    parser.add_argument("--data", default="sample_data", help="Directory with sample images")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--save_viz", action="store_true", help="Save annotated result images to eval_viz/")
    args = parser.parse_args()

    metrics = evaluate(args.weights, args.data, args.conf, args.save_viz)
    print(f"\nSummary JSON:\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
