"""
Fine-tune YOLOv8-nano on the pineapple cup dataset.

Usage:
    python train.py
    python train.py --model yolov8s.pt --epochs 150 --batch 16

Prerequisites:
    - dataset/ must be populated by running augment.py first
    - dataset.yaml must exist in the project root
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

# Resolve project root so YOLO always writes to <project_root>/runs/train
# regardless of which directory the script is invoked from.
PROJECT_ROOT = Path(__file__).parent.resolve()


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for pineapple cup detection")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model to fine-tune")
    parser.add_argument("--data", default="dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "train"))
    parser.add_argument("--name", default="pineapple_cup_v1")
    parser.add_argument("--device", default="", help="cuda device or 'cpu' (auto-detect if empty)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    data_yaml = PROJECT_ROOT / args.data
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {data_yaml}. Run augment.py first.")

    dataset_dir = PROJECT_ROOT / "dataset"
    if not (dataset_dir / "images" / "train").exists():
        raise FileNotFoundError("dataset/images/train/ not found. Run augment.py first.")

    # Change to project root so all relative paths inside YOLO resolve correctly
    os.chdir(PROJECT_ROOT)

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        weight_decay=0.0005,
        augment=False,       # augmentation done offline in augment.py
        patience=args.patience,
        save_period=10,
        project=args.project,
        name=args.name,
        device=args.device if args.device else None,
        verbose=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights: {best_weights}")
    print(f"\nTo run inference:")
    print(f"  python detect.py --weights {best_weights} --image sample_data/top_left.png")
    print(f"\nTo evaluate on all sample images:")
    print(f"  python evaluate.py --weights {best_weights}")

    return results


if __name__ == "__main__":
    main()
