#!/usr/bin/env python3
"""
train_yolo.py
-------------
Runs a single YOLO training cycle via Ultralytics and writes a metrics JSON
summarising the run (final losses, mAP, precision/recall, per-epoch curves,
and the path to the best checkpoint). Designed to be launched as a subprocess
by self_improve.py, one call per cycle.

Usage:
    python train_yolo.py --weights yolov8s.pt --data data/dataset.yaml --epochs 10 --metrics_output metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single YOLO training cycle")
    p.add_argument("--weights", required=True, help="Starting checkpoint, e.g. yolov8s.pt or path/to/best.pt")
    p.add_argument("--data", required=True, help="Path to dataset.yaml")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--project", default="outputs")
    p.add_argument("--name", default="train")
    p.add_argument("--device", default="0")
    p.add_argument("--metrics_output", required=True)

    # Tunable hyperparameters (mirrors ultralytics/cfg/default.yaml)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--warmup_epochs", type=float, default=3.0)
    p.add_argument("--warmup_momentum", type=float, default=0.8)
    p.add_argument("--warmup_bias_lr", type=float, default=0.1)
    p.add_argument("--box", type=float, default=7.5)
    p.add_argument("--cls", type=float, default=0.5)
    p.add_argument("--dfl", type=float, default=1.5)
    p.add_argument("--hsv_h", type=float, default=0.015)
    p.add_argument("--hsv_s", type=float, default=0.7)
    p.add_argument("--hsv_v", type=float, default=0.4)
    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--shear", type=float, default=0.0)
    p.add_argument("--perspective", type=float, default=0.0)
    p.add_argument("--flipud", type=float, default=0.0)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--copy_paste", type=float, default=0.0)

    return p.parse_args()


TRAIN_HYPERPARAM_KEYS = [
    "imgsz", "batch", "optimizer", "patience",
    "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
    "box", "cls", "dfl",
    "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
]


def read_results_csv(csv_path: Path) -> dict[str, list[float]]:
    """Read Ultralytics' per-epoch results.csv into {column: [values...]}, tolerant of header naming."""
    if not csv_path.exists():
        return {}
    curves: dict[str, list[float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                key = key.strip()
                try:
                    curves.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    continue
    return curves


def find_curve(curves: dict[str, list[float]], *substrings: str) -> list[float]:
    for key, values in curves.items():
        if all(s in key for s in substrings):
            return values
    return []


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    train_kwargs: dict[str, Any] = {k: getattr(args, k) for k in TRAIN_HYPERPARAM_KEYS}
    train_kwargs.update(
        data=args.data,
        epochs=args.epochs,
        project=args.project,
        name=args.name,
        device=args.device,
        exist_ok=True,
        verbose=True,
        # Windows: multi-worker DataLoader spawn deadlocks reliably on this dataset size.
        workers=0,
    )

    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    curves = read_results_csv(save_dir / "results.csv")

    box_metrics = results.box if hasattr(results, "box") else None

    metrics = {
        "epochs_trained": args.epochs,
        "map50": float(box_metrics.map50) if box_metrics is not None else None,
        "map50_95": float(box_metrics.map) if box_metrics is not None else None,
        "precision": float(box_metrics.mp) if box_metrics is not None else None,
        "recall": float(box_metrics.mr) if box_metrics is not None else None,
        "train_box_loss_curve": find_curve(curves, "train", "box_loss"),
        "train_cls_loss_curve": find_curve(curves, "train", "cls_loss"),
        "train_dfl_loss_curve": find_curve(curves, "train", "dfl_loss"),
        "val_box_loss_curve": find_curve(curves, "val", "box_loss"),
        "val_cls_loss_curve": find_curve(curves, "val", "cls_loss"),
        "val_dfl_loss_curve": find_curve(curves, "val", "dfl_loss"),
        "map50_curve": find_curve(curves, "mAP50(B)"),
        "map50_95_curve": find_curve(curves, "mAP50-95(B)"),
        "best_weights_path": str(best_weights.resolve()) if best_weights.exists() else None,
        "run_dir": str(save_dir.resolve()),
        "config": {k: getattr(args, k) for k in TRAIN_HYPERPARAM_KEYS},
    }

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[METRICS] map50={metrics['map50']}  map50_95={metrics['map50_95']}  "
          f"precision={metrics['precision']}  recall={metrics['recall']}")
    print(f"[METRICS] Wrote -> {metrics_path}")


if __name__ == "__main__":
    main()
