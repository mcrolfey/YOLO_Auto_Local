#!/usr/bin/env python3
"""
prepare_dataset.py
-------------------
Builds a YOLO dataset.yaml plus train/val image-list files from the raw
images/labels folders, without copying any image or label files.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --val_fraction 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare YOLO dataset.yaml + train/val splits")
    p.add_argument("--images_dir", default=r"C:\Users\User\Desktop\uncropped_all\combined05-07-26\images")
    p.add_argument("--labels_dir", default=r"C:\Users\User\Desktop\uncropped_all\combined05-07-26\labels")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="data")
    return p.parse_args()


def load_classes(labels_dir: Path) -> list[str]:
    classes_file = labels_dir / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found in {labels_dir}")
    with open(classes_file, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_classes(labels_dir)
    print(f"[CLASSES] {len(class_names)} classes: {class_names}")

    label_files = {f.stem: f for f in labels_dir.glob("*.txt") if f.stem != "classes"}
    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    matched = [f for f in image_files if f.stem in label_files]
    skipped = len(image_files) - len(matched)
    print(f"[MATCH] {len(matched)} images have matching label files ({skipped} skipped, no label file)")

    rng = random.Random(args.seed)
    matched_sorted = sorted(matched, key=lambda f: f.name)
    rng.shuffle(matched_sorted)

    n_val = max(1, int(len(matched_sorted) * args.val_fraction))
    val_files = matched_sorted[:n_val]
    train_files = matched_sorted[n_val:]

    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(str(p.resolve()) for p in train_files))
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(str(p.resolve()) for p in val_files))

    print(f"[SPLIT] train={len(train_files)}  val={len(val_files)}")

    def class_counts(files: list[Path]) -> collections.Counter:
        counts: collections.Counter = collections.Counter()
        for img in files:
            lbl = label_files[img.stem]
            with open(lbl, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    cid = int(line.split()[0])
                    counts[cid] += 1
        return counts

    for split_name, files in (("train", train_files), ("val", val_files)):
        counts = class_counts(files)
        print(f"[DIST] {split_name} class distribution:")
        for cid, name in enumerate(class_names):
            print(f"    {cid} {name}: {counts.get(cid, 0)}")

    dataset_yaml = out_dir / "dataset.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    yaml_content = (
        f"train: {train_txt.resolve().as_posix()}\n"
        f"val: {val_txt.resolve().as_posix()}\n"
        f"names:\n{names_block}\n"
    )
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[DONE] Wrote {dataset_yaml}")
    print(f"[DONE] Wrote {train_txt}")
    print(f"[DONE] Wrote {val_txt}")


if __name__ == "__main__":
    main()
