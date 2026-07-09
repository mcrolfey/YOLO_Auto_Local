#!/usr/bin/env python3
"""
prepare_dataset.py
-------------------
Builds a YOLO dataset.yaml plus train/val image-list files from one or more
raw images/labels folder pairs, without copying any image or label files.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --val_fraction 0.15 --seed 42

    # Merge extra data collected later, on top of the default folder:
    python prepare_dataset.py \\
        --images_dir "C:\\data\\combined05-07-26\\images" "C:\\data\\session2\\images" \\
        --labels_dir "C:\\data\\combined05-07-26\\labels" "C:\\data\\session2\\labels"
"""

from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path

DEFAULT_IMAGES_DIR = r"C:\Users\User\Desktop\uncropped_all\combined05-07-26\images"
DEFAULT_LABELS_DIR = r"C:\Users\User\Desktop\uncropped_all\combined05-07-26\labels"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare YOLO dataset.yaml + train/val splits")
    p.add_argument("--images_dir", nargs="+", default=[DEFAULT_IMAGES_DIR],
                   help="One or more image folders to merge. Defaults to the main dataset folder "
                        "so no arguments are needed for a normal run.")
    p.add_argument("--labels_dir", nargs="+", default=[DEFAULT_LABELS_DIR],
                   help="Matching label folders, same order and count as --images_dir.")
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


def match_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Match images to their label file within a single folder pair."""
    label_files = {f.stem: f for f in labels_dir.glob("*.txt") if f.stem != "classes"}
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES]

    matched = [(img, label_files[img.stem]) for img in image_files if img.stem in label_files]
    skipped = len(image_files) - len(matched)
    print(f"[MATCH] {images_dir} -> {len(matched)} images matched ({skipped} skipped, no label file)")
    return matched


def main() -> None:
    args = parse_args()
    if len(args.images_dir) != len(args.labels_dir):
        raise ValueError(
            f"--images_dir has {len(args.images_dir)} entries but --labels_dir has "
            f"{len(args.labels_dir)}; pass them in matching pairs."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folder_pairs = [(Path(i), Path(l)) for i, l in zip(args.images_dir, args.labels_dir)]

    class_names = load_classes(folder_pairs[0][1])
    print(f"[CLASSES] {len(class_names)} classes (from {folder_pairs[0][1]}): {class_names}")
    for _, labels_dir in folder_pairs[1:]:
        other_classes = load_classes(labels_dir)
        if other_classes != class_names:
            raise ValueError(
                f"classes.txt mismatch: {labels_dir} has {other_classes}, "
                f"expected {class_names} (from {folder_pairs[0][1]}). All merged folders must "
                "share the same class list and order."
            )

    all_pairs: list[tuple[Path, Path]] = []
    for images_dir, labels_dir in folder_pairs:
        all_pairs.extend(match_pairs(images_dir, labels_dir))

    print(f"[MATCH] {len(all_pairs)} images matched across {len(folder_pairs)} folder(s)")

    rng = random.Random(args.seed)
    pairs_sorted = sorted(all_pairs, key=lambda pair: str(pair[0]))
    rng.shuffle(pairs_sorted)

    n_val = max(1, int(len(pairs_sorted) * args.val_fraction))
    val_pairs = pairs_sorted[:n_val]
    train_pairs = pairs_sorted[n_val:]

    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(str(img.resolve()) for img, _ in train_pairs))
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(str(img.resolve()) for img, _ in val_pairs))

    print(f"[SPLIT] train={len(train_pairs)}  val={len(val_pairs)}")

    def class_counts(pairs: list[tuple[Path, Path]]) -> collections.Counter:
        counts: collections.Counter = collections.Counter()
        for _, lbl in pairs:
            with open(lbl, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    cid = int(line.split()[0])
                    counts[cid] += 1
        return counts

    for split_name, pairs in (("train", train_pairs), ("val", val_pairs)):
        counts = class_counts(pairs)
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
