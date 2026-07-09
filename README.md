# YOLO_Auto_Local

Self-improving local YOLO training loop for asbestos-fibre object detection.
Trains YOLO on a local RTX 3070 (8 GB VRAM); after each cycle, a local LLM
served by LM Studio reviews the metrics and suggests hyperparameter changes,
which are validated against VRAM/safety guardrails and applied to the next
training cycle. Training carries forward from the previous cycle's best
checkpoint, so the loop keeps refining a single model over time.

## Setup

1. Use the existing `yolo` conda environment (already has `ultralytics` +
   `torch` with CUDA):

   ```
   conda activate yolo
   pip install -r requirements.txt
   ```

2. Start LM Studio, load a model (default expected: `google/gemma-4-26b-a4b`),
   and start the local server (default `http://localhost:1234`).

3. Build the dataset split (run once, or again if the raw data changes):

   ```
   python prepare_dataset.py
   ```

   With no arguments this reads images/labels from the default paths below
   and writes `data/dataset.yaml`, `data/train.txt`, `data/val.txt`
   (file-list based, no images are copied):

   - Images: `C:\Users\User\Desktop\uncropped_all\combined05-07-26\images`
   - Labels: `C:\Users\User\Desktop\uncropped_all\combined05-07-26\labels`

   To use a different folder, or merge in extra images collected later,
   pass one or more `--images_dir`/`--labels_dir` pairs (same order, same
   count — all folders must share the same `classes.txt`):

   ```
   python prepare_dataset.py --images_dir "D:\new_batch\images" --labels_dir "D:\new_batch\labels"

   # merge the default folder with a new one:
   python prepare_dataset.py \
     --images_dir "C:\Users\User\Desktop\uncropped_all\combined05-07-26\images" "D:\new_batch\images" \
     --labels_dir "C:\Users\User\Desktop\uncropped_all\combined05-07-26\labels" "D:\new_batch\labels"
   ```

## Running the loop

```
python self_improve.py --cycles 5 --epochs_per_cycle 10
```

Useful flags:

- `--dry_run` — exercise the LM Studio suggestion loop without training.
- `--model yolov8s.pt` — starting checkpoint for cycle 1.
- `--lm_studio_model` — model identifier as shown in LM Studio.
- `--keep_lm_studio_loaded` — by default, the loop unloads the LM Studio model
  before each training subprocess and reloads it before each suggestion call
  (on an 8 GB GPU, training and LLM inference contending for VRAM at once is
  both tight and much slower). Pass this to keep it resident throughout.

Each cycle:
1. Trains via `train_yolo.py` (subprocess) starting from the previous cycle's
   `best.pt` (or `--model` on cycle 1).
2. Reads back mAP50, mAP50-95, precision/recall and per-epoch loss curves.
3. Sends the metrics + current config to the LM Studio model and asks for a
   JSON hyperparameter suggestion.
4. Clamps the suggestion to safe ranges for an 8 GB GPU (see `VRAM_GUARDS` in
   `self_improve.py`) and merges it into the config for the next cycle.
5. Logs everything to `outputs/self_improve/loop_logs/cycle_NN.json`.

At the end, the best-performing cycle's config and checkpoint path are saved
to `outputs/self_improve/best_config.json`.

## Files

- `prepare_dataset.py` — builds `data/dataset.yaml` + train/val splits.
- `train_yolo.py` — single-cycle Ultralytics training wrapper (subprocess target).
- `self_improve.py` — the self-improvement loop and LM Studio client.

## Notes

- Dataset has 7 classes (from `labels/classes.txt`): `A-AM, A-CF, A-COF, A-CP,
  A-CRO, NA-CS, NA-OF`, with heavy class imbalance (A-CP and NA-OF dominate;
  A-AM, A-COF, A-CRO are rare). This context is included in the LLM prompt.
- `batch` is guarded to `[4, 24]` and `imgsz` to `[320, 768]` (snapped to a
  multiple of 32) to stay within 8 GB VRAM on a `yolov8s` model.
