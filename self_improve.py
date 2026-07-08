#!/usr/bin/env python3
"""
self_improve.py
----------------
Recursive self-improvement loop for local YOLO training on an RTX 3070 (8 GB VRAM).

After each training cycle the local LLM (served by LM Studio on localhost:1234)
reviews the metrics and suggests hyperparameter changes. Those suggestions are
guard-railed for VRAM safety and applied to the next training cycle automatically.
Training continues from the previous cycle's best checkpoint each cycle.

Usage:
    python self_improve.py
    python self_improve.py --cycles 5 --epochs_per_cycle 10
    python self_improve.py --lm_studio_model "google/gemma-4-26b-a4b" --dry_run

Requirements:
    pip install openai ultralytics pyyaml
    LM Studio running locally with a model loaded and the local server started on port 1234.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-improving local YOLO training loop via LM Studio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Loop control
    p.add_argument("--cycles", type=int, default=5,
                   help="Number of self-improvement cycles to run.")
    p.add_argument("--epochs_per_cycle", type=int, default=10,
                   help="Training epochs per cycle.")
    # Initial training config (mirrors Ultralytics defaults)
    p.add_argument("--model", default="yolov8s.pt",
                   help="Starting checkpoint for cycle 1 (COCO-pretrained weights or path/to/best.pt).")
    p.add_argument("--data", default="data/dataset.yaml")
    p.add_argument("--project", default="outputs/self_improve")
    p.add_argument("--device", default="0")
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
    # LM Studio
    p.add_argument("--lm_studio_url", default="http://localhost:1234/v1",
                   help="LM Studio OpenAI-compatible API base URL.")
    p.add_argument("--lm_studio_model", default="google/gemma-4-26b-a4b",
                   help="Model identifier as shown in LM Studio.")
    p.add_argument("--llm_temperature", type=float, default=0.3)
    p.add_argument("--llm_max_tokens", type=int, default=4096,
                   help="google/gemma-4-26b-a4b is a reasoning model: it spends a chunk of this "
                        "budget on hidden 'reasoning_content' before emitting the JSON answer, "
                        "so this needs headroom beyond the JSON schema size alone.")
    p.add_argument("--keep_lm_studio_loaded", action="store_true",
                   help="By default, self_improve.py unloads the LM Studio model (via 'lms unload') "
                        "before each training subprocess and reloads it before each suggestion call — "
                        "on an 8GB GPU, training and LM Studio inference contending for the same GPU "
                        "at once is both VRAM-tight and makes LLM generation far slower. Pass this "
                        "flag to disable that and keep the model resident throughout (only sensible "
                        "with more VRAM headroom or a CPU-offloaded LM Studio model).")
    p.add_argument("--lms_cli", default=r"C:\Users\User\.lmstudio\bin\lms.exe",
                   help="Path to the LM Studio 'lms' CLI, used for unload/reload between cycles.")
    # Misc
    p.add_argument("--dry_run", action="store_true",
                   help="Skip actual training; only test the LM Studio suggestion loop.")
    p.add_argument("--log_dir", default="outputs/self_improve/loop_logs",
                   help="Directory for per-cycle JSON logs.")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to use for subprocess training calls.")
    return p.parse_args()


def lms_unload(args: argparse.Namespace) -> None:
    if args.keep_lm_studio_loaded:
        return
    print("[LMS] Unloading LM Studio model to free VRAM/GPU for training ...")
    subprocess.run([args.lms_cli, "unload", "--all"], text=True)


def lms_reload(args: argparse.Namespace) -> None:
    if args.keep_lm_studio_loaded:
        return
    print(f"[LMS] Reloading {args.lm_studio_model} for the suggestion call ...")
    subprocess.run([args.lms_cli, "load", args.lm_studio_model, "-y"], text=True)


CONFIG_KEYS = [
    "imgsz", "batch", "optimizer", "patience",
    "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
    "box", "cls", "dfl",
    "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
]


# ---------------------------------------------------------------------------
# Training subprocess
# ---------------------------------------------------------------------------

def run_training_cycle(
    cycle: int,
    config: dict[str, Any],
    weights: str,
    args: argparse.Namespace,
    metrics_path: Path,
) -> dict[str, Any] | None:
    """Launch train_yolo.py as a subprocess with the given config and starting weights."""
    script = Path(__file__).parent / "train_yolo.py"
    cmd = [
        args.python, str(script),
        "--weights", weights,
        "--data", args.data,
        "--epochs", str(args.epochs_per_cycle),
        "--project", args.project,
        "--name", f"cycle_{cycle:02d}",
        "--device", args.device,
        "--metrics_output", str(metrics_path),
    ]
    for key in CONFIG_KEYS:
        cmd += [f"--{key}", str(config[key])]

    print(f"\n{'='*60}")
    print(f"  CYCLE {cycle}: launching training subprocess")
    print(f"{'='*60}")
    print(f"  Starting weights: {weights}")
    print(f"  Command: {' '.join(cmd[:6])} ...")

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Training subprocess exited with code {result.returncode}")
        return None

    if not metrics_path.exists():
        print(f"[ERROR] Metrics file not written: {metrics_path}")
        return None

    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LM Studio suggestion engine
# ---------------------------------------------------------------------------

SUGGESTION_SCHEMA = """\
{
  "reasoning": "<1-3 sentences explaining the diagnosis and rationale>",
  "suggested_config": {
    "imgsz":           <int>,
    "batch":            <int>,
    "optimizer":        "<SGD|Adam|AdamW|NAdam|RAdam|RMSProp|auto>",
    "patience":         <int>,
    "lr0":              <float>,
    "lrf":              <float>,
    "momentum":         <float>,
    "weight_decay":     <float>,
    "warmup_epochs":    <float>,
    "warmup_momentum":  <float>,
    "warmup_bias_lr":   <float>,
    "box":              <float>,
    "cls":              <float>,
    "dfl":              <float>,
    "hsv_h":            <float>,
    "hsv_s":            <float>,
    "hsv_v":            <float>,
    "degrees":          <float>,
    "translate":        <float>,
    "scale":            <float>,
    "shear":            <float>,
    "perspective":      <float>,
    "flipud":           <float>,
    "fliplr":           <float>,
    "mosaic":           <float>,
    "mixup":            <float>,
    "copy_paste":       <float>
  },
  "data_suggestions": ["<optional list of data/augmentation improvement ideas>"]
}"""

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert computer-vision engineer specialising in fine-tuning YOLO
    object detectors on limited VRAM (8 GB RTX 3070 Laptop GPU). You review
    training metrics and suggest concrete hyperparameter changes to improve the
    next training cycle.

    Task context: detecting asbestos fibre types in polarised-light microscopy
    images. There are 7 classes with heavy class imbalance (instance counts):
    A-CP=10041, NA-OF=7925, A-CF=3895, NA-CS=1522, A-CRO=227, A-COF=114, A-AM=65.
    The rarest classes (A-AM, A-COF, A-CRO) are the hardest to detect reliably.

    Rules:
    - Hard GPU constraint: batch must stay in [4, 24] and imgsz in [320, 768] for an 8 GB GPU.
    - Prefer small, targeted changes over large sweeps.
    - If mAP is plateauing or overfitting (train loss falling, val loss rising), suggest
      stronger regularization/augmentation or a lower learning rate rather than more epochs.
    - If both train and val metrics are still improving steadily, prefer small refinements.
    - Keep your internal reasoning brief (a few sentences) — you are running on a shared local
      GPU with a limited token budget, so do not over-deliberate before answering.
    - Return ONLY valid JSON matching the schema — no markdown fences, no explanation outside the JSON.
""")


def build_user_prompt(cycle: int, metrics: dict[str, Any], current_config: dict[str, Any]) -> str:
    def trend(curve: list[float], label: str) -> str:
        if len(curve) < 2:
            return ""
        delta = curve[-1] - curve[0]
        return f"{label} moved {delta:+.4f} (start={curve[0]:.4f}, end={curve[-1]:.4f}). "

    trends = ""
    trends += trend(metrics.get("train_box_loss_curve", []), "train box_loss")
    trends += trend(metrics.get("val_box_loss_curve", []), "val box_loss")
    trends += trend(metrics.get("train_cls_loss_curve", []), "train cls_loss")
    trends += trend(metrics.get("val_cls_loss_curve", []), "val cls_loss")
    trends += trend(metrics.get("map50_95_curve", []), "mAP50-95")

    return textwrap.dedent(f"""\
        ## Training cycle {cycle} results

        mAP50     : {metrics.get('map50')}
        mAP50-95  : {metrics.get('map50_95')}
        Precision : {metrics.get('precision')}
        Recall    : {metrics.get('recall')}
        Epochs trained this cycle: {metrics.get('epochs_trained')}
        {trends}

        ## Current config
        {json.dumps(current_config, indent=2)}

        ## Task
        Suggest improvements for cycle {cycle + 1}. Return JSON matching this schema exactly:
        {SUGGESTION_SCHEMA}
    """)


def ask_lm_studio(
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Call LM Studio's OpenAI-compatible API and parse the JSON response."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] 'openai' package not installed. Run: pip install openai")
        return None

    client = OpenAI(base_url=args.lm_studio_url, api_key="lm-studio")

    print(f"\n[LLM] Asking LM Studio ({args.lm_studio_model}) for suggestions ...")
    try:
        response = client.chat.completions.create(
            model=args.lm_studio_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
        )
    except Exception as exc:
        print(f"[ERROR] LM Studio API call failed: {exc}")
        print("        Is LM Studio running with a model loaded and the local server started?")
        return None

    message = response.choices[0].message
    content = (message.content or "").strip()
    # google/gemma-4-26b-a4b is a reasoning model: it emits hidden chain-of-thought in
    # reasoning_content before (or instead of, if it runs out of budget) the real answer.
    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
    finish_reason = response.choices[0].finish_reason
    usage = response.usage
    if usage is not None:
        print(f"[LLM] completion_tokens={usage.completion_tokens} finish_reason={finish_reason}")

    if reasoning:
        print(f"[LLM] Reasoning ({len(reasoning)} chars):\n{reasoning}\n")
    print(f"[LLM] Content response:\n{content}\n")

    if not content:
        print("[WARN] Model produced no content — likely spent the whole token budget on "
              "reasoning_content. Increase --llm_max_tokens.")
        return None

    parsed = _extract_json(content)
    if parsed is None:
        print("[WARN] Could not parse LLM content as JSON.")
    return parsed


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Parse a JSON object out of raw text, tolerating markdown fences and truncation."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        last_brace = raw.rfind("}")
        if last_brace != -1:
            trimmed = raw[:last_brace + 1]
            depth = 0
            for i in range(len(trimmed) - 1, -1, -1):
                if trimmed[i] == "}":
                    depth += 1
                elif trimmed[i] == "{":
                    depth -= 1
                if depth == 0:
                    candidate = trimmed[i:]
                    try:
                        parsed = json.loads(candidate)
                        print("[WARN] LLM response was truncated — recovered partial JSON.")
                        return parsed
                    except json.JSONDecodeError:
                        break
        return None


# ---------------------------------------------------------------------------
# Apply suggestions safely
# ---------------------------------------------------------------------------

ALLOWED_CONFIG_KEYS = set(CONFIG_KEYS)

VALID_OPTIMIZERS = {"SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"}

VRAM_GUARDS = {
    "batch":            (4, 24),          # 8 GB ceiling for yolov8s
    "imgsz":            (320, 768),       # snapped to nearest multiple of 32 below
    "patience":         (5, 100),
    "lr0":              (1e-5, 1e-1),
    "lrf":              (0.001, 1.0),
    "momentum":         (0.6, 0.98),
    "weight_decay":     (0.0, 0.001),
    "warmup_epochs":    (0.0, 5.0),
    "warmup_momentum":  (0.0, 0.95),
    "warmup_bias_lr":   (0.0, 0.2),
    "box":              (0.5, 20.0),
    "cls":              (0.1, 4.0),
    "dfl":              (0.5, 5.0),
    "hsv_h":            (0.0, 1.0),
    "hsv_s":            (0.0, 1.0),
    "hsv_v":            (0.0, 1.0),
    "degrees":          (0.0, 45.0),
    "translate":        (0.0, 1.0),
    "scale":            (0.0, 1.0),
    "shear":            (0.0, 20.0),
    "perspective":      (0.0, 0.001),
    "flipud":           (0.0, 1.0),
    "fliplr":           (0.0, 1.0),
    "mosaic":           (0.0, 1.0),
    "mixup":            (0.0, 1.0),
    "copy_paste":       (0.0, 1.0),
}


def apply_suggestions(
    current_config: dict[str, Any],
    suggestion: dict[str, Any],
) -> dict[str, Any]:
    """Merge LLM suggestions into config with safety clamping."""
    new_config = deepcopy(current_config)
    suggested = suggestion.get("suggested_config", {})

    for key, value in suggested.items():
        if key not in ALLOWED_CONFIG_KEYS:
            print(f"[WARN] Ignoring unknown suggestion key: {key}")
            continue
        if key == "optimizer":
            if value not in VALID_OPTIMIZERS:
                print(f"[GUARD] optimizer: invalid value '{value}' — keeping '{current_config.get('optimizer')}'")
                continue
        elif key in VRAM_GUARDS:
            lo, hi = VRAM_GUARDS[key]
            try:
                clamped = max(lo, min(hi, value))
            except TypeError:
                print(f"[WARN] Ignoring non-numeric value for {key}: {value}")
                continue
            if key == "imgsz":
                clamped = int(round(clamped / 32) * 32)
            elif key == "batch" or key == "patience":
                clamped = int(round(clamped))
            if clamped != value:
                print(f"[GUARD] {key}: {value} -> clamped to {clamped}")
            value = clamped
        old = current_config.get(key)
        if old != value:
            print(f"  {key}: {old} -> {value}")
        new_config[key] = value

    return new_config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def save_cycle_log(
    log_dir: Path,
    cycle: int,
    config: dict[str, Any],
    metrics: dict[str, Any] | None,
    suggestion: dict[str, Any] | None,
    next_config: dict[str, Any],
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "cycle": cycle,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "suggestion": suggestion,
        "next_config": next_config,
    }
    log_path = log_dir / f"cycle_{cycle:02d}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)
    print(f"[LOG] Cycle log -> {log_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    log_dir = Path(args.log_dir)
    metrics_dir = Path(args.project) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {key: getattr(args, key) for key in CONFIG_KEYS}
    weights = args.model

    print("\n" + "="*60)
    print("  YOLO SELF-IMPROVEMENT LOOP")
    print(f"  Cycles: {args.cycles}  |  Epochs/cycle: {args.epochs_per_cycle}")
    print(f"  LM Studio: {args.lm_studio_url}  model={args.lm_studio_model}")
    print("="*60)

    best_map50_95 = -1.0
    best_config: dict[str, Any] = deepcopy(config)
    best_weights: str = weights
    best_cycle: int = 0

    for cycle in range(1, args.cycles + 1):
        print(f"\n{'#'*60}")
        print(f"  CYCLE {cycle} / {args.cycles}")
        print(f"{'#'*60}")
        print(f"  Config: imgsz={config['imgsz']}  batch={config['batch']}  lr0={config['lr0']}  "
              f"box={config['box']}  cls={config['cls']}  dfl={config['dfl']}")

        metrics_path = metrics_dir / f"cycle_{cycle:02d}_metrics.json"

        # --- Training ---
        if args.dry_run:
            print("[DRY RUN] Skipping training. Using fake metrics.")
            metrics = {
                "epochs_trained": args.epochs_per_cycle,
                "map50": round(0.3 + cycle * 0.05, 4),
                "map50_95": round(0.15 + cycle * 0.03, 4),
                "precision": round(0.5 + cycle * 0.02, 4),
                "recall": round(0.4 + cycle * 0.02, 4),
                "train_box_loss_curve": [round(2.0 - cycle * 0.1 + i * 0.01, 4) for i in range(5)],
                "val_box_loss_curve": [round(2.1 - cycle * 0.08 + i * 0.01, 4) for i in range(5)],
                "train_cls_loss_curve": [round(1.0 - cycle * 0.05 + i * 0.01, 4) for i in range(5)],
                "val_cls_loss_curve": [round(1.1 - cycle * 0.04 + i * 0.01, 4) for i in range(5)],
                "map50_95_curve": [round(0.15 + cycle * 0.03 + i * 0.005, 4) for i in range(5)],
                "best_weights_path": weights,
                "config": config,
            }
        else:
            lms_unload(args)
            metrics = run_training_cycle(cycle, config, weights, args, metrics_path)
            lms_reload(args)

        if metrics is None:
            print(f"[ERROR] Cycle {cycle} failed — stopping loop.")
            break

        map50_95 = metrics.get("map50_95") or -1.0
        print(f"\n[METRICS] map50={metrics.get('map50')}  map50_95={map50_95}  "
              f"precision={metrics.get('precision')}  recall={metrics.get('recall')}")

        # Chain next cycle's starting weights from this cycle's best checkpoint
        if metrics.get("best_weights_path"):
            weights = metrics["best_weights_path"]

        # Track the best config/weights seen so far
        if map50_95 > best_map50_95:
            best_map50_95 = map50_95
            best_config = deepcopy(config)
            best_weights = weights
            best_cycle = cycle
            print(f"[BEST]  New best mAP50-95={best_map50_95:.4f} at cycle {best_cycle}")

        if cycle == args.cycles:
            save_cycle_log(log_dir, cycle, config, metrics, None, config)
            print("\n[DONE] Final cycle complete. No further suggestions needed.")
            break

        # --- Ask LM Studio for suggestions ---
        user_prompt = build_user_prompt(cycle, metrics, config)
        suggestion = ask_lm_studio(user_prompt, args)

        if suggestion is None:
            print("[WARN] No valid suggestion received — keeping current config for next cycle.")
            next_config = deepcopy(config)
        else:
            reasoning = suggestion.get("reasoning", "")
            if reasoning:
                print(f"\n[LLM REASONING] {reasoning}")
            data_tips = suggestion.get("data_suggestions", [])
            if data_tips:
                print("[LLM DATA TIPS]")
                for tip in data_tips:
                    print(f"  - {tip}")
            print("\n[CONFIG CHANGES]")
            next_config = apply_suggestions(config, suggestion)

        save_cycle_log(log_dir, cycle, config, metrics, suggestion, next_config)
        config = next_config

    # -----------------------------------------------------------------------
    # Save the best config so future runs can start from it
    # -----------------------------------------------------------------------
    best_config_path = Path(args.project) / "best_config.json"
    best_config_out = dict(best_config)
    best_config_out["_best_cycle"] = best_cycle
    best_config_out["_best_map50_95"] = best_map50_95
    best_config_out["_best_weights_path"] = best_weights
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config_out, f, indent=2)

    print(f"\n[BEST CONFIG] Cycle {best_cycle} had highest mAP50-95={best_map50_95:.4f}")
    print(f"[BEST CONFIG] Best weights -> {best_weights}")
    print(f"[BEST CONFIG] Saved -> {best_config_path}")
    for k, v in best_config_out.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("  SELF-IMPROVEMENT LOOP COMPLETE")
    print(f"  Logs -> {log_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
