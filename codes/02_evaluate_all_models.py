"""
02_evaluate_all_models.py
=========================
Evaluates all trained models on the SAME test set.
Collects: Precision, Recall, F1, mAP@0.5, FPS, Inference Time.

Run after 01_train_all_models.py
    python 02_evaluate_all_models.py
"""

import json
import time
import csv
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_YAML   = "data.yaml"
IMGSZ       = 640
DEVICE      = 0          # GPU 0, or "cpu"
CONF_THRESH = 0.25
IOU_THRESH  = 0.50
BATCH       = 1          # batch=1 for realistic FPS measurement
WARMUP_RUNS = 10         # discard first N inferences (GPU warmup)
MEASURE_RUNS = 100       # number of runs for FPS timing

SAVE_DIR = Path("runs/baseline")
RESULTS_DIR = Path("runs/evaluation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── MODEL WEIGHT PATHS ────────────────────────────────────────────────────────
# These are created by 01_train_all_models.py
# Edit paths if you moved the weights files
MODELS = {
    "YOLO-Nano"    : SAVE_DIR / "YOLO-Nano"  / "weights" / "best.pt",
    "YOLOv8n"      : SAVE_DIR / "YOLOv8n"    / "weights" / "best.pt",
    "YOLO11n"      : SAVE_DIR / "YOLO11n"    / "weights" / "best.pt",
    # Add NanoDet / MobileNet-SSD paths here if trained separately
    # "NanoDet"    : Path("runs/nanodet/best.pt"),
    # "MobileNetSSD": Path("runs/mobilessd/best.pt"),
}
# ─────────────────────────────────────────────────────────────────────────────


def measure_fps(model: YOLO, img_dir: Path) -> tuple[float, float]:
    """
    Measure real inference FPS on actual test images.
    Returns (fps, avg_ms_per_image).
    """
    img_files = list(img_dir.glob("*.jpg")) + \
                list(img_dir.glob("*.jpeg")) + \
                list(img_dir.glob("*.png"))

    if not img_files:
        print("  ⚠ No test images found for FPS measurement")
        return 0.0, 0.0

    # Cycle images if fewer than WARMUP + MEASURE runs
    all_imgs = (img_files * ((WARMUP_RUNS + MEASURE_RUNS) // len(img_files) + 1))
    warmup_imgs  = [str(p) for p in all_imgs[:WARMUP_RUNS]]
    measure_imgs = [str(p) for p in all_imgs[WARMUP_RUNS : WARMUP_RUNS + MEASURE_RUNS]]

    # Warmup (GPU needs a few runs to reach stable speed)
    for img in warmup_imgs:
        model.predict(img, imgsz=IMGSZ, device=DEVICE,
                      conf=CONF_THRESH, verbose=False)

    # Timed measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for img in measure_imgs:
        model.predict(img, imgsz=IMGSZ, device=DEVICE,
                      conf=CONF_THRESH, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed   = time.perf_counter() - start
    fps       = MEASURE_RUNS / elapsed
    ms_per_img = (elapsed / MEASURE_RUNS) * 1000

    return round(fps, 2), round(ms_per_img, 2)


def count_parameters(model: YOLO) -> int:
    """Count trainable parameters in the model."""
    try:
        return sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    except Exception:
        return -1


def model_size_mb(weight_path: Path) -> float:
    """Return weight file size in MB."""
    return round(weight_path.stat().st_size / (1024 ** 2), 2)


def evaluate_model(name: str, weight_path: Path) -> dict:
    """Run full evaluation for one model. Returns metric dict."""
    print(f"\n{'─'*55}")
    print(f"  Evaluating: {name}")
    print(f"  Weights   : {weight_path}")
    print(f"{'─'*55}")

    if not weight_path.exists():
        print(f"  ✗ Weights not found: {weight_path}")
        return {"model": name, "error": "weights not found"}

    model = YOLO(str(weight_path))

    # ── 1. Standard validation metrics (P, R, mAP) ──────────────────────────
    val_results = model.val(
        data    = DATA_YAML,
        split   = "test",           # evaluate on test set
        imgsz   = IMGSZ,
        batch   = BATCH,
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        device  = DEVICE,
        verbose = False,
    )

    # Extract metrics from Ultralytics results object
    precision = float(val_results.box.mp)          # mean precision
    recall    = float(val_results.box.mr)          # mean recall
    map50     = float(val_results.box.map50)       # mAP@0.5
    map50_95  = float(val_results.box.map)         # mAP@0.5:0.95

    # F1 score (harmonic mean of P and R)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # ── 2. FPS / Inference time ───────────────────────────────────────────────
    test_img_dir = Path("images/test")
    fps, inf_time_ms = measure_fps(model, test_img_dir)

    # ── 3. Model efficiency info ──────────────────────────────────────────────
    params    = count_parameters(model)
    size_mb   = model_size_mb(weight_path)

    metrics = {
        "model"           : name,
        "precision"       : round(precision, 4),
        "recall"          : round(recall, 4),
        "f1_score"        : round(f1, 4),
        "mAP_50"          : round(map50, 4),
        "mAP_50_95"       : round(map50_95, 4),
        "fps"             : fps,
        "inference_ms"    : inf_time_ms,
        "params_M"        : round(params / 1e6, 2) if params > 0 else "N/A",
        "model_size_mb"   : size_mb,
        "weight_path"     : str(weight_path),
    }

    print(f"\n  Results for {name}:")
    print(f"    Precision      : {precision:.4f}")
    print(f"    Recall         : {recall:.4f}")
    print(f"    F1-score       : {f1:.4f}")
    print(f"    mAP@0.5        : {map50:.4f}")
    print(f"    mAP@0.5:0.95   : {map50_95:.4f}")
    print(f"    FPS            : {fps}")
    print(f"    Inference time : {inf_time_ms} ms")
    print(f"    Params         : {params/1e6:.2f}M")
    print(f"    Model size     : {size_mb} MB")

    return metrics


def main():
    all_results = []

    for name, weight_path in MODELS.items():
        result = evaluate_model(name, weight_path)
        all_results.append(result)

    # ── Save as JSON ──────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "all_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ JSON results saved: {json_path}")

    # ── Save as CSV ───────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "all_metrics.csv"
    fieldnames = [
        "model", "precision", "recall", "f1_score",
        "mAP_50", "mAP_50_95", "fps", "inference_ms",
        "params_M", "model_size_mb"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"✓ CSV results saved : {csv_path}")
    print("\nNext: run 03_results_table.py")


if __name__ == "__main__":
    main()
