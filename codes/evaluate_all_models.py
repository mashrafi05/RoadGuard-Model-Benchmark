"""
02_evaluate_all_models.py
=========================
Evaluates all trained YOLO models on the held-out test set
(new_test/image + new_test/labels) at native 640 resolution, and reports:

    mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-score, Accuracy(proxy),
    FPS, Latency (ms), Params (M), GFLOPs, Model size (MB)

...plus a full class-wise breakdown for every model.

Results are written to:
    I:/Project_Conference/runs/evaluation/all_metrics.json
    I:/Project_Conference/runs/evaluation/all_metrics.csv
    I:/Project_Conference/runs/evaluation/classwise_metrics.csv
    I:/Project_Conference/runs/evaluation/<model_name>_classwise.csv
"""

import os
import json
import csv
import time
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# ── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"I:\Project_Conference")
TEST_IMG_DIR = Path(r"I:\Project_Conference\21431547\United_States\train\images")
TEST_LBL_DIR = Path(r"I:\Project_Conference\21431547\United_States\train\labels")

# Explicit paths to your trained weights
TRAINED_MODELS = {
    "YOLO11n":  PROJECT_ROOT / "runs" / "YOLO11n"  / "weights" / "best.pt",
    "YOLOv5n":  PROJECT_ROOT / "runs" / "YOLOv5n"  / "weights" / "best.pt",
    "YOLOv8n":  PROJECT_ROOT / "runs" / "YOLOv8n"  / "weights" / "best.pt",
    "YOLOv9t":  PROJECT_ROOT / "runs" / "YOLOv9t"  / "weights" / "best.pt",
    "YOLOv10n": PROJECT_ROOT / "runs" / "YOLOv10n" / "weights" / "best.pt",
}
 
NC = 5
NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Speed Breaker",
}

IMGSZ        = 640
DEVICE       = 0 if torch.cuda.is_available() else "cpu"
CONF_THRESH  = 0.1  # Standard threshold (eliminates false-positive explosion)
IOU_THRESH   = 0.4   # Standard NMS threshold
EVAL_BATCH   = 16     # Speeds up 5155 image validation
USE_TTA      = True   # Test-Time Augmentation (boosts detection accuracy on 640px)
WARMUP_RUNS  = 10
MEASURE_RUNS = 100

RESULTS_DIR = PROJECT_ROOT / "new_runs_USA" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEMP_ROOT = PROJECT_ROOT / "new_runs_USA" / "_eval_temp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def make_test_dataset_yaml() -> Path:
    """Prepares images/ and labels/ directories so Ultralytics can pair labels correctly."""
    images_dir = TEMP_ROOT / "images"
    labels_dir = TEMP_ROOT / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    def link_dir_contents(src_dir: Path, dst_dir: Path):
        for src_file in src_dir.iterdir():
            if not src_file.is_file():
                continue
            dst_file = dst_dir / src_file.name
            if dst_file.exists():
                continue
            try:
                os.link(src_file, dst_file)  # Hardlink
            except OSError:
                shutil.copy2(src_file, dst_file)  # Fallback to copy

    n_before_img = sum(1 for _ in images_dir.iterdir())
    if n_before_img == 0:
        link_dir_contents(TEST_IMG_DIR, images_dir)

    n_before_lbl = sum(1 for _ in labels_dir.iterdir())
    if n_before_lbl == 0:
        link_dir_contents(TEST_LBL_DIR, labels_dir)

    n_img = sum(1 for _ in images_dir.iterdir())
    n_lbl = sum(1 for _ in labels_dir.iterdir())
    print(f"  Dataset verified: {n_img} images, {n_lbl} labels in {TEMP_ROOT.resolve()}")

    yaml_path = TEMP_ROOT / "eval_data.yaml"
    yaml_content = (
        f"path: {TEMP_ROOT.resolve().as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"test: images\n"
        f"nc: {NC}\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in NAMES.items())
    )
    yaml_path.write_text(yaml_content)
    return yaml_path


def measure_fps(model: YOLO, img_dir: Path) -> tuple[float, float]:
    """Measures single-image inference FPS and latency (ms)."""
    img_files = (
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.jpeg"))
        + list(img_dir.glob("*.png"))
    )
    if not img_files:
        return 0.0, 0.0

    all_imgs = img_files * ((WARMUP_RUNS + MEASURE_RUNS) // len(img_files) + 1)
    warmup_imgs = [str(p) for p in all_imgs[:WARMUP_RUNS]]
    measure_imgs = [str(p) for p in all_imgs[WARMUP_RUNS: WARMUP_RUNS + MEASURE_RUNS]]

    for img in warmup_imgs:
        model.predict(img, imgsz=IMGSZ, device=DEVICE, conf=CONF_THRESH, verbose=False)

    if torch.cuda.is_available() and DEVICE != "cpu":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for img in measure_imgs:
        model.predict(img, imgsz=IMGSZ, device=DEVICE, conf=CONF_THRESH, verbose=False)
    if torch.cuda.is_available() and DEVICE != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    fps = MEASURE_RUNS / elapsed
    ms_per_img = (elapsed / MEASURE_RUNS) * 1000
    return round(fps, 2), round(ms_per_img, 2)


def model_size_mb(weight_path: Path) -> float:
    return round(weight_path.stat().st_size / (1024 ** 2), 2)


def get_model_info(model: YOLO) -> tuple[float, float]:
    """Calculates model parameters (M) and GFLOPs."""
    params_m = -1.0
    gflops = -1.0
    try:
        if hasattr(model, "model") and model.model is not None:
            n_p = sum(p.numel() for p in model.model.parameters())
            params_m = round(n_p / 1e6, 2)
    except Exception:
        pass

    try:
        from ultralytics.utils.torch_utils import get_flops
        flops = get_flops(model.model, imgsz=IMGSZ)
        gflops = round(flops, 2)
    except Exception:
        pass

    return params_m, gflops


def class_accuracy_from_confusion(val_results, nc: int) -> list[float]:
    """Calculates proxy accuracy TP / (TP + FP + FN) per class from confusion matrix."""
    try:
        cm = val_results.confusion_matrix.matrix
        accs = []
        for i in range(nc):
            tp = cm[i, i]
            fp = cm[i, :].sum() - tp
            fn = cm[:, i].sum() - tp
            denom = tp + fp + fn
            accs.append(round(float(tp / denom), 4) if denom > 0 else 0.0)
        return accs
    except Exception:
        return [0.0] * nc


def evaluate_model(name: str, weight_path: Path, data_yaml: Path) -> tuple[dict, list[dict]]:
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {name}")
    print(f"  Weights   : {weight_path}")
    print(f"  TTA Mode  : {'Enabled' if USE_TTA else 'Disabled'}")
    print(f"{'─'*60}")

    if not weight_path.exists():
        print(f"  ✗ Weight file not found: {weight_path}")
        return {"model": name, "error": "file not found"}, []

    model = YOLO(str(weight_path))

    val_results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=EVAL_BATCH,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        augment=USE_TTA,
        device=DEVICE,
        verbose=False,
    )

    precision = float(val_results.box.mp)
    recall    = float(val_results.box.mr)
    map50     = float(val_results.box.map50)
    map50_95  = float(val_results.box.map)
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    class_acc = class_accuracy_from_confusion(val_results, NC)
    overall_accuracy = round(sum(class_acc) / len(class_acc), 4) if class_acc else 0.0

    fps, inf_ms = measure_fps(model, TEST_IMG_DIR)
    params_m, gflops = get_model_info(model)
    size_mb = model_size_mb(weight_path)

    overall = {
        "model": name,
        "mAP_50": round(map50, 4),
        "mAP_50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": overall_accuracy,
        "fps": fps,
        "inference_ms": inf_ms,
        "params_M": params_m,
        "GFLOPs": gflops,
        "model_size_mb": size_mb,
        "weight_path": str(weight_path),
    }

    # Extract class-wise statistics
    classwise = []
    try:
        ap_class_idx   = list(val_results.box.ap_class_index)
        p_per_class    = val_results.box.p
        r_per_class    = val_results.box.r
        ap50_per_class = val_results.box.ap50
        ap_per_class   = val_results.box.ap
        for row, cls_idx in enumerate(ap_class_idx):
            p_c = float(p_per_class[row])
            r_c = float(r_per_class[row])
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            classwise.append({
                "model": name,
                "class_id": int(cls_idx),
                "class_name": NAMES.get(int(cls_idx), str(cls_idx)),
                "precision": round(p_c, 4),
                "recall": round(r_c, 4),
                "f1_score": round(f1_c, 4),
                "mAP_50": round(float(ap50_per_class[row]), 4),
                "mAP_50_95": round(float(ap_per_class[row]), 4),
                "accuracy": class_acc[int(cls_idx)] if int(cls_idx) < len(class_acc) else 0.0,
            })
    except Exception as e:
        print(f"  ⚠ Could not extract class-wise metrics: {e}")

    print(f"\n  Results for {name}:")
    for k, v in overall.items():
        if k != "weight_path":
            print(f"    {k:15s}: {v}")

    return overall, classwise


def main():
    print(f"Starting evaluation on {len(TRAINED_MODELS)} model(s)...")
    data_yaml = make_test_dataset_yaml()

    all_overall = []
    all_classwise = []

    for name, weight_path in TRAINED_MODELS.items():
        overall, classwise = evaluate_model(name, weight_path, data_yaml)
        all_overall.append(overall)
        all_classwise.extend(classwise)

        if classwise:
            per_model_csv = RESULTS_DIR / f"{name}_classwise.csv"
            with open(per_model_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(classwise[0].keys()))
                writer.writeheader()
                writer.writerows(classwise)
            print(f"  ✓ Saved class-wise CSV: {per_model_csv}")

    # Save overall JSON
    json_path = RESULTS_DIR / "all_metrics.json"
    with open(json_path, "w") as f:
        json.dump({"overall": all_overall, "classwise": all_classwise}, f, indent=2)
    print(f"\n✓ Overall JSON saved : {json_path}")

    # Save overall CSV
    csv_path = RESULTS_DIR / "all_metrics.csv"
    fieldnames = [
        "model", "mAP_50", "mAP_50_95", "precision", "recall", "f1_score",
        "accuracy", "fps", "inference_ms", "params_M", "GFLOPs", "model_size_mb",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_overall)
    print(f"✓ Overall CSV saved  : {csv_path}")

    # Save combined class-wise CSV
    if all_classwise:
        cw_path = RESULTS_DIR / "classwise_metrics.csv"
        with open(cw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_classwise[0].keys()))
            writer.writeheader()
            writer.writerows(all_classwise)
        print(f"✓ Classwise CSV saved: {cw_path}")

    print(f"\nEvaluation complete. All outputs saved in: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()