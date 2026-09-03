import gc
import json
import time
import csv
import shutil
from pathlib import Path
from datetime import datetime

import torch

# ── CONFIG — identical for every model ───────────────────────────────────────
DATA_YAML    = r"I:/Project_Conference/data.yaml"
EPOCHS       = 100
IMGSZ        = 512
BATCH        = 24
OPTIMIZER    = "SGD"
LR0          = 0.01
LRF          = 0.01
MOMENTUM     = 0.937
WEIGHT_DECAY = 0.0005
WORKERS      = 0           # MUST be 0 on Windows
DEVICE       = 0           # GPU index; "cpu" if no GPU
PATIENCE     = 15
AMP          = True
CLOSE_MOSAIC = 10
CHECKPOINT_EVERY = 10

# Seed 42 is already done — skipped in multi-seed runs
SEEDS = [0, 21]    # 2 new seeds + seed 42 already done = 3 total

SAVE_ROOT = Path(r"I:/Project_Conference/runs")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
    "Speed Breaker",
]

MODELS = [
    ("YOLOv5n",  "yolov5nu.pt", "YOLOv5 nano — ultralytics version"),
    ("YOLOv8n",  "yolov8n.pt",  "YOLOv8 nano — standard baseline"),
    ("YOLOv9t",  "yolov9t.pt",  "YOLOv9 tiny"),
    ("YOLOv10n", "yolov10n.pt", "YOLOv10 nano"),
    ("YOLO11n",  "yolo11n.pt",  "YOLO11 nano — latest architecture"),
]

LOG_PATH = SAVE_ROOT / "master_log.json"

# ── UTILITIES ─────────────────────────────────────────────────────────────────

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_master_log() -> dict:
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            return json.load(f)
    return {}


def save_master_log(log: dict):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def model_save_dir(name: str) -> Path:
    return SAVE_ROOT / name


def metrics_path(name: str) -> Path:
    return model_save_dir(name) / "metrics.json"


# ── CHECKPOINT CALLBACK ───────────────────────────────────────────────────────

class CheckpointCallback:
    def __init__(self, save_dir: Path, every: int):
        self.save_dir = save_dir / "weights"
        self.every    = every
        self.epoch    = 0

    def on_train_epoch_end(self, trainer):
        self.epoch += 1
        if self.epoch % self.every == 0:
            src = Path(trainer.best)
            if src.exists():
                dst = self.save_dir / f"epoch_{self.epoch}.pt"
                shutil.copy2(src, dst)
                print(f"  [checkpoint] saved epoch_{self.epoch}.pt")


# ── ORIGINAL SINGLE-SEED TRAINING (seed=42, already done) ────────────────────

def train_model(name: str, weights: str, master_log: dict) -> bool:
    from ultralytics import YOLO

    save_dir = model_save_dir(name)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  TRAINING: {name}  (seed=42)")
    print(f"  Weights : {weights}")
    print(f"{'='*62}")

    info = master_log.get(name, {})
    if info.get("train_status") == "success":
        print(f"  Already trained — skipping.")
        return True

    clear_gpu()

    try:
        model = YOLO(weights)
        cb = CheckpointCallback(save_dir, CHECKPOINT_EVERY)
        model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)

        start = time.time()
        model.train(
            data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
            optimizer=OPTIMIZER, lr0=LR0, lrf=LRF, momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY, workers=WORKERS, device=DEVICE,
            patience=PATIENCE, seed=42, amp=AMP, close_mosaic=CLOSE_MOSAIC,
            project=str(SAVE_ROOT), name=name, exist_ok=True,
            verbose=True, save=True, save_period=CHECKPOINT_EVERY,
        )
        elapsed = time.time() - start

        master_log.setdefault(name, {}).update({
            "train_status"      : "success",
            "weights"           : weights,
            "training_time_min" : round(elapsed / 60, 2),
            "best_pt"           : str(save_dir / "weights" / "best.pt"),
            "last_pt"           : str(save_dir / "weights" / "last.pt"),
            "timestamp"         : datetime.now().isoformat(),
        })
        save_master_log(master_log)
        print(f"\n  DONE: {name} in {elapsed/60:.1f} min")
        return True

    except torch.cuda.OutOfMemoryError:
        msg = f"GPU OOM. Reduce BATCH (currently {BATCH})."
        print(f"\n  ERROR: {msg}")
        master_log.setdefault(name, {})["train_status"] = f"error: {msg}"
        save_master_log(master_log)
        return False

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"\n  ERROR training {name}: {msg}")
        master_log.setdefault(name, {})["train_status"] = f"error: {msg}"
        save_master_log(master_log)
        return False

    finally:
        clear_gpu()


# ── EVALUATION (seed=42 run) ──────────────────────────────────────────────────

def measure_fps(model, test_img_dir: Path,
                warmup: int = 20, runs: int = 200) -> tuple:
    imgs = (list(test_img_dir.glob("*.jpg")) +
            list(test_img_dir.glob("*.jpeg")) +
            list(test_img_dir.glob("*.png")))
    if not imgs:
        return 0.0, 0.0

    pool         = (imgs * ((warmup + runs) // len(imgs) + 1))
    warmup_imgs  = [str(p) for p in pool[:warmup]]
    measure_imgs = [str(p) for p in pool[warmup: warmup + runs]]

    for img in warmup_imgs:
        model.predict(img, imgsz=IMGSZ, device=DEVICE, conf=0.25, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for img in measure_imgs:
        model.predict(img, imgsz=IMGSZ, device=DEVICE, conf=0.25, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed    = time.perf_counter() - t0
    fps        = runs / elapsed
    ms_per_img = elapsed / runs * 1000
    return round(fps, 2), round(ms_per_img, 3)


def count_params(model) -> float:
    try:
        return round(
            sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    except Exception:
        return -1.0


def get_gflops(model) -> float:
    try:
        info = model.info(verbose=False)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            return round(float(info[3]), 2)
    except Exception:
        pass
    try:
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        model.info(verbose=True)
        sys.stdout = old_stdout
        for line in buffer.getvalue().splitlines():
            if "GFLOPs" in line or "GFLOP" in line:
                for token in line.split():
                    try:
                        return round(float(token), 2)
                    except ValueError:
                        continue
    except Exception:
        pass
    return -1.0


def evaluate_model(name: str, master_log: dict) -> dict:
    from ultralytics import YOLO

    save_dir = model_save_dir(name)
    best_pt  = save_dir / "weights" / "best.pt"

    print(f"\n{'─'*62}")
    print(f"  EVALUATING: {name}")
    print(f"  Weights  : {best_pt}")
    print(f"{'─'*62}")

    if not best_pt.exists():
        print(f"  ERROR: best.pt not found.")
        return {}

    info = master_log.get(name, {})
    if info.get("eval_status") == "success" and metrics_path(name).exists():
        print(f"  Already evaluated — loading saved metrics.")
        with open(metrics_path(name)) as f:
            return json.load(f)

    clear_gpu()

    try:
        model = YOLO(str(best_pt))

        val = model.val(
            data=DATA_YAML, split="test", imgsz=IMGSZ, batch=BATCH,
            conf=0.25, iou=0.5, device=DEVICE, workers=WORKERS,
            verbose=True, project=str(save_dir), name="eval",
            exist_ok=True, plots=True,
        )

        precision = float(val.box.mp)
        recall    = float(val.box.mr)
        map50     = float(val.box.map50)
        map50_95  = float(val.box.map)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        per_class = {}
        try:
            maps  = val.box.maps
            ap50s = val.box.ap50
            for i, cname in enumerate(CLASS_NAMES):
                per_class[cname] = {
                    "mAP50"   : round(float(ap50s[i]), 4) if i < len(ap50s) else None,
                    "mAP50_95": round(float(maps[i]),  4) if i < len(maps)  else None,
                }
        except Exception:
            pass

        test_img_dir = Path("images/test")
        fps, ms      = measure_fps(model, test_img_dir)
        params_m     = count_params(model)
        gflops       = get_gflops(model)
        size_mb      = round(best_pt.stat().st_size / 1024**2, 2)

        metrics = {
            "model"     : name,
            "mAP50"     : round(map50,     4),
            "mAP50_95"  : round(map50_95,  4),
            "precision" : round(precision,  4),
            "recall"    : round(recall,     4),
            "f1"        : round(f1,         4),
            "accuracy"  : round(map50,      4),
            "fps"       : fps,
            "ms_per_img": ms,
            "params_M"  : params_m,
            "gflops"    : gflops,
            "size_mb"   : size_mb,
            "per_class" : per_class,
            "best_pt"   : str(best_pt),
            "eval_time" : datetime.now().isoformat(),
        }

        with open(metrics_path(name), "w") as f:
            json.dump(metrics, f, indent=2)

        master_log.setdefault(name, {})["eval_status"] = "success"
        save_master_log(master_log)

        print(f"\n  Results for {name}:")
        print(f"    mAP@0.5      : {map50:.4f}")
        print(f"    mAP@0.5:0.95 : {map50_95:.4f}")
        print(f"    Precision    : {precision:.4f}")
        print(f"    Recall       : {recall:.4f}")
        print(f"    F1-score     : {f1:.4f}")
        print(f"    FPS          : {fps}")
        print(f"    Params       : {params_m}M")
        print(f"    GFLOPs       : {gflops}")

        return metrics

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"\n  ERROR evaluating {name}: {msg}")
        master_log.setdefault(name, {})["eval_status"] = f"error: {msg}"
        save_master_log(master_log)
        return {}

    finally:
        clear_gpu()


# ── TFLITE EXPORT ─────────────────────────────────────────────────────────────

def export_tflite(name: str, master_log: dict):
    from ultralytics import YOLO

    save_dir   = model_save_dir(name)
    best_pt    = save_dir / "weights" / "best.pt"
    tflite_dir = save_dir / "tflite"

    info = master_log.get(name, {})
    if info.get("tflite_status") == "success":
        print(f"  TFLite already exported for {name} — skipping.")
        return

    if not best_pt.exists():
        print(f"  Skipping TFLite export — best.pt not found for {name}")
        return

    print(f"\n  Exporting TFLite for {name} ...")
    try:
        model = YOLO(str(best_pt))
        export_path = model.export(
            format="tflite", imgsz=IMGSZ, int8=True, data=DATA_YAML,
        )
        tflite_dir.mkdir(exist_ok=True)
        src = Path(export_path) if export_path else None
        if src and src.exists():
            dst = tflite_dir / src.name
            shutil.copy2(src, dst)
            print(f"  TFLite saved: {dst}")
            master_log.setdefault(name, {})["tflite_status"] = "success"
            master_log[name]["tflite_path"] = str(dst)
        else:
            print(f"  TFLite export path not returned.")
            master_log.setdefault(name, {})["tflite_status"] = "error: path not found"

    except Exception as e:
        print(f"  TFLite export failed for {name}: {e}")
        master_log.setdefault(name, {})["tflite_status"] = f"error: {e}"

    save_master_log(master_log)
    clear_gpu()


# ── COMPARISON TABLE ──────────────────────────────────────────────────────────

COLUMNS = [
    ("mAP50",      "mAP@0.5",      ".4f"),
    ("mAP50_95",   "mAP@0.5:0.95", ".4f"),
    ("precision",  "Precision",    ".4f"),
    ("recall",     "Recall",       ".4f"),
    ("f1",         "F1-score",     ".4f"),
    ("accuracy",   "Accuracy",     ".4f"),
    ("fps",        "FPS",          ".1f"),
    ("ms_per_img", "Inf.(ms)",     ".2f"),
    ("params_M",   "Params(M)",    ".2f"),
    ("gflops",     "GFLOPs",       ".2f"),
    ("size_mb",    "Size(MB)",     ".1f"),
]


def best_in_col(all_metrics: list, key: str):
    lower_better = {"ms_per_img", "params_M", "gflops", "size_mb"}
    vals = [m[key] for m in all_metrics
            if isinstance(m.get(key), (int, float)) and m[key] >= 0]
    if not vals:
        return None
    return min(vals) if key in lower_better else max(vals)


def build_comparison_table(all_metrics: list):
    if not all_metrics:
        print("  No metrics to compare yet.")
        return

    csv_path = SAVE_ROOT / "comparison_table.csv"
    txt_path = SAVE_ROOT / "comparison_table.txt"
    lat_path = SAVE_ROOT / "latex_table.tex"

    fieldnames = ["model"] + [c[0] for c in COLUMNS]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_metrics)
    print(f"\n  CSV saved : {csv_path}")

    col_w   = 12
    header  = f"{'Model':<12}" + "".join(f"{c[1]:>{col_w}}" for c in COLUMNS)
    divider = "-" * len(header)
    lines   = [divider, "  BASELINE COMPARISON TABLE", divider, header, divider]

    for m in all_metrics:
        row = f"{m['model']:<12}"
        for key, label, fmt in COLUMNS:
            val  = m.get(key)
            best = best_in_col(all_metrics, key)
            if val is None or val < 0:
                cell = f"{'N/A':>{col_w}}"
            else:
                s    = format(val, fmt)
                if val == best:
                    s = f"[{s}]"
                cell = f"{s:>{col_w}}"
            row += cell
        lines.append(row)

    lines += [divider, "[x] = best in column", divider]
    table_str = "\n".join(lines)
    txt_path.write_text(table_str, encoding="utf-8")
    print(f"  TXT saved : {txt_path}")
    print("\n" + table_str)

    lat_lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\caption{Baseline Comparison of Lightweight YOLO Models}",
        r"\label{tab:baseline}",
        r"\begin{tabular}{lcccccccccc}",
        r"\hline",
        (r"Model & mAP@0.5 & mAP@0.5:0.95 & Precision & Recall & "
         r"F1 & FPS & Inf.(ms) & Params(M) & GFLOPs \\"),
        r"\hline",
    ]
    for m in all_metrics:
        def f(k, fmt):
            v = m.get(k)
            return format(v, fmt) if isinstance(v, (int, float)) and v >= 0 else "--"
        lat_lines.append(
            f"{m['model']} & {f('mAP50','.4f')} & {f('mAP50_95','.4f')} & "
            f"{f('precision','.4f')} & {f('recall','.4f')} & "
            f"{f('f1','.4f')} & {f('fps','.1f')} & "
            f"{f('ms_per_img','.2f')} & {f('params_M','.2f')} & "
            f"{f('gflops','.2f')} \\\\"
        )
    lat_lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    lat_path.write_text("\n".join(lat_lines), encoding="utf-8")
    print(f"  LaTeX saved: {lat_path}")


# ── MULTI-SEED TRAINING + EVALUATION ─────────────────────────────────────────
# Seed 42 is already done. SEEDS = [0, 21] are new runs.
# After all seeds, the seed-42 mAP50 is loaded from metrics.json and merged in.

def train_and_eval_multiseed(name: str, weights: str, master_log: dict) -> list:
    """
    Trains model with SEEDS [0,7,21,99], evaluates each, returns list of
    5 mAP50 values (seed-42 result prepended from existing metrics.json).
    """
    from ultralytics import YOLO

    seed_results = {}  # {seed: mAP50}

    # ── Load seed-42 result from the already-done metrics.json ───────────────
    mp = metrics_path(name)
    if mp.exists():
        with open(mp) as f:
            m42 = json.load(f)
        if "mAP50" in m42:
            seed_results[42] = m42["mAP50"]
            print(f"  Loaded seed=42 result for {name}: mAP50={m42['mAP50']}")
    else:
        print(f"  WARNING: metrics.json not found for {name} (seed=42). "
              f"Run baseline first or include seed 42 in SEEDS.")

    # ── New seeds ─────────────────────────────────────────────────────────────
    for seed in SEEDS:
        run_name = f"{name}_seed{seed}"
        log_key  = f"multiseed_{run_name}"

        # Skip if already done
        info = master_log.get(log_key, {})
        if info.get("status") == "success" and "mAP50" in info:
            seed_results[seed] = info["mAP50"]
            print(f"  {run_name} already done — mAP50={info['mAP50']}")
            continue

        print(f"\n  {'─'*58}")
        print(f"  MULTI-SEED RUN: {run_name}")
        print(f"  {'─'*58}")

        clear_gpu()
        run_save_dir = SAVE_ROOT / run_name
        run_save_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = YOLO(weights)
            save_dir = model_save_dir(name)
            last_pt = save_dir / "weights" / "last.pt"
            resume  = last_pt.exists() and info.get("train_status") != "success"

            model.train(
                data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                optimizer=OPTIMIZER, lr0=LR0, lrf=LRF, momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY, workers=WORKERS, device=DEVICE,
                patience=PATIENCE, seed=seed, amp=AMP,
                close_mosaic=CLOSE_MOSAIC,
                project=str(SAVE_ROOT), name=run_name,
                exist_ok=True, verbose=False, save=True,
                resume=resume,
            )

            best_pt = run_save_dir / "weights" / "best.pt"
            if not best_pt.exists():
                raise FileNotFoundError(f"best.pt not found at {best_pt}")

            val_model = YOLO(str(best_pt))
            val = val_model.val(
                data=DATA_YAML, split="test", imgsz=IMGSZ, batch=BATCH,
                conf=0.25, iou=0.5, device=DEVICE, workers=WORKERS,
                verbose=False,
            )
            map50 = round(float(val.box.map50), 4)
            seed_results[seed] = map50

            master_log[log_key] = {
                "status" : "success",
                "model"  : name,
                "seed"   : seed,
                "mAP50"  : map50,
                "run_dir": str(run_save_dir),
                "time"   : datetime.now().isoformat(),
            }
            save_master_log(master_log)
            print(f"  {run_name} → mAP50={map50}")

        except Exception as e:
            print(f"  ERROR in {run_name}: {e}")
            master_log[log_key] = {
                "status": f"error: {e}",
                "model" : name,
                "seed"  : seed,
            }
            save_master_log(master_log)

        finally:
            clear_gpu()

    # Return sorted by seed for reproducibility
    ordered = [seed_results[s] for s in sorted(seed_results.keys())
               if s in seed_results]
    return ordered


# ── STATISTICAL ANALYSIS ──────────────────────────────────────────────────────

def run_statistics(multi_seed_results: dict):

    try:
        from scipy import stats
        import numpy as np
        from itertools import combinations
    except ImportError:
        print("\n  ERROR: scipy not installed. Run: pip install scipy")
        return

    print("\n\n" + "="*62)
    print("  STATISTICAL HYPOTHESIS TESTING")
    print("="*62)

    # ── 1. Summary stats ──────────────────────────────────────────────────────
    print(f"\n  {'Model':<12} {'N':>3} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-"*50)
    for name, scores in multi_seed_results.items():
        if not scores:
            print(f"  {name:<12} {'NO DATA':>3}")
            continue
        arr = np.array(scores)
        print(f"  {name:<12} {len(arr):>3} {arr.mean():>8.4f} "
              f"{arr.std(ddof=1):>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")

    # ── 2. Shapiro-Wilk normality test ────────────────────────────────────────
    print("\n  Shapiro-Wilk Normality Test (p > 0.05 → normal):")
    print("  " + "-"*40)
    for name, scores in multi_seed_results.items():
        if len(scores) < 3:
            print(f"  {name:<12} insufficient data")
            continue
        _, p = stats.shapiro(scores)
        flag = "NORMAL    " if p > 0.05 else "NOT NORMAL"
        print(f"  {name:<12}  p = {p:.4f}  → {flag}")

    # ── 3. Kruskal-Wallis (overall) ───────────────────────────────────────────
    valid = {k: v for k, v in multi_seed_results.items() if len(v) >= 3}
    if len(valid) < 2:
        print("\n  Not enough valid models for Kruskal-Wallis test.")
        return

    kw_stat, kw_p = stats.kruskal(*valid.values())
    print(f"\n  Kruskal-Wallis H-test across all models:")
    print(f"    H = {kw_stat:.4f},  p = {kw_p:.4f}")
    if kw_p < 0.05:
        print("    → At least one model is significantly different (p < 0.05)")
        print("    → Proceeding to pairwise post-hoc tests...")
    else:
        print("    → No significant overall difference found (p >= 0.05)")
        print("    → Pairwise tests shown below for completeness.")

    # ── 4. Pairwise Mann-Whitney U with Bonferroni correction ─────────────────
    model_names = list(valid.keys())
    pairs       = list(combinations(model_names, 2))
    alpha_adj   = 0.05 / len(pairs)   # Bonferroni

    print(f"\n  Pairwise Mann-Whitney U  (Bonferroni-corrected α = {alpha_adj:.4f}):")
    print(f"  {'Pair':<26} {'U':>8} {'p-value':>10} {'Sig':>5} {'Cohen d':>9} {'Effect':>10}")
    print("  " + "-"*72)

    rows = []
    for m1, m2 in pairs:
        a  = np.array(valid[m1])
        b  = np.array(valid[m2])
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')

        # Cohen's d
        pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
        d      = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

        sig = "*" if p < alpha_adj else "ns"

        abs_d = abs(d)
        if   abs_d < 0.2: effect = "negligible"
        elif abs_d < 0.5: effect = "small"
        elif abs_d < 0.8: effect = "medium"
        else:             effect = "large"

        pair_str = f"{m1} vs {m2}"
        print(f"  {pair_str:<26} {u:>8.1f} {p:>10.4f} {sig:>5} {d:>9.3f} {effect:>10}")
        rows.append({
            "pair"       : pair_str,
            "U_statistic": round(u, 2),
            "p_value"    : round(p, 4),
            "significant": sig,
            "cohens_d"   : round(d, 3),
            "effect_size": effect,
        })

    print(f"\n  * = significant at Bonferroni-corrected α={alpha_adj:.4f}")
    print(f"  Cohen's d effect: <0.2 negligible, 0.2–0.5 small, "
          f"0.5–0.8 medium, >0.8 large")

    # ── 5. Save results ───────────────────────────────────────────────────────
    stat_csv = SAVE_ROOT / "statistical_results.csv"
    with open(stat_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["pair","U_statistic","p_value",
                           "significant","cohens_d","effect_size"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {stat_csv}")

    # ── 6. Multi-seed summary CSV ─────────────────────────────────────────────
    seed_csv = SAVE_ROOT / "multiseed_results.csv"
    all_seeds = sorted({42} | set(SEEDS))
    with open(seed_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + [f"seed_{s}" for s in all_seeds] + ["mean", "std"])
        for name, scores in multi_seed_results.items():
            arr  = np.array(scores)
            row  = [name] + [round(v, 4) for v in scores]
            row += [round(arr.mean(), 4), round(arr.std(ddof=1), 4)]
            writer.writerow(row)
    print(f"  Saved: {seed_csv}")

    # ── 7. LaTeX stats table ──────────────────────────────────────────────────
    lat_stat = SAVE_ROOT / "latex_stats_table.tex"
    lat_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Statistical Comparison of YOLO Models (mAP@0.5, $n=5$ runs)}",
        r"\label{tab:stats}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Model & Mean & Std & Min & Max \\",
        r"\hline",
    ]
    for name, scores in multi_seed_results.items():
        if not scores:
            continue
        arr = np.array(scores)
        lat_lines.append(
            f"{name} & {arr.mean():.4f} & {arr.std(ddof=1):.4f} & "
            f"{arr.min():.4f} & {arr.max():.4f} \\\\"
        )
    lat_lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    lat_stat.write_text("\n".join(lat_lines), encoding="utf-8")
    print(f"  Saved: {lat_stat}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def print_gpu_info():
    if torch.cuda.is_available():
        p     = torch.cuda.get_device_properties(0)
        total = p.total_memory / 1024**3
        print(f"  GPU  : {p.name}  ({total:.1f} GB VRAM)")
        if total < 6:
            print(f"  WARN : <6 GB VRAM — if OOM, reduce BATCH at top of script.")
    else:
        print("  GPU  : Not available — CPU mode (slow)")


def main():
    print("\n" + "="*62)
    print("  YOLO Baseline Study — Train + Evaluate + Multi-Seed + Stats")
    print("="*62)
    print_gpu_info()
    print(f"\n  Models      : {[m[0] for m in MODELS]}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch       : {BATCH}")
    print(f"  Image size  : {IMGSZ}")
    print(f"  Seeds (new) : {SEEDS}  (+42 already done)")
    print(f"  Output      : {SAVE_ROOT}")
    print("="*62)

    master_log  = load_master_log()
    all_metrics = []

    # ── PHASE 1: Baseline (seed=42) — already done, just load metrics ────────
    print("\n\n" + "#"*62)
    print("  PHASE 1: Baseline Evaluation (seed=42)")
    print("#"*62)

    for name, weights, desc in MODELS:
        print(f"\n  MODEL: {name}  —  {desc}")

        train_ok = train_model(name, weights, master_log)
        if not train_ok:
            print(f"  Skipping evaluation for {name} (training failed).")
            continue

        metrics = evaluate_model(name, master_log)
        if metrics:
            all_metrics.append(metrics)

        export_tflite(name, master_log)

        print(f"\n  Updating comparison table ({len(all_metrics)} model(s))...")
        build_comparison_table(all_metrics)

    # ── PHASE 2: Multi-seed runs (seeds 0, 7, 21, 99) ────────────────────────
    print("\n\n" + "#"*62)
    print("  PHASE 2: Multi-Seed Runs for Statistical Testing")
    print(f"  New seeds: {SEEDS}  (seed=42 loaded from Phase 1 results)")
    print("#"*62)

    multi_seed_results = {}   # {model_name: [mAP50_s42, mAP50_s0, ...]}

    for name, weights, desc in MODELS:
        print(f"\n\n  {'='*58}")
        print(f"  Multi-seed: {name}  —  {desc}")
        print(f"  {'='*58}")

        scores = train_and_eval_multiseed(name, weights, master_log)
        multi_seed_results[name] = scores

        print(f"\n  {name} seed results: {scores}")
        if scores:
            import numpy as np
            arr = np.array(scores)
            print(f"  Mean={arr.mean():.4f}  Std={arr.std(ddof=1):.4f}")

    # Save multi-seed results to master log for reference
    master_log["multiseed_summary"] = {
        k: v for k, v in multi_seed_results.items()
    }
    save_master_log(master_log)

    # ── PHASE 3: Statistical Analysis ────────────────────────────────────────
    print("\n\n" + "#"*62)
    print("  PHASE 3: Statistical Hypothesis Testing")
    print("#"*62)

    run_statistics(multi_seed_results)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*62}")
    print("  ALL DONE")
    print("="*62)

    log = load_master_log()
    print(f"\n  {'Model':<12} {'Train':>6} {'Eval':>6} {'TFLite':>8} {'Min':>6}")
    print("  " + "-"*44)
    for name, _, _ in MODELS:
        info = log.get(name, {})
        t_ok = "OK" if info.get("train_status")  == "success" else "FAIL"
        e_ok = "OK" if info.get("eval_status")   == "success" else "FAIL"
        f_ok = "OK" if info.get("tflite_status") == "success" else "----"
        mins = info.get("training_time_min", "?")
        print(f"  {name:<12} {t_ok:>6} {e_ok:>6} {f_ok:>8} {mins:>6}")

    print(f"\n  Output folder : {SAVE_ROOT}")
    print(f"  comparison_table.csv      — baseline results")
    print(f"  comparison_table.txt      — formatted for paper")
    print(f"  latex_table.tex           — baseline LaTeX table")
    print(f"  multiseed_results.csv     — per-seed mAP50 values")
    print(f"  statistical_results.csv   — pairwise p-values + Cohen's d")
    print(f"  latex_stats_table.tex     — stats LaTeX table")
    print(f"  master_log.json           — full run log")


if __name__ == "__main__":
    main()