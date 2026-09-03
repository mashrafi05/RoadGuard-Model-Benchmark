"""
03_results_table.py
===================
Reads runs/evaluation/all_metrics.csv and prints:
  1. A formatted comparison table (ready to paste into your paper)
  2. Which model wins each metric
  3. How much YOLO-Nano improves over each baseline

Run after 02_evaluate_all_models.py
    python 03_results_table.py
"""

import csv
import json
from pathlib import Path

RESULTS_CSV  = Path("runs/evaluation/all_metrics.csv")
PROPOSED     = "YOLO-Nano"   # your model name — must match 01_train_all_models.py

METRICS = [
    ("precision",    "Precision",       True,  ".4f"),  # (key, label, higher=better, fmt)
    ("recall",       "Recall",          True,  ".4f"),
    ("f1_score",     "F1-score",        True,  ".4f"),
    ("mAP_50",       "mAP@0.5",         True,  ".4f"),
    ("mAP_50_95",    "mAP@0.5:0.95",   True,  ".4f"),
    ("fps",          "FPS",             True,  ".1f"),
    ("inference_ms", "Inf. time (ms)",  False, ".1f"),  # lower is better
    ("params_M",     "Params (M)",      False, ".2f"),
    ("model_size_mb","Size (MB)",       False, ".2f"),
]


def load_results() -> list[dict]:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_CSV}\n"
            "Run 02_evaluate_all_models.py first."
        )
    results = []
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            # Convert numeric fields
            for key in ["precision","recall","f1_score","mAP_50","mAP_50_95",
                        "fps","inference_ms","params_M","model_size_mb"]:
                try:
                    row[key] = float(row[key])
                except (ValueError, KeyError):
                    row[key] = None
            results.append(row)
    return results


def best_value(results: list[dict], key: str, higher_better: bool):
    vals = [r[key] for r in results if r.get(key) is not None]
    return max(vals) if higher_better else min(vals)


def print_table(results: list[dict]):
    models = [r["model"] for r in results]
    col_w  = max(len(m) for m in models) + 2  # column width

    # Header
    metric_labels = [m[1] for m in METRICS]
    header_row = f"{'Model':<{col_w}}" + "  ".join(f"{lbl:>16}" for lbl in metric_labels)
    print("\n" + "="*len(header_row))
    print("  BASELINE STUDY — Results Table")
    print("="*len(header_row))
    print(f"{'Model':<{col_w}}" + "  ".join(f"{lbl:>16}" for lbl in metric_labels))
    print("─"*len(header_row))

    for r in results:
        row = f"{r['model']:<{col_w}}"
        for key, label, higher_better, fmt in METRICS:
            val = r.get(key)
            best = best_value(results, key, higher_better)
            if val is None:
                cell = f"{'N/A':>16}"
            else:
                cell_str = format(val, fmt)
                # Mark best value with *
                if val == best:
                    cell_str = f"*{cell_str}"
                cell = f"{cell_str:>16}"
            row += "  " + cell
        # Highlight proposed model
        marker = "  ← proposed" if r["model"] == PROPOSED else ""
        print(row + marker)

    print("─"*len(header_row))
    print("* = best in column")


def improvement_summary(results: list[dict]):
    proposed = next((r for r in results if r["model"] == PROPOSED), None)
    if not proposed:
        print(f"\n⚠ Proposed model '{PROPOSED}' not found in results.")
        return

    print(f"\n{'='*55}")
    print(f"  YOLO-Nano improvement over baselines")
    print(f"{'='*55}")

    baselines = [r for r in results if r["model"] != PROPOSED]
    key_metrics = [("mAP_50","mAP@0.5",True), ("fps","FPS",True),
                   ("f1_score","F1",True), ("inference_ms","Inf.time",False)]

    for baseline in baselines:
        print(f"\n  vs {baseline['model']}:")
        for key, label, higher_better in key_metrics:
            p_val = proposed.get(key)
            b_val = baseline.get(key)
            if p_val is None or b_val is None or b_val == 0:
                continue
            delta = p_val - b_val
            pct   = (delta / b_val) * 100
            sign  = "+" if delta > 0 else ""
            better = (higher_better and delta > 0) or (not higher_better and delta < 0)
            symbol = "✓" if better else "✗"
            print(f"    {symbol} {label:<15}: {sign}{delta:+.4f}  ({sign}{pct:+.1f}%)")


def latex_table(results: list[dict]):
    """Generate a LaTeX table snippet for your conference paper."""
    lines = [
        "% ── Paste this into your paper ────────────────────────",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Baseline Study Results}",
        r"\label{tab:baseline}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Model & Precision & Recall & F1 & mAP@0.5 & FPS & Inf.(ms) \\",
        r"\hline",
    ]

    for r in results:
        p  = r.get("precision",    "─")
        re = r.get("recall",       "─")
        f1 = r.get("f1_score",     "─")
        m  = r.get("mAP_50",       "─")
        fps= r.get("fps",          "─")
        it = r.get("inference_ms", "─")
        name = r["model"]
        if name == PROPOSED:
            name = r"\textbf{" + name + "}"   # bold for proposed

        def fmt(v, f=".4f"):
            return format(v, f) if isinstance(v, float) else str(v)

        lines.append(
            f"{name} & {fmt(p)} & {fmt(re)} & {fmt(f1)} & "
            f"{fmt(m)} & {fmt(fps,'.1f')} & {fmt(it,'.1f')} \\\\"
        )

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "% ──────────────────────────────────────────────────────",
    ]

    latex_path = Path("runs/evaluation/latex_table.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n✓ LaTeX table saved: {latex_path}")
    print("\n" + "\n".join(lines))


def main():
    results = load_results()
    print_table(results)
    improvement_summary(results)
    latex_table(results)

    print("\n✓ Done. Files in runs/evaluation/")
    print("  all_metrics.csv  — raw numbers")
    print("  latex_table.tex  — paste into your paper")


if __name__ == "__main__":
    main()
