"""
train_yolo_roc.py
─────────────────────────────────────────────────────────────────────────────
Paper 1: YOLO-ROC
Modification: Replace SPPF → BMS-SPPF (MSSA + CAP + MHSA)

Run independently (can run in parallel with train_yolov8_enhanced.py):
    python train_yolo_roc.py

NOTE: Uses GPU 0. If running both scripts simultaneously on separate GPUs,
      change DEVICE in shared_modules.py per script, or pass via CLI.

Evaluation is NOT done here. Run evaluate_external.py after both scripts finish.
"""
  
import torch
from shared_modules import (
    BMS_SPPF,
    SAVE_ROOT, LOG_PATH,
    load_log, save_log,
    print_model_layers,
    train_external,
)

RUN_NAME = "YOLO_ROC"


# ═════════════════════════════════════════════════════════════════════════════
#  BUILD YOLO-ROC
# ═════════════════════════════════════════════════════════════════════════════

def build_yolo_roc():
    """
    Load YOLOv8n and replace SPPF → BMS_SPPF.

    Key fix: ultralytics Conv stores the actual nn.Conv2d inside .conv
    So channel count is at  module.cv1.conv.in_channels  (not module.cv1.in_channels)
    """
    from ultralytics import YOLO

    model    = YOLO("yolov8n.pt")
    replaced = 0

    replacements = []
    for name, module in model.model.named_modules():
        if type(module).__name__ == "SPPF":
            in_ch  = module.cv1.conv.in_channels   # ← ultralytics Conv wrapper fix
            out_ch = module.cv2.conv.out_channels
            replacements.append((name, in_ch, out_ch))

    for name, in_ch, out_ch in replacements:
        parts  = name.split(".")
        parent = model.model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        setattr(parent, parts[-1], BMS_SPPF(in_ch, out_ch))
        replaced += 1
        print(f"  [YOLO-ROC] Replaced SPPF → BMS_SPPF at '{name}' "
              f"(in={in_ch}, out={out_ch})")

    if replaced == 0:
        print("  WARNING: No SPPF found. Printing layers for inspection:")
        print_model_layers(model)
    else:
        total_params = sum(p.numel() for p in model.model.parameters()) / 1e6
        print(f"  [YOLO-ROC] Done. Total params: {total_params:.3f}M")

    return model


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*62)
    print("  EXTERNAL BASELINE 1: YOLO-ROC")
    print("  Modification: BMS-SPPF (replaces SPPF)")
    print("="*62)

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}  ({p.total_memory/1024**3:.1f} GB)")
    else:
        print("  GPU: Not available — CPU mode (slow)")

    log   = load_log()
    model = build_yolo_roc()
    ok    = train_external(RUN_NAME, model, log)

    if ok:
        print(f"\n  Training complete. Run evaluate_external.py to get metrics.")
    else:
        print(f"\n  Training failed. Check {LOG_PATH} for error details.")

    print(f"\n  Output dir : {SAVE_ROOT / RUN_NAME}")
    print(f"  Log file   : {LOG_PATH}")


if __name__ == "__main__":
    main()
