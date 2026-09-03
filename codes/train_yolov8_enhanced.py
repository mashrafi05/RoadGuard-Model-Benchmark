"""
train_yolov8_enhanced.py
─────────────────────────────────────────────────────────────────────────────
Paper 2: Enhanced YOLOv8
Modifications: DSConv + SimAM + GELU

Run independently (can run in parallel with train_yolo_roc.py):
    python train_yolov8_enhanced.py

NOTE: Uses GPU 0. If running both scripts simultaneously on separate GPUs,
      change DEVICE in shared_modules.py per script, or pass via CLI.

Evaluation is NOT done here. Run evaluate_external.py after both scripts finish.
"""
import torch
import torch.nn as nn
from shared_modules import (
    DSConv, SimAM,
    SAVE_ROOT, LOG_PATH,
    get_conv_channels,
    load_log, save_log,
    print_model_layers,
    train_external,
)

RUN_NAME = "YOLOv8_DSConv_SimAM_GELU"


# ═════════════════════════════════════════════════════════════════════════════
#  BUILD ENHANCED YOLOv8
# ═════════════════════════════════════════════════════════════════════════════

def build_yolov8_enhanced():
    """
    Load YOLOv8n and apply Paper 2 modifications:
    1. Replace SiLU → GELU in all activation layers
    2. Inject SimAM after C2f blocks (backbone + neck, skip first C2f)
    3. Replace Conv → DSConv in backbone (skip first Conv layer)

    Key fix: ultralytics Conv is a wrapper — we replace the whole wrapper.
    Channel count lives at module.conv.in_channels (inner nn.Conv2d).
    """
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    # ── Step 1: Replace SiLU → GELU ───────────────────────────────────────────
    silu_count = 0
    for name, module in model.model.named_modules():
        if isinstance(module, nn.SiLU):
            parts  = name.split(".")
            parent = model.model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], nn.GELU())
            silu_count += 1
    print(f"  [Enhanced] Replaced {silu_count} SiLU → GELU")

    # ── Step 2: Inject SimAM after C2f blocks (skip first) ───────────────────
    c2f_count        = 0
    c2f_skipped      = 0
    c2f_replacements = []

    for name, module in model.model.named_modules():
        if type(module).__name__ == "C2f":
            if c2f_skipped == 0:
                c2f_skipped += 1
                print(f"  [Enhanced] Skipping first C2f at '{name}' (low-level features)")
                continue
            c2f_replacements.append(name)

    for name in c2f_replacements:
        parts  = name.split(".")
        parent = model.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        orig = getattr(parent, attr)

        class C2fWithSimAM(nn.Module):
            def __init__(self, c2f_module):
                super().__init__()
                self.c2f   = c2f_module
                self.simam = SimAM()
            def forward(self, x):
                return self.simam(self.c2f(x))

        setattr(parent, attr, C2fWithSimAM(orig))
        c2f_count += 1

    print(f"  [Enhanced] Injected SimAM after {c2f_count} C2f blocks")

    # ── Step 3: Replace Conv → DSConv (backbone only, skip first Conv) ────────
    conv_count        = 0
    first_conv        = True
    conv_replacements = []

    for name, module in model.model.named_modules():
        if type(module).__name__ != "Conv":
            continue

        if first_conv:
            first_conv = False
            print(f"  [Enhanced] Skipping first Conv at '{name}'")
            continue

        # Only replace backbone layers (index < 10 in YOLOv8n)
        try:
            layer_idx = int(name.split(".")[2])
            if layer_idx >= 10:
                continue
        except (IndexError, ValueError):
            continue

        in_ch, out_ch = get_conv_channels(
            module.conv if hasattr(module, "conv") else module
        )
        if in_ch is None:
            continue

        conv_replacements.append((name, in_ch, out_ch))

    for name, in_ch, out_ch in conv_replacements:
        parts  = name.split(".")
        parent = model.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        try:
            setattr(parent, parts[-1], DSConv(in_ch, out_ch))
            conv_count += 1
        except Exception as e:
            print(f"  [Enhanced] Skipped Conv at {name}: {e}")

    print(f"  [Enhanced] Replaced {conv_count} Conv → DSConv in backbone")

    total_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"  [Enhanced] Done. Total params: {total_params:.3f}M")

    return model


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*62)
    print("  EXTERNAL BASELINE 2: Enhanced YOLOv8")
    print("  Modifications: DSConv + SimAM + GELU")
    print("="*62)

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}  ({p.total_memory/1024**3:.1f} GB)")
    else:
        print("  GPU: Not available — CPU mode (slow)")

    log   = load_log()
    model = build_yolov8_enhanced()
    ok    = train_external(RUN_NAME, model, log)

    if ok:
        print(f"\n  Training complete. Run evaluate_external.py to get metrics.")
    else:
        print(f"\n  Training failed. Check {LOG_PATH} for error details.")

    print(f"\n  Output dir : {SAVE_ROOT / RUN_NAME}")
    print(f"  Log file   : {LOG_PATH}")


if __name__ == "__main__":
    main()
