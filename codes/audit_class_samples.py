"""
audit_new_test_labels.py
===========================
Visual + statistical audit of I:\\Project_Conference\\new_test\\image and
\\labels — a flat folder pair, not the images/<split> structure used
elsewhere in this project.

What it does:
  1. Scans every label file and reports:
       - per-class instance counts across the whole new_test set
       - images with a label file but zero boxes (background)
       - images with NO label file at all (unlabeled — flagged, not
         treated as background, since this is unexpected for a labeled
         test set)
       - label files with no matching image (orphaned labels)
  2. Builds a grid image of a random sample with all GT boxes drawn, so
     you can eyeball box quality/placement — same idea as
     audit_class_samples.py, but works on this flat folder and can show
     ALL classes at once instead of filtering to one.

Run:
    python audit_new_test_labels.py
Then open the saved grid: I:\\Project_Conference\\new_test\\audit_grid.jpg
"""

import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGES_DIR = Path(r"I:\Project_Conference\21431547\United_States\train\images")
LABELS_DIR = Path(r"I:\Project_Conference\21431547\United_States\train\labels")

# None -> sample across ALL classes. Set to a class id (0-4) to only
# sample images that contain that class.
TARGET_CLASS_ID = None

NUM_SAMPLES = 16
GRID_COLS   = 4
THUMB_SIZE  = 360
SEED        = 42

OUTPUT_PATH = LABELS_DIR.parent / "audit_grid.jpg"

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Speed Breaker",
}
CLASS_COLORS = {
    0: (0, 200, 255),
    1: (255, 215, 0),
    2: (255, 107, 53),
    3: (168, 85, 247),
    4: (34, 197, 94),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ─────────────────────────────────────────────────────────────────────────────


def find_image_path(stem: str):
    for ext in IMG_EXTS:
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_boxes(lbl_path: Path):
    boxes = []
    for line in lbl_path.read_text(errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:])
            boxes.append((cls_id, cx, cy, w, h))
        except ValueError:
            continue
    return boxes


def draw_boxes(img: Image.Image, boxes, highlight_class):
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for cls_id, cx, cy, w, h in boxes:
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        is_target = (highlight_class is None) or (cls_id == highlight_class)
        width = 4 if is_target else 2

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        label = CLASS_NAMES.get(cls_id, str(cls_id))
        tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else len(label) * 8
        draw.rectangle([x1, max(0, y1 - 18), x1 + tw + 8, max(18, y1)], fill=color)
        draw.text((x1 + 4, max(0, y1 - 17)), label, fill=(0, 0, 0), font=font)

    return img


def build_grid(thumbnails, cols, thumb_size):
    n = len(thumbnails)
    rows = (n + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (20, 20, 30))

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (thumb, caption) in enumerate(thumbnails):
        r, c = divmod(i, cols)
        thumb = thumb.copy()
        thumb.thumbnail((thumb_size, thumb_size - 24))
        x = c * thumb_size + (thumb_size - thumb.width) // 2
        y = r * thumb_size
        grid.paste(thumb, (x, y))

        draw = ImageDraw.Draw(grid)
        draw.text((c * thumb_size + 4, y + thumb_size - 20), caption,
                  fill=(230, 230, 230), font=font)

    return grid


def main():
    print("\n" + "=" * 70)
    print("  NEW_TEST LABEL AUDIT")
    print("=" * 70)
    print(f"  Images : {IMAGES_DIR}")
    print(f"  Labels : {LABELS_DIR}")

    if not IMAGES_DIR.exists():
        print(f"\n  ERROR: images folder not found: {IMAGES_DIR}")
        return
    if not LABELS_DIR.exists():
        print(f"\n  ERROR: labels folder not found: {LABELS_DIR}")
        return

    image_paths = sorted(p for p in IMAGES_DIR.iterdir()
                          if p.is_file() and p.suffix.lower() in IMG_EXTS)
    label_paths = sorted(LABELS_DIR.glob("*.txt"))
    image_stems = {p.stem for p in image_paths}
    label_stems = {p.stem for p in label_paths}

    print(f"\n  Images found       : {len(image_paths)}")
    print(f"  Label files found  : {len(label_paths)}")

    unlabeled = image_stems - label_stems
    orphaned  = label_stems - image_stems
    if unlabeled:
        print(f"  WARNING: {len(unlabeled)} image(s) have NO label file:")
        for s in sorted(list(unlabeled))[:10]:
            print(f"    - {s}")
        if len(unlabeled) > 10:
            print(f"    ... and {len(unlabeled) - 10} more")
    if orphaned:
        print(f"  WARNING: {len(orphaned)} label file(s) have NO matching image:")
        for s in sorted(list(orphaned))[:10]:
            print(f"    - {s}")
        if len(orphaned) > 10:
            print(f"    ... and {len(orphaned) - 10} more")

    # ── Per-class instance counts + background count ──────────────────────────
    class_counts = Counter()
    background_count = 0
    stem_to_boxes = {}

    for lbl_path in label_paths:
        boxes = read_boxes(lbl_path)
        stem_to_boxes[lbl_path.stem] = boxes
        if not boxes:
            background_count += 1
        for cls_id, *_ in boxes:
            class_counts[cls_id] += 1

    print(f"\n  Background labels (empty .txt): {background_count}")
    print(f"\n  Per-class instance counts:")
    for cls_id in sorted(CLASS_NAMES):
        print(f"    {cls_id} {CLASS_NAMES[cls_id]:<20}: {class_counts.get(cls_id, 0)}")
    unknown_classes = set(class_counts) - set(CLASS_NAMES)
    if unknown_classes:
        print(f"  WARNING: unexpected class ids found: {sorted(unknown_classes)}")

    # ── Build sample grid ───────────────────────────────────────────────────────
    if TARGET_CLASS_ID is None:
        candidate_stems = [s for s, b in stem_to_boxes.items() if b]
    else:
        candidate_stems = [s for s, b in stem_to_boxes.items()
                           if any(cid == TARGET_CLASS_ID for cid, *_ in b)]

    if not candidate_stems:
        print("\n  No labeled images available to sample for the grid.")
        return

    rng = random.Random(SEED)
    sample = rng.sample(candidate_stems, min(NUM_SAMPLES, len(candidate_stems)))

    thumbnails = []
    for stem in sample:
        img_path = find_image_path(stem)
        if img_path is None:
            print(f"  WARNING: image not found for label {stem}")
            continue
        img = Image.open(img_path)
        boxes = stem_to_boxes[stem]
        annotated = draw_boxes(img, boxes, TARGET_CLASS_ID)
        thumbnails.append((annotated, f"{stem} ({len(boxes)} box{'es' if len(boxes) != 1 else ''})"))

    grid = build_grid(thumbnails, GRID_COLS, THUMB_SIZE)
    grid.save(OUTPUT_PATH, quality=90)

    print(f"\n  Saved audit grid: {OUTPUT_PATH}")
    print("  Look for: boxes on the right object, consistent box tightness,")
    print("  and whether this imagery matches your training data's style/scale.")


if __name__ == "__main__":
    main()