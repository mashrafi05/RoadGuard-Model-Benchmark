"""
convert_voc_to_yolo.py
========================
Converts Pascal VOC XML annotations (RDD-style, e.g. the Japan Road
Damage Dataset) into YOLO .txt label files, using the class ids already
used elsewhere in this project:

    0 = Longitudinal Crack  (D00)
    1 = Transverse Crack    (D10)
    2 = Alligator Crack     (D20)
    3 = Pothole             (D40)

Any other damage code found in the XML (D0w0, D01, D11, D43, D44, D50,
etc. — common in the full RDD dataset) is NOT one of your four target
classes, so it is skipped and counted, not mapped to a wrong id.

Input:
    I:\\Project_Conference\\21431547\\Japan\\train\\annotations\\xmls\\*.xml
    I:\\Project_Conference\\21431547\\Japan\\train\\images\\*.jpg

Output:
    I:\\Project_Conference\\21431547\\Japan\\train\\labels\\*.txt
    (one .txt per image, same stem as the image filename — ready to be
    merged into your main dataset the same way as the speed-breaker data)

Run:
    python convert_voc_to_yolo.py
"""

import csv
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"I:/Project_Conference/21431547/United_States/train")
XML_DIR     = BASE_DIR / "annotations" / "xmls"
IMAGES_DIR  = BASE_DIR / "images"
LABELS_DIR  = BASE_DIR / "labels"

# RDD damage code -> project class id (must match CLASS_NAMES elsewhere)
CLASS_MAP = {
    "D00": 0,   # Longitudinal Crack
    "D10": 1,   # Transverse Crack
    "D20": 2,   # Alligator Crack
    "D40": 3,   # Pothole
}
CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ─────────────────────────────────────────────────────────────────────────────


def get_image_size(xml_root, xml_path: Path):
    """Prefer <size> from the XML; fall back to opening the actual image
    if the XML size is missing or zero."""
    size_el = xml_root.find("size")
    if size_el is not None:
        try:
            w = int(float(size_el.findtext("width", "0")))
            h = int(float(size_el.findtext("height", "0")))
            if w > 0 and h > 0:
                return w, h
        except ValueError:
            pass

    # Fallback: open the actual image
    filename = xml_root.findtext("filename", "").strip()
    stem = Path(filename).stem if filename else xml_path.stem
    for ext in IMG_EXTS:
        candidate = IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists() and PIL_AVAILABLE:
            with Image.open(candidate) as im:
                return im.size  # (w, h)
    return None, None


def convert_one(xml_path: Path, stats: dict):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        stats["parse_errors"] += 1
        print(f"  ERROR parsing {xml_path.name}: {e}")
        return

    filename = root.findtext("filename", "").strip()
    stem = Path(filename).stem if filename else xml_path.stem

    if not (IMAGES_DIR / f"{stem}.jpg").exists():
        # try any known extension before flagging missing
        if not any((IMAGES_DIR / f"{stem}{ext}").exists() for ext in IMG_EXTS):
            stats["missing_images"] += 1

    img_w, img_h = get_image_size(root, xml_path)
    if not img_w or not img_h:
        stats["size_unknown"] += 1
        print(f"  SKIP {xml_path.name}: could not determine image size.")
        return

    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()

        if name not in CLASS_MAP:
            stats["other_classes"][name] += 1
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            stats["missing_bndbox"] += 1
            continue

        try:
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            stats["invalid_boxes"] += 1
            continue

        # Clamp to image bounds
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            stats["invalid_boxes"] += 1
            continue

        cls_id = CLASS_MAP[name]
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        stats["class_counts"][cls_id] += 1

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABELS_DIR / f"{stem}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    stats["files_written"] += 1
    if lines:
        stats["files_with_boxes"] += 1
    else:
        stats["files_background"] += 1


def main():
    print("\n" + "=" * 70)
    print("  VOC XML -> YOLO TXT CONVERSION")
    print("=" * 70)
    print(f"  XML dir    : {XML_DIR}")
    print(f"  Images dir : {IMAGES_DIR}")
    print(f"  Labels out : {LABELS_DIR}")

    if not XML_DIR.exists():
        print(f"\n  ERROR: XML folder not found: {XML_DIR}")
        return

    xml_files = sorted(XML_DIR.glob("*.xml"))
    print(f"\n  Found {len(xml_files)} XML files.")
    if not xml_files:
        return

    stats = {
        "files_written": 0,
        "files_with_boxes": 0,
        "files_background": 0,
        "parse_errors": 0,
        "missing_images": 0,
        "size_unknown": 0,
        "missing_bndbox": 0,
        "invalid_boxes": 0,
        "class_counts": Counter(),
        "other_classes": Counter(),
    }

    for i, xml_path in enumerate(xml_files, 1):
        convert_one(xml_path, stats)
        if i % 1000 == 0:
            print(f"  ... {i}/{len(xml_files)} processed")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CONVERSION SUMMARY")
    print("=" * 70)
    print(f"  XML files processed   : {len(xml_files)}")
    print(f"  Label files written   : {stats['files_written']}")
    print(f"    with >=1 target box : {stats['files_with_boxes']}")
    print(f"    background (empty)  : {stats['files_background']}")
    print(f"  Parse errors           : {stats['parse_errors']}")
    print(f"  Missing image files    : {stats['missing_images']}")
    print(f"  Size undetermined      : {stats['size_unknown']}")
    print(f"  Missing <bndbox>       : {stats['missing_bndbox']}")
    print(f"  Invalid/degenerate box : {stats['invalid_boxes']}")

    print(f"\n  Target class instance counts:")
    for cls_id, name in CLASS_NAMES.items():
        print(f"    {cls_id} {name:<20}: {stats['class_counts'].get(cls_id, 0)}")

    if stats["other_classes"]:
        print(f"\n  Non-target damage codes skipped (not in your 4 classes):")
        for code, cnt in sorted(stats["other_classes"].items(),
                                 key=lambda x: -x[1]):
            print(f"    {code:<10}: {cnt}")

    # ── CSV summary ───────────────────────────────────────────────────────────
    csv_path = BASE_DIR / "conversion_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "instance_count"])
        for cls_id, name in CLASS_NAMES.items():
            writer.writerow([cls_id, name, stats["class_counts"].get(cls_id, 0)])
        writer.writerow([])
        writer.writerow(["skipped_damage_code", "count"])
        for code, cnt in sorted(stats["other_classes"].items(), key=lambda x: -x[1]):
            writer.writerow([code, cnt])
        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerow(["xml_files", len(xml_files)])
        writer.writerow(["labels_written", stats["files_written"]])
        writer.writerow(["files_with_boxes", stats["files_with_boxes"]])
        writer.writerow(["files_background", stats["files_background"]])
        writer.writerow(["parse_errors", stats["parse_errors"]])
        writer.writerow(["missing_images", stats["missing_images"]])
        writer.writerow(["size_unknown", stats["size_unknown"]])
        writer.writerow(["invalid_boxes", stats["invalid_boxes"]])

    print(f"\n  CSV summary saved: {csv_path}")
    print(f"  YOLO labels saved to: {LABELS_DIR}")


if __name__ == "__main__":
    main()
