"""
00_prepare_test_split.py
========================
Proper stratified 70:15:15 split for a 5-class YOLO dataset.

Fixes applied vs previous version
----------------------------------
- Uses sklearn-style per-class stratification: each class bucket is
  independently shuffled and sliced 70/15/15, then deduplicated.
- Two-phase write (temp folder) avoids reading from folders we already deleted.
- print_stats re-reads labels from the NEW final locations, not stale paths.

Run ONCE from I:/Project_Conference/
    python 00_prepare_test_split.py
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT        = Path("I:/Project_Conference")
SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Speed Breaker",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ─────────────────────────────────────────────────────────────────────────────


def find_label(img_path: Path) -> Path:
    """images/train/abc.jpg  →  labels/train/abc.txt"""
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
    except ValueError:
        return None
    parts[idx] = "labels"
    lbl = Path(*parts).with_suffix(".txt")
    return lbl if lbl.exists() else None


def get_classes(lbl_path: Path) -> set:
    classes = set()
    if lbl_path is None:
        return classes
    try:
        with open(lbl_path) as f:
            for line in f:
                s = line.strip()
                if s:
                    classes.add(int(s.split()[0]))
    except Exception:
        pass
    return classes


def collect_all() -> list:
    """Return list of (img_path, lbl_path_or_None, frozenset_of_classes)."""
    records = []
    img_root = ROOT / "images"
    for split in ("train", "val"):
        d = img_root / split
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                lbl = find_label(p)
                cls = frozenset(get_classes(lbl))
                records.append((p, lbl, cls))
    return records


def stratified_split(records: list):
    """
    True per-class stratified split.

    Algorithm
    ---------
    1. Build one bucket per class containing the indices of images that
       have that class.
    2. Shuffle each bucket with a fixed seed and slice 70/15/15.
    3. Union the train/val/test index sets across all classes.
    4. Any image not yet assigned (because it appeared in zero or
       low-frequency classes) is assigned proportionally by remainder.
    5. Images without labels are split proportionally and appended.

    This guarantees every class has ~70/15/15 representation.
    """
    random.seed(SEED)

    labelled   = [(i, r) for i, r in enumerate(records) if r[2]]
    unlabelled = [(i, r) for i, r in enumerate(records) if not r[2]]

    # Per-class index buckets (index into `labelled` list)
    class_buckets = defaultdict(list)
    for bucket_idx, (orig_idx, (img, lbl, cls)) in enumerate(labelled):
        for c in cls:
            class_buckets[c].append(bucket_idx)

    print(f"\n  Total images   : {len(records)}")
    print(f"  With labels    : {len(labelled)}")
    print(f"  Without labels : {len(unlabelled)}")
    print("\nPer-class image counts:")
    for c in sorted(CLASS_NAMES):
        print(f"  Class {c} {CLASS_NAMES[c]:<22}: {len(class_buckets[c])} images")

    # Stratify per class
    train_idx = set()
    val_idx   = set()
    test_idx  = set()

    for cls_id in sorted(class_buckets):
        bucket = class_buckets[cls_id].copy()
        random.shuffle(bucket)
        n       = len(bucket)
        n_train = round(n * TRAIN_RATIO)
        n_val   = round(n * VAL_RATIO)
        # test gets remainder so counts are exact
        n_test  = n - n_train - n_val

        train_idx.update(bucket[:n_train])
        val_idx.update(bucket[n_train : n_train + n_val])
        test_idx.update(bucket[n_train + n_val :])

    # An image assigned to multiple classes by different bucket slices
    # may appear in more than one set — resolve conflicts: train > val > test
    # (earlier split wins; this is deterministic because we sorted class_buckets)
    val_idx  -= train_idx
    test_idx -= train_idx
    test_idx -= val_idx

    # Assign remaining labelled images not yet in any set
    assigned = train_idx | val_idx | test_idx
    unassigned = [i for i, _ in enumerate(labelled) if i not in assigned]
    random.shuffle(unassigned)
    n_u = len(unassigned)
    n_u_train = round(n_u * TRAIN_RATIO)
    n_u_val   = round(n_u * VAL_RATIO)
    train_idx.update(unassigned[:n_u_train])
    val_idx.update(unassigned[n_u_train : n_u_train + n_u_val])
    test_idx.update(unassigned[n_u_train + n_u_val :])

    # Convert bucket indices back to records
    def idx_to_records(idx_set):
        return [labelled[i][1] for i in idx_set]

    train_recs = idx_to_records(train_idx)
    val_recs   = idx_to_records(val_idx)
    test_recs  = idx_to_records(test_idx)

    # Split unlabelled proportionally
    ul_imgs = [r for _, r in unlabelled]
    random.shuffle(ul_imgs)
    n_ul        = len(ul_imgs)
    n_ul_train  = round(n_ul * TRAIN_RATIO)
    n_ul_val    = round(n_ul * VAL_RATIO)
    n_ul_test   = n_ul - n_ul_train - n_ul_val
    train_recs += ul_imgs[:n_ul_train]
    val_recs   += ul_imgs[n_ul_train : n_ul_train + n_ul_val]
    test_recs  += ul_imgs[n_ul_train + n_ul_val :]

    print(f"\n  Unlabelled split: "
          f"train={n_ul_train}  val={n_ul_val}  test={n_ul_test}")

    return train_recs, val_recs, test_recs


def verify(train, val, test):
    t  = {r[0].stem for r in train}
    v  = {r[0].stem for r in val}
    ts = {r[0].stem for r in test}
    print("\n── Leak check ─────────────────────────────────────")
    ok = True
    for na, a, nb, b in [("train",t,"val",v),
                          ("train",t,"test",ts),
                          ("val",v,"test",ts)]:
        ov = a & b
        if ov:
            print(f"  X {na}/{nb} overlap: {len(ov)} images!")
            ok = False
        else:
            print(f"  OK  No {na}/{nb} overlap")
    return ok


def write_splits(train, val, test):
    """
    Phase 1: copy everything to _split_tmp/  (reads original files)
    Phase 2: move _split_tmp/ into place     (deletes old folders)
    """
    tmp = ROOT / "_split_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    split_map = {"train": train, "val": val, "test": test}

    print("\nPhase 1: copying to temp folder ...")
    for split_name, recs in split_map.items():
        ti = tmp / "images" / split_name
        tl = tmp / "labels" / split_name
        ti.mkdir(parents=True)
        tl.mkdir(parents=True)
        no_lbl = 0
        for img, lbl, cls in recs:
            shutil.copy2(img, ti / img.name)
            if lbl and lbl.exists():
                shutil.copy2(lbl, tl / (img.stem + ".txt"))
            else:
                no_lbl += 1
        tag = f"  ({no_lbl} unlabelled)" if no_lbl else ""
        print(f"  {split_name:<6}: {len(recs):>5} images{tag}")

    print("\nPhase 2: moving into place ...")
    for split_name in ("train", "val", "test"):
        for kind in ("images", "labels"):
            dest = ROOT / kind / split_name
            src  = tmp / kind / split_name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))

    shutil.rmtree(tmp)
    print("  Done.")


def print_stats(train, val, test):
    """
    Re-read labels from their NEW locations (ROOT/labels/<split>/)
    so counts are accurate after the move.
    """
    total = len(train) + len(val) + len(test)
    print("\n── Final split sizes ──────────────────────────────")
    for name, recs, tgt in [("train",train,TRAIN_RATIO),
                              ("val",val,VAL_RATIO),
                              ("test",test,TEST_RATIO)]:
        print(f"  {name:<6} {len(recs):>5} images  "
              f"({len(recs)/total:.1%}  target {tgt:.0%})")
    print(f"  {'TOTAL':<6} {total:>5} images")

    per_class = {c: {"train":0,"val":0,"test":0} for c in CLASS_NAMES}
    no_lbl    = {"train":0,"val":0,"test":0}

    for split_name, recs in [("train",train),("val",val),("test",test)]:
        lbl_dir = ROOT / "labels" / split_name
        for img, _, _ in recs:
            new_lbl = lbl_dir / (img.stem + ".txt")
            if new_lbl.exists():
                classes = get_classes(new_lbl)
                for c in classes:
                    if c in per_class:
                        per_class[c][split_name] += 1
            else:
                no_lbl[split_name] += 1

    print("\n── Per-class distribution ─────────────────────────")
    print(f"  {'Class':<26} {'Train':>6} {'Val':>6} {'Test':>6} "
          f"{'Total':>6}  Ratio")
    print(f"  {'─'*65}")
    for c in sorted(CLASS_NAMES):
        d = per_class[c]
        rt = d["train"] + d["val"] + d["test"]
        if rt:
            rs = (f"({d['train']/rt:.0%}/"
                  f"{d['val']/rt:.0%}/"
                  f"{d['test']/rt:.0%})")
        else:
            rs = "(no samples)"
        print(f"  {c}: {CLASS_NAMES[c]:<22} "
              f"{d['train']:>6} {d['val']:>6} {d['test']:>6} {rt:>6}  {rs}")

    nl_tot = sum(no_lbl.values())
    if nl_tot:
        rs = (f"({no_lbl['train']/nl_tot:.0%}/"
              f"{no_lbl['val']/nl_tot:.0%}/"
              f"{no_lbl['test']/nl_tot:.0%})")
        print(f"  {'(no label)':<26} "
              f"{no_lbl['train']:>6} {no_lbl['val']:>6} "
              f"{no_lbl['test']:>6} {nl_tot:>6}  {rs}")


def main():
    print("=" * 55)
    print("  Stratified 70 : 15 : 15 Split")
    print("  Road Damage Detection — 5 Classes")
    print("=" * 55)

    records = collect_all()
    if not records:
        raise FileNotFoundError(
            f"No images found under {ROOT/'images'}.\n"
            "Check ROOT at the top of this script."
        )
    print(f"\nImages pooled: {len(records)}")

    train_r, val_r, test_r = stratified_split(records)
    verify(train_r, val_r, test_r)
    write_splits(train_r, val_r, test_r)
    print_stats(train_r, val_r, test_r)

    print("\n" + "=" * 55)
    print("  Split complete!")
    print(f"  train : {len(train_r)}")
    print(f"  val   : {len(val_r)}")
    print(f"  test  : {len(test_r)}")
    print("  Next  : update data.yaml, run 01_train_all_models.py")
    print("=" * 55)


if __name__ == "__main__":
    main()