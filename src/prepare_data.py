"""Dataset preparation and verification (never modifies dataset files).

Usage:
  python -m src.prepare_data --counts-only   # uses only the official split files
  python -m src.prepare_data                 # also verifies the downloaded images

--counts-only:
  * parses train.txt / val.txt / test.txt
  * checks that the 12 selected classes exist and have consistent labels
  * exact image counts per class and split, checked against config.MIN_IMAGES
  * checks that no image path appears in more than one split (or twice in one split)
  * writes results/data_audit/split_counts.csv and data/sparse_paths.txt (folders to download)

Full mode additionally:
  * checks every selected image exists and can be decoded
  * records image size, mode and EXIF orientation
  * writes data/manifest.csv and results/data_audit/dataset_summary.csv
"""
import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from src import config


def read_split_file(split: str) -> pd.DataFrame:
    path = config.SPLIT_FILES[split]
    if not path.exists():
        sys.exit(f"[ERROR] Missing split file: {path}\n"
                 "Run the sparse clone step first (see data/README.md).")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                raise ValueError(f"{path.name}:{line_no}: unexpected format: {line!r}")
            rel_path = parts[0].replace("\\", "/")
            rows.append({
                "split": split,
                "rel_path": rel_path,
                "fine_id": int(parts[1]),
                "coarse_id": int(parts[2]),
                "class_name": Path(rel_path).parent.name,
            })
    return pd.DataFrame(rows)


def load_class_names() -> dict:
    """Map fine-grained class ID -> class name from the official classes.csv."""
    if not config.CLASSES_CSV.exists():
        print(f"[WARN] {config.CLASSES_CSV} not found; skipping classes.csv cross-check.")
        return {}
    df = pd.read_csv(config.CLASSES_CSV)
    return dict(zip(df.iloc[:, 1].astype(int), df.iloc[:, 0].astype(str)))


def check_counts(all_rows: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    ok = True
    print("=== Official split files (all 81 classes) ===")
    for split in config.SPLITS:
        part = all_rows[all_rows.split == split]
        print(f"{split:5s}: {len(part):5d} images, {part.class_name.nunique():3d} classes")

    # 1. Selected classes exist
    available = set(all_rows.class_name)
    missing = [c for c in config.CLASSES if c not in available]
    if missing:
        for c in missing:
            close = difflib.get_close_matches(c, sorted(available), n=3)
            print(f"[ERROR] Class folder not found: {c}  (closest: {close})")
        sys.exit("Fix the class names in src/config.py and re-run.")

    sel = all_rows[all_rows.class_name.isin(config.CLASSES)].copy()

    # 2. Label consistency (one fine_id per folder; name matches classes.csv)
    id_to_name = load_class_names()
    print("\n=== Label consistency ===")
    for c in config.CLASSES:
        ids = sorted(sel.loc[sel.class_name == c, "fine_id"].unique())
        csv_name = id_to_name.get(ids[0], "?") if len(ids) == 1 else "?"
        status = "OK" if len(ids) == 1 else "INCONSISTENT"
        if len(ids) != 1:
            ok = False
        print(f"{c:36s} fine_id={ids}  classes.csv name={csv_name!r}  {status}")

    # 3. Exact counts vs minimums
    counts = (sel.groupby(["class_name", "split"]).size().unstack(fill_value=0)
              .reindex(index=config.CLASSES, columns=config.SPLITS, fill_value=0))
    counts["total"] = counts.sum(axis=1)
    counts.insert(0, "group", [config.CLASS_TO_GROUP[c] for c in counts.index])
    below = []
    for c in config.CLASSES:
        for split in config.SPLITS:
            if counts.loc[c, split] < config.MIN_IMAGES[split]:
                below.append(f"{c} / {split}: {counts.loc[c, split]} < {config.MIN_IMAGES[split]}")
    print("\n=== Exact counts for the 12 selected classes ===")
    print(counts.to_string())
    print("TOTAL", {s: int(counts[s].sum()) for s in config.SPLITS}, "all:", int(counts.total.sum()))
    if below:
        ok = False
        print("\n[WARN] Below agreed minimum:")
        for b in below:
            print("   ", b)
    else:
        print(f"\nAll classes meet the minimums {config.MIN_IMAGES}.")

    # 4. Split integrity (whole dataset, not only the selection)
    print("\n=== Split integrity ===")
    dup_within = all_rows[all_rows.duplicated(["split", "rel_path"], keep=False)]
    print(f"Paths listed twice within a split : {dup_within.rel_path.nunique()}")
    path_splits = all_rows.groupby("rel_path").split.nunique()
    cross = path_splits[path_splits > 1]
    print(f"Paths listed in more than one split: {len(cross)}")
    sel_names = sel.assign(fname=sel.rel_path.map(lambda p: Path(p).name))
    name_splits = sel_names.groupby("fname").split.nunique()
    print(f"Selected file names in >1 split    : {int((name_splits > 1).sum())}  "
          "(informational only: same file name in different split folders is not a duplicate by itself)")
    if len(dup_within) or len(cross):
        ok = False
    return counts, ok


def write_counts_outputs(sel: pd.DataFrame, counts: pd.DataFrame) -> None:
    config.DATA_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(config.DATA_AUDIT_DIR / "split_counts.csv", index_label="class_name")
    folders = sorted({str(Path(p).parent).replace("\\", "/") for p in sel.rel_path})
    patterns = ["/dataset/*.txt", "/dataset/classes.csv", "/LICENSE", "/README.md"]
    patterns += [f"/dataset/{folder}/" for folder in folders]
    config.SPARSE_PATHS_TXT.write_text("\n".join(patterns) + "\n", encoding="utf-8")
    print(f"\nWrote {config.DATA_AUDIT_DIR / 'split_counts.csv'}")
    print(f"Wrote {config.SPARSE_PATHS_TXT} ({len(folders)} image folders)")


def verify_images(sel: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    print("\n=== Verifying image files ===")
    records, problems = [], []
    for row in sel.itertuples(index=False):
        path = config.DATASET_ROOT / row.rel_path
        rec = {"split": row.split, "rel_path": row.rel_path, "class_name": row.class_name,
               "label": config.CLASS_TO_IDX[row.class_name], "group": config.CLASS_TO_GROUP[row.class_name],
               "fine_id": row.fine_id}
        if not path.exists():
            problems.append(f"missing: {row.rel_path}")
            continue
        try:
            with Image.open(path) as im:
                im.verify()                       # integrity check (does not decode pixels)
            with Image.open(path) as im:
                orientation = im.getexif().get(274, 1)
                im = ImageOps.exif_transpose(im)
                im.load()                         # full decode
                rec.update(width=im.width, height=im.height, mode=im.mode,
                           exif_orientation=orientation, bytes=path.stat().st_size)
        except Exception as exc:  # noqa: BLE001 - report any decode failure
            problems.append(f"unreadable: {row.rel_path} ({exc})")
            continue
        records.append(rec)
    manifest = pd.DataFrame(records)
    print(f"Readable images: {len(manifest)} / {len(sel)}")
    for p in problems:
        print("[ERROR]", p)
    return manifest, not problems


def write_summary(manifest: pd.DataFrame) -> None:
    manifest.to_csv(config.MANIFEST_CSV, index=False)
    counts = (manifest.groupby(["class_name", "split"]).size().unstack(fill_value=0)
              .reindex(index=config.CLASSES, columns=config.SPLITS, fill_value=0))
    sizes = (manifest.assign(min_side=manifest[["width", "height"]].min(axis=1))
             .groupby("class_name")
             .agg(median_width=("width", "median"), median_height=("height", "median"),
                  smallest_side=("min_side", "min")))
    summary = counts.join(sizes)
    summary.insert(0, "label", [config.CLASS_TO_IDX[c] for c in summary.index])
    summary.insert(1, "group", [config.CLASS_TO_GROUP[c] for c in summary.index])
    summary["total"] = summary[config.SPLITS].sum(axis=1)
    config.DATA_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.DATA_AUDIT_DIR / "dataset_summary.csv", index_label="class_name")
    print("\n=== Dataset summary ===")
    print(summary.to_string())
    print("\nImage sizes (w x h) across selection:")
    print(manifest.groupby(["width", "height"]).size().sort_values(ascending=False).head(10).to_string())
    print(f"Images with EXIF rotation != 1: {(manifest.exif_orientation != 1).sum()}")
    print(f"Colour modes: {manifest['mode'].value_counts().to_dict()}")
    print(f"\nWrote {config.MANIFEST_CSV}")
    print(f"Wrote {config.DATA_AUDIT_DIR / 'dataset_summary.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts-only", action="store_true",
                        help="only analyse the split files (no image files needed)")
    args = parser.parse_args()

    all_rows = pd.concat([read_split_file(s) for s in config.SPLITS], ignore_index=True)
    counts, ok = check_counts(all_rows)
    sel = all_rows[all_rows.class_name.isin(config.CLASSES)]
    write_counts_outputs(sel, counts)

    if not args.counts_only:
        manifest, files_ok = verify_images(sel)
        ok = ok and files_ok
        write_summary(manifest)

    print("\nRESULT:", "all checks passed" if ok else "issues found - see messages above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
