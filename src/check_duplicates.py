"""Report-only duplicate / near-duplicate check across the official splits.

Usage:  python -m src.check_duplicates
Requires data/manifest.csv (run `python -m src.prepare_data` first).

This script NEVER modifies, moves or deletes any image. It only writes reports to
results/data_audit/duplicates/. Any removal is a separate, manually approved decision.

Three signals, computed for every pair of selected images (~780 images -> ~300k pairs):
  1. SHA-256 of file bytes            -> exact file duplicates
  2. 64-bit perceptual hash (pHash)   -> near-identical images (resized / re-encoded / tiny crop)
  3. Cosine similarity of ImageNet-pretrained MobileNetV3-Small features (no fine-tuning,
     no labels used)                  -> same shot / same scene with slightly different framing

Pair categories (priority order):
  1 = different splits, different labels   (same shot under two class names)
  2 = different splits, same label         (potential train/test leakage)
  3 = same split, different labels
  4 = same split, same label

Outputs (results/data_audit/duplicates/; the .png review sheets are not committed):
  image_hashes.csv         per-image SHA-256 and pHash
  duplicate_pairs.csv      every pair meeting a REPORT threshold, with all three scores
  nearest_train.csv        for each val/test image: most similar train image and its scores
  summary.json             counts per category/signal, thresholds, similarity statistics
  *.png                    side-by-side visual comparisons for manual review
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import imagehash
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src import config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATEGORY_LABELS = {
    1: "different splits, different label",
    2: "different splits, same label",
    3: "same split, different label",
    4: "same split, same label",
}


def load_rgb(path: Path) -> Image.Image:
    """Open an image, apply EXIF orientation, return RGB (read-only access)."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_hashes(paths: list[Path]) -> tuple[list[str], list[str], np.ndarray]:
    shas, phash_hex, bits = [], [], []
    for p in paths:
        shas.append(sha256_file(p))
        ph = imagehash.phash(load_rgb(p))          # 8x8 = 64-bit DCT hash
        phash_hex.append(str(ph))
        bits.append(ph.hash.flatten().astype(np.uint8))
    return shas, phash_hex, np.stack(bits)


@torch.no_grad()
def compute_embeddings(paths: list[Path], device: torch.device, batch_size: int) -> np.ndarray:
    """L2-normalised 576-d pooled features of the ImageNet-pretrained MobileNetV3-Small."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights).eval().to(device)
    preprocess = weights.transforms()                 # resize 256, centre crop 224, ImageNet norm
    feats = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([preprocess(load_rgb(p)) for p in paths[start:start + batch_size]]).to(device)
        x = model.avgpool(model.features(batch)).flatten(1)
        feats.append(torch.nn.functional.normalize(x, dim=1).cpu())
    return torch.cat(feats).numpy()


def pairwise_hamming(bits: np.ndarray, chunk: int = 128) -> np.ndarray:
    n = len(bits)
    out = np.empty((n, n), dtype=np.uint8)
    for s in range(0, n, chunk):
        out[s:s + chunk] = (bits[s:s + chunk, None, :] != bits[None, :, :]).sum(-1)
    return out


def build_pairs(m: pd.DataFrame, shas: list[str], ham: np.ndarray, cos: np.ndarray) -> pd.DataFrame:
    sha_arr = np.array(shas)
    iu, ju = np.triu_indices(len(m), k=1)
    sha_eq = sha_arr[iu] == sha_arr[ju]
    h = ham[iu, ju]
    c = cos[iu, ju]
    keep = sha_eq | (h <= config.PHASH_REPORT_MAX) | (c >= config.COSINE_REPORT_MIN)
    iu, ju, sha_eq, h, c = iu[keep], ju[keep], sha_eq[keep], h[keep], c[keep]

    a, b = m.iloc[iu].reset_index(drop=True), m.iloc[ju].reset_index(drop=True)
    pairs = pd.DataFrame({
        "rel_path_a": a.rel_path, "split_a": a.split, "class_a": a.class_name,
        "rel_path_b": b.rel_path, "split_b": b.split, "class_b": b.class_name,
        "sha256_equal": sha_eq, "phash_distance": h.astype(int), "cosine": np.round(c, 4),
    })
    cross = pairs.split_a != pairs.split_b
    conflict = pairs.class_a != pairs.class_b
    pairs["category"] = np.select([cross & conflict, cross, conflict], [1, 2, 3], default=4)
    pairs["category_label"] = pairs.category.map(CATEGORY_LABELS)
    sig_exact = pairs.sha256_equal
    sig_phash = pairs.phash_distance <= config.PHASH_FLAG_MAX
    sig_cos = pairs.cosine >= config.COSINE_FLAG_MIN
    pairs["flagged"] = sig_exact | sig_phash | sig_cos
    pairs["flag_signals"] = [
        "+".join(n for n, on in (("sha256", e), ("phash", p), ("cosine", q)) if on)
        for e, p, q in zip(sig_exact, sig_phash, sig_cos)
    ]
    # Within each category: exact copies first, then other flagged pairs, then most similar.
    order = ["category", "sha256_equal", "flagged", "phash_distance", "cosine"]
    return pairs.sort_values(order, ascending=[True, False, False, True, False]).reset_index(drop=True)


def nearest_train(m: pd.DataFrame, ham: np.ndarray, cos: np.ndarray) -> pd.DataFrame:
    train_idx = np.where(m.split.values == "train")[0]
    rows = []
    for i in np.where(m.split.values != "train")[0]:
        j = train_idx[np.argmax(cos[i, train_idx])]
        rows.append({
            "rel_path": m.rel_path.iat[i], "split": m.split.iat[i], "class_name": m.class_name.iat[i],
            "nearest_train_path": m.rel_path.iat[j], "nearest_train_class": m.class_name.iat[j],
            "cosine": round(float(cos[i, j]), 4), "phash_distance": int(ham[i, j]),
            "min_phash_to_any_train": int(ham[i, train_idx].min()),
            "same_class": m.class_name.iat[i] == m.class_name.iat[j],
        })
    return pd.DataFrame(rows).sort_values("cosine", ascending=False).reset_index(drop=True)


def save_sheets(items: pd.DataFrame, caption, out_stem: str, title: str,
                per_page: int, max_items: int) -> list[str]:
    """Save side-by-side comparison pages.

    `items` must have columns left_path/left_split/left_class and right_path/right_split/right_class.
    `caption(row)` returns the score text shown under the right-hand image.
    """
    items = items.head(max_items)
    written = []
    for page, start in enumerate(range(0, len(items), per_page), start=1):
        chunk = items.iloc[start:start + per_page]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(7, 3.2 * len(chunk)), squeeze=False)
        for r, (_, row) in enumerate(chunk.iterrows()):
            for c, side in enumerate(("left", "right")):
                rel = row[f"{side}_path"]
                img = load_rgb(config.DATASET_ROOT / rel)
                img.thumbnail((320, 320))
                ax = axes[r, c]
                ax.imshow(img)
                ax.axis("off")
                text = f"{row[f'{side}_split']} | {row[f'{side}_class']}\n{Path(rel).name}"
                if side == "right":
                    text += f"\n{caption(row)}"
                ax.set_title(text, fontsize=7)
        fig.suptitle(f"{title} (page {page})", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.985))   # leave room so the page title doesn't overlap row 1
        out = config.DUPLICATES_DIR / f"{out_stem}_{page:02d}.png"
        fig.savefig(out, dpi=80)
        plt.close(fig)
        written.append(out.name)
    return written


def pair_sheet_cols(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs.rename(columns={
        "rel_path_a": "left_path", "split_a": "left_split", "class_a": "left_class",
        "rel_path_b": "right_path", "split_b": "right_split", "class_b": "right_class"})


def pair_caption(row) -> str:
    return f"pHash d={row.phash_distance}  cos={row.cosine:.3f}  sha={'=' if row.sha256_equal else 'diff'}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Report-only duplicate check")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--per-page", type=int, default=8, help="pairs per comparison image")
    args = parser.parse_args()

    if not config.MANIFEST_CSV.exists():
        sys.exit(f"[ERROR] {config.MANIFEST_CSV} not found. Run `python -m src.prepare_data` first.")
    m = pd.read_csv(config.MANIFEST_CSV)
    paths = [config.DATASET_ROOT / p for p in m.rel_path]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    config.DUPLICATES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Images: {len(m)}  device: {device}")

    print("Computing SHA-256 and pHash ...")
    shas, phash_hex, bits = compute_hashes(paths)
    m.assign(sha256=shas, phash=phash_hex)[["split", "class_name", "rel_path", "sha256", "phash"]] \
        .to_csv(config.DUPLICATES_DIR / "image_hashes.csv", index=False)

    print("Computing pretrained MobileNetV3-Small features ...")
    emb = compute_embeddings(paths, device, args.batch_size)
    ham = pairwise_hamming(bits)
    cos = emb @ emb.T

    pairs = build_pairs(m, shas, ham, cos)
    pairs.to_csv(config.DUPLICATES_DIR / "duplicate_pairs.csv", index=False)
    near = nearest_train(m, ham, cos)
    near.to_csv(config.DUPLICATES_DIR / "nearest_train.csv", index=False)

    # Similarity statistics, to judge whether the thresholds are sensible.
    iu, ju = np.triu_indices(len(m), k=1)
    same_cls = m.class_name.values[iu] == m.class_name.values[ju]
    c_all = cos[iu, ju]

    def pct(x):
        return {q: round(float(np.percentile(x, q)), 4) for q in (50, 95, 99)} | {"max": round(float(x.max()), 4)}

    summary = {
        "note": "Report only. No image was modified, moved or deleted.",
        "n_images": int(len(m)),
        "images_per_split": m.split.value_counts().to_dict(),
        "thresholds": {
            "phash_report_max": config.PHASH_REPORT_MAX, "phash_flag_max": config.PHASH_FLAG_MAX,
            "cosine_report_min": config.COSINE_REPORT_MIN, "cosine_flag_min": config.COSINE_FLAG_MIN,
        },
        "reported_pairs_by_category": {CATEGORY_LABELS[k]: int(v) for k, v in pairs.category.value_counts().sort_index().items()},
        "flagged_pairs_by_category": {CATEGORY_LABELS[k]: int(v) for k, v in pairs[pairs.flagged].category.value_counts().sort_index().items()},
        "flagged_pairs_by_signal": pairs[pairs.flagged].flag_signals.value_counts().to_dict(),
        "exact_sha256_duplicate_pairs": int(pairs.sha256_equal.sum()),
        "cosine_stats_same_class_pairs": pct(c_all[same_cls]),
        "cosine_stats_different_class_pairs": pct(c_all[~same_cls]),
        "nearest_train_cosine_for_test": pct(near[near.split == "test"].cosine.values),
        "nearest_train_cosine_for_val": pct(near[near.split == "val"].cosine.values),
        "test_images_whose_nearest_train_is_other_class": int((~near[near.split == "test"].same_class).sum()),
    }

    print("Saving visual comparisons ...")
    sheets = {}
    cross = pair_sheet_cols(pairs[pairs.category.isin([1, 2])])
    sheets["cross_split"] = save_sheets(cross, pair_caption, "cross_split_pairs",
                                        "Cross-split pairs (flagged first, then by cosine)", args.per_page, 32)
    conflict = pair_sheet_cols(pairs[pairs.category.isin([1, 3])])
    sheets["label_conflict"] = save_sheets(conflict, pair_caption, "label_conflict_pairs",
                                           "Different-label pairs (flagged first, then by cosine)", args.per_page, 24)
    within = pair_sheet_cols(pairs[(pairs.category == 4) & pairs.flagged])
    sheets["within_split_same_label"] = save_sheets(within, pair_caption, "within_split_flagged",
                                                    "Same split, same label, flagged", args.per_page, 16)
    near_test = near[near.split == "test"].rename(columns={
        "rel_path": "left_path", "split": "left_split", "class_name": "left_class",
        "nearest_train_path": "right_path", "nearest_train_class": "right_class"}).assign(right_split="train")
    sheets["test_vs_nearest_train"] = save_sheets(
        near_test, lambda r: f"pHash d={r.phash_distance}  cos={r.cosine:.3f}",
        "test_vs_nearest_train", "Test images most similar to a train image", args.per_page, 16)
    summary["comparison_images"] = sheets

    with open(config.DUPLICATES_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nReports written to {config.DUPLICATES_DIR}  (no dataset files were changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
