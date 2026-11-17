"""Evaluate the selected checkpoint on the held-out TEST split - exactly once.

Usage:  python -m src.evaluate --expect-epoch 12

* Loads checkpoints/best_model.pt read-only (its SHA-256 is recorded before and after).
* Runs inference over the 355 test images ONCE and saves every prediction to
  results/metrics/test_predictions.csv. Error analysis reads that file; it never re-runs the model.
* Refuses to run again if results/metrics/test_predictions.csv already exists (use --force only
  for a documented, deliberate re-run - nothing in this project needs it).

Outputs:
  results/metrics/test_predictions.csv       one row per test image: true/pred class, confidence, all 12 probs
  results/metrics/metrics.json               accuracy, macro P/R/F1 with 95% bootstrap CIs, checkpoint info
  results/metrics/classification_report.csv  per-class precision / recall / F1 / support
  results/confusion_matrices/confusion_matrix(.csv|.png)             raw counts (rows = true class)
  results/confusion_matrices/confusion_matrix_normalized(.csv|.png)  row-normalised (diagonal = recall)
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src import config
from src.dataset import build_loader
from src.model import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PREDICTIONS_CSV = config.METRICS_DIR / "test_predictions.csv"
N_CLASSES = len(config.CLASSES)
LABELS = list(range(N_CLASSES))
# Single-hue sequential ramp (light -> dark blue) for magnitude in the confusion matrix.
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def predict_split(model, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = build_loader(split, batch_size=config.BATCH_SIZE, num_workers=0, augment=False)
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x), dim=1))
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), list(loader.dataset.samples.rel_path)


def macro_scores(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {"precision_macro": p, "recall_macro": r, "f1_macro": f}


def bootstrap_cis(y_true, y_pred, n_boot: int, seed: int) -> dict:
    """95% percentile intervals.
    accuracy      : ordinary bootstrap (resample all test images with replacement)
    macro metrics : stratified bootstrap (resample within each true class, keeping each
                    class's size), so all 12 classes are present in every resample
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    by_class = [np.where(y_true == c)[0] for c in LABELS]
    acc, macro = [], {k: [] for k in ("precision_macro", "recall_macro", "f1_macro")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        acc.append(accuracy_score(y_true[idx], y_pred[idx]))
        sidx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class])
        for k, v in macro_scores(y_true[sidx], y_pred[sidx]).items():
            macro[k].append(v)

    def ci(values):
        lo, hi = np.percentile(values, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    out = {"accuracy": ci(acc)}
    out.update({k: ci(v) for k, v in macro.items()})
    return out


def short_name(name: str) -> str:
    return (name.replace("-Ecological-", "-Eco-").replace("Medium-Fat", "MedFat")
            .replace("-Yoghurt", "").replace("-Juice", "").replace("God-Morgon", "GodMorgon"))


def plot_confusion(cm: np.ndarray, path, normalized: bool) -> None:
    names = [short_name(c) for c in config.CLASSES]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    vmax = 1.0 if normalized else cm.max()
    im = ax.imshow(cm, cmap=SEQ_CMAP, vmin=0, vmax=vmax)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            v = cm[i, j]
            if v == 0:
                continue
            text = f"{v:.2f}" if normalized else f"{int(v)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * vmax else "#1f1f1f")
    ax.set_xticks(LABELS, names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(LABELS, names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(-0.5, N_CLASSES), minor=True)
    ax.set_yticks(np.arange(-0.5, N_CLASSES), minor=True)
    ax.grid(which="minor", color="#e6e6e6", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = "Test confusion matrix - row-normalised (diagonal = recall)" if normalized \
        else "Test confusion matrix - counts"
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="One-time evaluation on the test split")
    parser.add_argument("--checkpoint", default=str(config.CHECKPOINT_PATH))
    parser.add_argument("--expect-epoch", type=int, default=None,
                        help="abort unless the checkpoint was saved at this epoch")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing evaluation")
    args = parser.parse_args()

    if PREDICTIONS_CSV.exists() and not args.force:
        sys.exit(f"[STOP] {PREDICTIONS_CSV} exists: the test set has already been evaluated once.")

    sha_before = sha256_file(args.checkpoint)
    model, ckpt = load_checkpoint(args.checkpoint, device="cpu")
    if args.expect_epoch is not None and ckpt.get("epoch") != args.expect_epoch:
        sys.exit(f"[STOP] checkpoint epoch is {ckpt.get('epoch')}, expected {args.expect_epoch}")
    if list(ckpt["classes"]) != list(config.CLASSES):
        sys.exit("[STOP] checkpoint class order differs from src/config.py")
    torch.set_num_threads(6)
    print(f"Checkpoint: epoch {ckpt['epoch']} ({ckpt['stage']}), val macro-F1 {ckpt['val_macro_f1']:.4f}, "
          f"sha256 {sha_before[:16]}...")

    # ---- the single pass over the test set ----
    probs, y_true, paths = predict_split(model, "test")
    y_pred = probs.argmax(1)
    print(f"Test images evaluated: {len(y_true)}")

    pred_df = pd.DataFrame({
        "rel_path": paths,
        "true_label": y_true, "true_class": [config.CLASSES[i] for i in y_true],
        "pred_label": y_pred, "pred_class": [config.CLASSES[i] for i in y_pred],
        "confidence": probs.max(1).round(4),
        "true_class_prob": probs[np.arange(len(y_true)), y_true].round(4),
        "correct": y_true == y_pred,
    })
    for i, c in enumerate(config.CLASSES):
        pred_df[f"p_{c}"] = probs[:, i].round(4)
    for d in (config.METRICS_DIR, config.CONFUSION_DIR):
        d.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    # ---- metrics ----
    acc = accuracy_score(y_true, y_pred)
    macro = macro_scores(y_true, y_pred)
    cis = bootstrap_cis(y_true, y_pred, args.n_boot, args.seed)

    report = classification_report(y_true, y_pred, labels=LABELS, target_names=config.CLASSES,
                                   output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report).T
    rep_df.loc["accuracy"] = [np.nan, np.nan, acc, len(y_true)]
    rep_df["support"] = rep_df["support"].astype(int)
    rep_df.round(4).to_csv(config.METRICS_DIR / "classification_report.csv", index_label="class")

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    pd.DataFrame(cm, index=config.CLASSES, columns=config.CLASSES).to_csv(
        config.CONFUSION_DIR / "confusion_matrix.csv", index_label="true \\ predicted")
    pd.DataFrame(cm_norm.round(4), index=config.CLASSES, columns=config.CLASSES).to_csv(
        config.CONFUSION_DIR / "confusion_matrix_normalized.csv", index_label="true \\ predicted")
    plot_confusion(cm, config.CONFUSION_DIR / "confusion_matrix.png", normalized=False)
    plot_confusion(cm_norm, config.CONFUSION_DIR / "confusion_matrix_normalized.png", normalized=True)

    sha_after = sha256_file(args.checkpoint)
    metrics = {
        "split": "test",
        "n_images": int(len(y_true)),
        "n_correct": int((y_true == y_pred).sum()),
        "accuracy": round(float(acc), 4),
        **{k: round(float(v), 4) for k, v in macro.items()},
        "ci95": cis,
        "bootstrap": {"n_resamples": args.n_boot, "seed": args.seed, "method": "percentile",
                      "accuracy": "ordinary bootstrap over test images",
                      "macro_metrics": "stratified bootstrap within each true class",
                      "caveat": "assumes independent images; burst shots in the test set make the intervals somewhat optimistic"},
        "checkpoint": {"path": os.path.relpath(args.checkpoint, config.PROJECT_ROOT), "epoch": ckpt["epoch"], "stage": ckpt["stage"],
                       "val_macro_f1": round(float(ckpt["val_macro_f1"]), 4),
                       "val_acc": round(float(ckpt["val_acc"]), 4),
                       "sha256_before": sha_before, "sha256_after": sha_after,
                       "unchanged": sha_before == sha_after},
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
    }
    with open(config.METRICS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: metrics[k] for k in ("n_images", "n_correct", "accuracy", "precision_macro",
                                              "recall_macro", "f1_macro", "ci95")}, indent=2))
    print("\nPer-class report:")
    print(rep_df.round(3).to_string())
    print(f"\nCheckpoint unchanged: {sha_before == sha_after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
