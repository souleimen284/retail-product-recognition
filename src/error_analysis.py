"""Error analysis of the one-time test evaluation.

Usage:  python -m src.error_analysis
Reads results/metrics/test_predictions.csv (written by src.evaluate). It does NOT run the
model again and does not modify any image.

Outputs (results/error_analysis/):
  misclassified.csv                 every wrong test prediction, most confident first
  confusion_pairs.csv               directed (true -> predicted) error counts
  confusion_pairs_symmetric.csv     undirected pair counts (A<->B), with brand/group flags
  error_summary.json                headline numbers (errors per class, confidence stats)
  examples/NN_<true>__as__<pred>.png
                                    one example per confusion pair (its most confident error),
                                    shown next to a TRAINING image of the predicted class
"""
import json
import sys

import matplotlib
import pandas as pd

from src import config
from src.dataset import load_image, load_manifest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRED_CSV = config.METRICS_DIR / "test_predictions.csv"
OUT_DIR = config.ERROR_DIR / "examples"


def brand(name: str) -> str:
    return "God-Morgon" if name.startswith("God-Morgon") else name.split("-")[0]


def second_choice(row) -> tuple[str, float]:
    probs = {c: row[f"p_{c}"] for c in config.CLASSES if c != row.pred_class}
    c = max(probs, key=probs.get)
    return c, probs[c]


def select_examples(errors: pd.DataFrame, pairs: pd.DataFrame, max_examples: int = 12) -> pd.DataFrame:
    """One example per confusion pair (its most confident error), most frequent pairs first,
    at most `max_examples`. Deterministic (errors are sorted by confidence; no random choice)."""
    chosen = []
    for _, p in pairs.head(max_examples).iterrows():
        sub = errors[(errors.true_class == p.true_class) & (errors.pred_class == p.pred_class)]
        chosen.append(sub.index[0])
    return errors.loc[chosen]


def reference_train_image(pred_class: str, manifest: pd.DataFrame) -> str:
    """First training image of the predicted class (what the model 'thought' it saw)."""
    return manifest[(manifest.split == "train") & (manifest.class_name == pred_class)].rel_path.iloc[0]


def save_example(ax_pair, row, ref_path):
    left, right = ax_pair
    left.imshow(load_image(config.DATASET_ROOT / row.rel_path))
    sc, sp = second_choice(row)
    left.set_title(f"TEST  true: {row.true_class}\npred: {row.pred_class} ({row.confidence:.0%})\n"
                   f"p(true) {row.true_class_prob:.0%} | 2nd: {sc} ({sp:.0%})", fontsize=7)
    right.imshow(load_image(config.DATASET_ROOT / ref_path))
    right.set_title(f"TRAIN example of predicted class\n{row.pred_class}", fontsize=7)
    for ax in ax_pair:
        ax.axis("off")


def main() -> int:
    if not PRED_CSV.exists():
        sys.exit(f"[ERROR] {PRED_CSV} not found - run `python -m src.evaluate` first.")
    df = pd.read_csv(PRED_CSV)
    manifest = load_manifest()
    errors = df[~df.correct].sort_values("confidence", ascending=False)
    errors.insert(0, "file", errors.rel_path.str.split("/").str[-1])

    # Misclassified list
    cols = ["file", "true_class", "pred_class", "confidence", "true_class_prob", "rel_path"]
    config.ERROR_DIR.mkdir(parents=True, exist_ok=True)
    errors[cols].to_csv(config.ERROR_DIR / "misclassified.csv", index=False)

    # Directed and undirected confusion pairs
    pairs = (errors.groupby(["true_class", "pred_class"]).size().rename("count").reset_index()
             .sort_values(["count", "true_class"], ascending=[False, True]).reset_index(drop=True))
    support = df.true_class.value_counts()
    pairs["share_of_true_class"] = (pairs["count"] / pairs.true_class.map(support)).round(3)
    pairs["same_brand"] = [brand(a) == brand(b) for a, b in zip(pairs.true_class, pairs.pred_class)]
    pairs["same_group"] = [config.CLASS_TO_GROUP[a] == config.CLASS_TO_GROUP[b]
                           for a, b in zip(pairs.true_class, pairs.pred_class)]
    pairs.to_csv(config.ERROR_DIR / "confusion_pairs.csv", index=False)

    und = errors.assign(pair=[" <-> ".join(sorted((a, b))) for a, b in zip(errors.true_class, errors.pred_class)])
    sym = und.groupby("pair").size().rename("count").sort_values(ascending=False).reset_index()
    sym["same_brand"] = [brand(p.split(" <-> ")[0]) == brand(p.split(" <-> ")[1]) for p in sym.pair]
    sym["same_group"] = [config.CLASS_TO_GROUP[p.split(" <-> ")[0]] == config.CLASS_TO_GROUP[p.split(" <-> ")[1]]
                         for p in sym.pair]
    sym.to_csv(config.ERROR_DIR / "confusion_pairs_symmetric.csv", index=False)

    # Example images (one per confusion pair)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = select_examples(errors, pairs)
    written = []
    for k, (_, row) in enumerate(examples.iterrows(), start=1):
        ref = reference_train_image(row.pred_class, manifest)
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
        save_example(axes, row, ref)
        fig.tight_layout()
        name = f"{k:02d}_{row.true_class}__as__{row.pred_class}.png"
        fig.savefig(OUT_DIR / name, dpi=110)
        plt.close(fig)
        written.append(name)
    # Summary
    per_class_err = (errors.true_class.value_counts().reindex(config.CLASSES, fill_value=0))
    correct = df[df.correct]
    summary = {
        "n_test": int(len(df)),
        "n_errors": int(len(errors)),
        "errors_per_true_class": {c: f"{int(per_class_err[c])}/{int(support[c])}" for c in config.CLASSES},
        "errors_within_same_group": int(pairs.loc[pairs.same_group, "count"].sum()),
        "errors_within_same_brand": int(pairs.loc[pairs.same_brand, "count"].sum()),
        "top_directed_pairs": pairs.head(10)[["true_class", "pred_class", "count"]].to_dict("records"),
        "top_undirected_pairs": sym.head(10)[["pair", "count"]].to_dict("records"),
        "confidence_mean_correct": round(float(correct.confidence.mean()), 3) if len(correct) else None,
        "confidence_mean_errors": round(float(errors.confidence.mean()), 3) if len(errors) else None,
        "errors_with_confidence_over_80pct": int((errors.confidence > 0.8).sum()),
        "errors_where_true_class_was_2nd_choice": int(sum(second_choice(r)[0] == r.true_class
                                                          for _, r in errors.iterrows())),
        "example_images": written,
    }
    with open(config.ERROR_DIR / "error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Errors: {len(errors)} / {len(df)} test images\n")
    print("Directed confusion pairs (true -> predicted):")
    print(pairs.head(15).to_string(index=False))
    print("\nUndirected pairs:")
    print(sym.head(10).to_string(index=False))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k not in ("top_directed_pairs", "top_undirected_pairs")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
