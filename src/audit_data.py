"""Final data-integrity audit (read-only; standard library only).

Usage:  python -m src.audit_data

1. Re-computes the SHA-256 of every selected image and compares it with the record written during data preparation
   (results/data_audit/duplicates/image_hashes.csv) -> proves no image changed since data preparation.
2. Re-checks split integrity: each image in exactly one split; counts unchanged; the
   evaluated test predictions cover exactly the test split.
3. Summarises cross-split duplicates from the duplicate-analysis report (exact SHA-256, pHash,
   pretrained-feature similarity).
4. Groups within-split burst shots (pairs flagged by pHash <= 6 or cosine >= 0.95) into
   clusters, giving the number of effectively independent images per split.
Writes results/data_audit/data_integrity_audit.json. Nothing is modified; nothing feeds back into training.
"""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict

from src import config

DUP = config.DUPLICATES_DIR


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clusters(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    return list(groups.values())


def main() -> int:
    manifest = read_csv(config.MANIFEST_CSV)
    split_of = {r["rel_path"]: r["split"] for r in manifest}
    report, ok = {}, True

    # 1. Images unchanged since data preparation
    recorded = {r["rel_path"]: r["sha256"] for r in read_csv(DUP / "image_hashes.csv")}
    changed = [p for p in split_of if sha256(config.DATASET_ROOT / p) != recorded.get(p)]
    report["images_checked"] = len(split_of)
    report["images_changed_since_preparation"] = len(changed)
    ok &= not changed

    # 2. Split integrity
    paths = [r["rel_path"] for r in manifest]
    report["paths_in_more_than_one_split"] = len(paths) - len(set(paths))
    report["images_per_split"] = dict(Counter(r["split"] for r in manifest))
    preds = read_csv(config.METRICS_DIR / "test_predictions.csv")
    test_paths = {p for p, s in split_of.items() if s == "test"}
    report["test_predictions_match_test_split"] = {r["rel_path"] for r in preds} == test_paths and len(preds) == len(test_paths)
    ok &= report["paths_in_more_than_one_split"] == 0 and report["test_predictions_match_test_split"]
    hashes = Counter(recorded[p] for p in split_of)
    report["exact_duplicate_files_anywhere"] = sum(c - 1 for c in hashes.values() if c > 1)

    # 3. Cross-split near duplicates (duplicate-analysis pairs)
    pairs = read_csv(DUP / "duplicate_pairs.csv")
    cross = [p for p in pairs if p["split_a"] != p["split_b"]]
    flag = lambda p: p["flagged"] == "True"
    report["cross_split"] = {
        "pairs_flagged_as_duplicates": sum(flag(p) for p in cross),
        "exact_sha256_pairs": sum(p["sha256_equal"] == "True" for p in cross),
        "min_phash_distance": min(int(p["phash_distance"]) for p in cross) if cross else None,
        "max_feature_cosine": max(float(p["cosine"]) for p in cross) if cross else None,
        "thresholds": {"phash_flag_max": config.PHASH_FLAG_MAX, "cosine_flag_min": config.COSINE_FLAG_MIN},
    }
    ok &= report["cross_split"]["pairs_flagged_as_duplicates"] == 0
    near = read_csv(DUP / "nearest_train.csv")
    med = lambda xs: sorted(xs)[len(xs) // 2]
    report["median_similarity_to_nearest_train_image"] = {
        s: round(med([float(r["cosine"]) for r in near if r["split"] == s]), 3) for s in ("val", "test")}

    # 4. Within-split burst-shot clusters
    burst = {}
    for split in config.SPLITS:
        nodes = [p for p, s in split_of.items() if s == split]
        edges = [(p["rel_path_a"], p["rel_path_b"]) for p in pairs
                 if flag(p) and p["split_a"] == split and p["split_b"] == split]
        groups = clusters(nodes, edges)
        multi = [g for g in groups if len(g) > 1]
        burst[split] = {"images": len(nodes), "flagged_pairs": len(edges),
                        "images_in_burst_clusters": sum(len(g) for g in multi),
                        "burst_clusters": len(multi), "largest_cluster": max((len(g) for g in multi), default=1),
                        "effective_independent_images": len(groups)}
    report["within_split_bursts"] = burst
    report["result"] = "PASS" if ok else "ISSUES FOUND"
    report["note"] = "Read-only audit. No file was modified; results were not used for tuning."

    with open(config.DATA_AUDIT_DIR / "data_integrity_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
