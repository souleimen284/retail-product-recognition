"""Lightweight pipeline checks - NOT real training.

Usage:  python -m src.smoke_test
Uses a handful of images, runs 2 batches per stage, and writes only to a temporary folder
(deleted automatically). Never touches checkpoints/best_model.pt or results/.
"""
import sys
import tempfile
import traceback
from pathlib import Path

import torch

from src import config
from src.dataset import GroceryDataset, build_loader, eval_transform, load_manifest
from src.model import build_model, count_parameters, load_checkpoint, set_backbone_trainable
from src.predict import predict_image
from src.train import parse_args, train

CHECKS = []


def check(name):
    """Register a check; checks run only from main() (DataLoader workers on Windows
    re-import this module, so nothing may run at import time)."""
    def register(fn):
        CHECKS.append((name, fn))
        return fn
    return register


@check("1. class-to-index mapping")
def check_1():
    assert len(config.CLASSES) == 12 and len(set(config.CLASSES)) == 12
    assert [config.CLASS_TO_IDX[c] for c in config.CLASSES] == list(range(12))
    m = load_manifest()                       # raises if manifest labels disagree with config
    assert sorted(m.label.unique()) == list(range(12))
    return "12 classes, labels 0..11, manifest consistent"


@check("2. official splits, no overlap")
def check_2():
    sizes = {s: len(GroceryDataset(s, eval_transform())) for s in config.SPLITS}
    assert sizes == {"train": 366, "val": 59, "test": 355}, sizes
    m = load_manifest()
    sets = {s: set(m[m.split == s].rel_path) for s in config.SPLITS}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    src = Path(config.PROJECT_ROOT / "src" / "train.py").read_text(encoding="utf-8")
    assert 'build_loader("test"' not in src and "'test'" not in src, "train.py references the test split"
    return f"{sizes}; train.py never loads the test split"


@check("3. augmentation on train only")
def check_3():
    tr, va, te = (build_loader(s, num_workers=0) for s in config.SPLITS)
    names = lambda t: [type(x).__name__ for x in t.transforms]
    assert "RandomResizedCrop" in names(tr.dataset.transform)
    for loader in (va, te):
        assert not any(n.startswith("Random") or n == "ColorJitter" for n in names(loader.dataset.transform))
    assert isinstance(tr.sampler, torch.utils.data.RandomSampler)
    assert isinstance(va.sampler, torch.utils.data.SequentialSampler) and isinstance(te.sampler, torch.utils.data.SequentialSampler)
    a, b = tr.dataset[0][0], tr.dataset[0][0]
    assert not torch.equal(a, b), "train transform should be random"
    return f"train: {names(tr.dataset.transform)}; val/test: {names(va.dataset.transform)}; only train shuffled"


@check("4. val/test preprocessing deterministic")
def check_4():
    for split in ("val", "test"):
        ds = GroceryDataset(split, eval_transform())
        for idx in (0, len(ds) - 1):
            x1, y1 = ds[idx]
            x2, y2 = ds[idx]
            assert torch.equal(x1, x2) and y1 == y2
            assert tuple(x1.shape) == (3, 224, 224)
    return "identical tensors on repeated loads, shape (3, 224, 224)"


@check("5. DataLoader with 2 workers (Windows spawn)")
def check_5():
    loader = build_loader("train", batch_size=config.BATCH_SIZE, num_workers=2)
    x, y = next(iter(loader))
    assert tuple(x.shape) == (config.BATCH_SIZE, 3, 224, 224) and y.dtype == torch.int64
    return f"batch {tuple(x.shape)}, labels {y.tolist()}"


@check("6. model: pretrained backbone, 12-class head, freezing")
def check_6():
    model = build_model(pretrained=True)
    assert model.classifier[3].out_features == 12 and model.classifier[0].in_features == 576
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 12)
    set_backbone_trainable(model, False)
    total, frozen_trainable = count_parameters(model)
    set_backbone_trainable(model, True)
    _, all_trainable = count_parameters(model)
    assert frozen_trainable < all_trainable == total
    return f"output (2, 12); trainable params frozen={frozen_trainable:,} / unfrozen={all_trainable:,}"


@check("7. mini training run + checkpoint save/load + prediction")
def check_7():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        args = parse_args(["--epochs-head", "1", "--epochs-finetune", "1", "--max-batches", "2",
                           "--workers", "0", "--checkpoint", str(tmp / "smoke.pt"), "--out-dir", str(tmp)])
        summary = train(args)
        assert (tmp / "smoke.pt").exists() and (tmp / "training_history.csv").exists()

        model, ckpt = load_checkpoint(tmp / "smoke.pt")
        assert ckpt["classes"] == config.CLASSES, "class order differs between training and checkpoint"
        # Reloaded weights must reproduce the saved model's outputs exactly.
        reference = build_model(pretrained=False)
        reference.load_state_dict(torch.load(tmp / "smoke.pt")["model_state"])
        reference.eval()
        x, _ = next(iter(build_loader("val", batch_size=4, num_workers=0)))
        with torch.no_grad():
            assert torch.equal(model(x), reference(x))

        val_img = config.DATASET_ROOT / GroceryDataset("val", eval_transform()).samples.rel_path.iloc[0]
        r = predict_image(model, ckpt["classes"], val_img)
        assert len(r["probabilities"]) == 12 and abs(sum(r["probabilities"].values()) - 1) < 1e-4
        assert r["predicted_class"] in config.CLASSES
        assert r["predicted_class"] == max(r["probabilities"], key=r["probabilities"].get)
    return (f"best epoch {summary['best']['epoch']}, checkpoint reloads identically, "
            f"predict -> {r['predicted_class']} ({r['confidence']:.1%}, untrained so meaningless)")


def main() -> int:
    results = []
    for name, fn in CHECKS:
        try:
            detail = fn() or ""
            results.append(True)
            print(f"PASS  {name}  {detail}", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append(False)
            print(f"FAIL  {name}  {exc!r}", flush=True)
            traceback.print_exc()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
