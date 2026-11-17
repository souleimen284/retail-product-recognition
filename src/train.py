"""Fine-tune MobileNetV3-Small on the 12 selected classes (train split), selecting the
checkpoint on the validation split. The test split is never loaded here.

Usage (CPU, defaults agreed in the plan):
  python -m src.train
  python -m src.train --workers 0          # lowest RAM use

Two stages:
  1. head    : backbone frozen (BatchNorm statistics frozen too), train the classifier,
               AdamW lr 1e-3, constant learning rate
  2. finetune: whole network, AdamW lr 1e-4 with cosine decay, early stopping on
               validation macro-F1 (patience 6)
The best epoch over both stages (highest val macro-F1, ties -> lower val loss) is saved.

Outputs:
  checkpoints/best_model.pt         best checkpoint (weights + class list + metadata)
  results/training/training_history.csv   per-epoch losses, accuracy, macro-F1, lr, time, RAM
  results/training/training_config.json   all settings and environment, for reproducibility
  results/training/training_curves.png    loss / validation metric curves
  results/training/training_summary.json  best epoch, total time, peak RAM, BatchNorm check
"""
import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src import config
from src.dataset import build_loader
from src.model import build_model, count_parameters, save_checkpoint, set_backbone_trainable

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune MobileNetV3-Small (train/val splits only)")
    p.add_argument("--epochs-head", type=int, default=3)
    p.add_argument("--epochs-finetune", type=int, default=25)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=6, help="early stopping (finetune stage), in epochs")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--threads", type=int, default=6, help="CPU threads for PyTorch (6 physical cores)")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_PATH)
    p.add_argument("--out-dir", type=Path, default=config.TRAINING_DIR)
    p.add_argument("--log-file", type=Path, default=None, help="also write console output to this file")
    p.add_argument("--max-batches", type=int, default=None, help="testing only: limit batches per epoch")
    return p.parse_args(argv)


def rel(path: Path) -> str:
    """Project-relative path for logs/JSON (keeps local user folders out of the repository)."""
    try:
        return str(Path(path).resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def ram_used_gb() -> float:
    """RAM of this process plus its DataLoader worker processes."""
    proc = psutil.Process()
    rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
    return rss / 1024**3


def batchnorm_layers(model) -> list:
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def bn_running_stats(model) -> list:
    """Copies of every BatchNorm running mean/var (used to prove stage 1 keeps them fixed)."""
    return [t.detach().clone() for m in batchnorm_layers(model) for t in (m.running_mean, m.running_var)]


class Tee:
    """Write console output to the screen and to a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_bn=False, max_batches=None) -> dict:
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates (no gradients)."""
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:
        model.features.eval()      # frozen backbone: keep its BatchNorm running statistics fixed
    total_loss, n, preds, targets = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total_loss / n,
        "acc": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro",
                             labels=list(range(len(config.CLASSES))), zero_division=0),
    }


def plot_history(history: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history.epoch, history.train_loss, label="train loss (label-smoothed)")
    ax1.plot(history.epoch, history.val_loss, label="val loss")
    ax2.plot(history.epoch, history.train_acc, label="train accuracy")
    ax2.plot(history.epoch, history.val_acc, label="val accuracy")
    ax2.plot(history.epoch, history.val_macro_f1, label="val macro-F1")
    head_end = history[history.stage == "head"].epoch.max()
    for ax in (ax1, ax2):
        if not np.isnan(head_end):
            ax.axvline(head_end + 0.5, color="grey", ls="--", lw=1)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax1.set_title("Loss")
    ax2.set_title("Accuracy / macro-F1 (dashed line: backbone unfrozen)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Only the train and val splits are used for training and model selection.
    train_loader = build_loader("train", args.batch_size, args.workers, args.seed)
    val_loader = build_loader("val", args.batch_size, args.workers, args.seed)

    model = build_model(pretrained=True).to(device)
    bns = batchnorm_layers(model)
    assert all(any(m is b for b in model.features.modules()) for m in bns), "BatchNorm outside backbone"
    print(f"BatchNorm: {len(bns)} layers (all in the backbone), momentum {bns[0].momentum}, eps {bns[0].eps}. "
          "Stage 'head': running stats frozen. Stage 'finetune': running stats updated.")
    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()      # plain CE: comparable across runs and settings

    run_config = {
        **{k: rel(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "classes": config.CLASSES,
        "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
        "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
        "selection_metric": "val macro-F1 (ties: lower val loss)",
    }
    with open(args.out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"Device {device} | train {run_config['n_train']} | val {run_config['n_val']} | "
          f"threads {args.threads} | workers {args.workers} | batch {args.batch_size}")

    stages = [("head", args.epochs_head, False, args.lr_head),
              ("finetune", args.epochs_finetune, True, args.lr_finetune)]
    history, best = [], {"val_macro_f1": -1.0, "val_loss": float("inf"), "epoch": None}
    epoch, start_all, bn_check = 0, time.perf_counter(), {}

    for stage, n_epochs, backbone_trainable, lr in stages:
        if n_epochs <= 0:
            continue
        set_backbone_trainable(model, backbone_trainable)
        total, trainable = count_parameters(model)
        print(f"\n--- Stage '{stage}': {trainable:,} / {total:,} parameters trainable, lr {lr} ---")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=lr, weight_decay=args.weight_decay)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
                     if stage == "finetune" else None)
        epochs_without_improvement = 0
        bn_before = bn_running_stats(model)

        for _ in range(n_epochs):
            epoch += 1
            t0 = time.perf_counter()
            current_lr = optimizer.param_groups[0]["lr"]
            tr = run_epoch(model, train_loader, train_criterion, device, optimizer,
                           freeze_bn=not backbone_trainable, max_batches=args.max_batches)
            va = run_epoch(model, val_loader, val_criterion, device, max_batches=args.max_batches)
            if scheduler is not None:
                scheduler.step()

            improved = (va["macro_f1"] > best["val_macro_f1"]
                        or (va["macro_f1"] == best["val_macro_f1"] and va["loss"] < best["val_loss"]))
            if improved:
                best = {"val_macro_f1": va["macro_f1"], "val_loss": va["loss"], "val_acc": va["acc"],
                        "epoch": epoch, "stage": stage}
                save_checkpoint(args.checkpoint, model, epoch=epoch, stage=stage,
                                val_macro_f1=va["macro_f1"], val_acc=va["acc"], val_loss=va["loss"],
                                train_args=run_config)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            row = {"epoch": epoch, "stage": stage, "lr": current_lr,
                   "train_loss": tr["loss"], "train_acc": tr["acc"],
                   "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
                   "best": improved, "epoch_seconds": time.perf_counter() - t0, "ram_gb": ram_used_gb()}
            history.append(row)
            pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
            print(f"epoch {epoch:2d} [{stage:8s}] lr {current_lr:.2e} | train loss {tr['loss']:.3f} "
                  f"acc {tr['acc']:.3f} | val loss {va['loss']:.3f} acc {va['acc']:.3f} "
                  f"F1 {va['macro_f1']:.3f} | {row['epoch_seconds']:.0f}s RAM {row['ram_gb']:.2f} GB"
                  f"{'  * saved' if improved else ''}", flush=True)

            if stage == "finetune" and epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val macro-F1 improvement for {args.patience} epochs.")
                break

        unchanged = all(torch.equal(a, b) for a, b in zip(bn_before, bn_running_stats(model)))
        bn_check[stage] = "running stats unchanged" if unchanged else "running stats updated"
        print(f"BatchNorm check after stage '{stage}': {bn_check[stage]}")
        if stage == "head" and not unchanged:
            raise RuntimeError("BatchNorm running statistics changed while the backbone was frozen")

    hist = pd.DataFrame(history)
    plot_history(hist, args.out_dir / "training_curves.png")
    summary = {"best": best, "epochs_run": epoch,
               "total_minutes": round((time.perf_counter() - start_all) / 60, 2),
               "checkpoint": rel(args.checkpoint), "batchnorm_check": bn_check,
               "peak_ram_gb": round(float(hist.ram_gb.max()), 2)}
    with open(args.out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest epoch {best['epoch']} ({best.get('stage')}): val macro-F1 {best['val_macro_f1']:.3f}, "
          f"val acc {best.get('val_acc', float('nan')):.3f}, val loss {best['val_loss']:.3f}")
    print(f"Total time {summary['total_minutes']} min. Checkpoint: {args.checkpoint}")
    return summary


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.log_file:
        cli_args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(cli_args.log_file, "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, log_handle)
        sys.stderr = Tee(sys.__stderr__, log_handle)
    train(cli_args)
    sys.exit(0)
