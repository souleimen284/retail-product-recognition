"""Predict the product class of one image with the trained checkpoint.

Usage:
  python -m src.predict path\\to\\image.jpg
  python -m src.predict path\\to\\image.jpg --checkpoint checkpoints\\best_model.pt --json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from src import config
from src.dataset import eval_transform, load_image
from src.model import load_checkpoint


@torch.no_grad()
def predict_image(model, classes: list[str], image_path, image_size: int = config.IMAGE_SIZE) -> dict:
    """Return the top class, its confidence and the probability of every class."""
    x = eval_transform(image_size)(load_image(image_path)).unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    probs = torch.softmax(model(x), dim=1)[0].cpu()
    order = torch.argsort(probs, descending=True).tolist()
    return {
        "image": str(image_path),
        "predicted_class": classes[order[0]],
        "confidence": float(probs[order[0]]),
        "probabilities": {classes[i]: float(probs[i]) for i in order},   # sorted, highest first
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict the product in an image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_PATH)
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f"[ERROR] image not found: {args.image}")
    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] checkpoint not found: {args.checkpoint} (train first: python -m src.train)")

    model, ckpt = load_checkpoint(args.checkpoint)
    if list(ckpt["classes"]) != list(config.CLASSES):
        print("[WARN] checkpoint class list differs from src/config.py - using the checkpoint's list.")
    result = predict_image(model, list(ckpt["classes"]), args.image, ckpt.get("image_size", config.IMAGE_SIZE))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Image     : {result['image']}")
        print(f"Predicted : {result['predicted_class']}  ({result['confidence']:.1%})")
        print("All classes:")
        for name, p in result["probabilities"].items():
            print(f"  {name:36s} {p:6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
