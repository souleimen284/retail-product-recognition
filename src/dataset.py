"""Dataset and data loaders for the 12 selected classes, using the official splits.

Images are listed in data/manifest.csv (written by src.prepare_data from the official
train.txt / val.txt / test.txt). Each split is loaded independently; no image is moved,
copied or shared between splits.

Preprocessing
  train : random crop/scale, small rotation, mild brightness/contrast/saturation change
          (no horizontal flip: mirrored label text is unrealistic; no hue shift: colour is a
          key cue, e.g. orange vs green Bravo cartons)
  val / test : deterministic resize + centre crop to 224x224
"""
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def eval_transform(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Deterministic: identical input -> identical tensor. Square 348x348 images are simply
    resized; the few 348x464 images lose a little of their top/bottom in the centre crop."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def load_image(path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def load_manifest() -> pd.DataFrame:
    if not config.MANIFEST_CSV.exists():
        raise FileNotFoundError(f"{config.MANIFEST_CSV} not found - run `python -m src.prepare_data` first.")
    m = pd.read_csv(config.MANIFEST_CSV)
    # Labels are always derived from the class list in config, never trusted blindly from the CSV.
    expected = m.class_name.map(config.CLASS_TO_IDX)
    if expected.isna().any() or not (expected == m.label).all():
        raise ValueError("manifest labels do not match config.CLASS_TO_IDX - re-run prepare_data")
    # Leakage guard: an image may belong to exactly one split.
    if m.rel_path.duplicated().any():
        raise ValueError("an image path appears in more than one split")
    return m


class GroceryDataset(Dataset):
    """Images of one official split, with labels 0..11 in config.CLASSES order."""

    def __init__(self, split: str, transform):
        if split not in config.SPLITS:
            raise ValueError(f"unknown split {split!r}")
        m = load_manifest()
        self.split = split
        self.samples = m[m.split == split].reset_index(drop=True)
        self.transform = transform
        self.classes = list(config.CLASSES)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        image = load_image(config.DATASET_ROOT / row.rel_path)
        return self.transform(image), int(row.label)


def seed_worker(worker_id: int) -> None:
    """Make numpy/random inside DataLoader workers reproducible (torch is seeded by the loader)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_loader(split: str, batch_size: int = config.BATCH_SIZE, num_workers: int = config.NUM_WORKERS,
                 seed: int = config.SEED, augment: bool | None = None) -> DataLoader:
    """Augmentation and shuffling are used only for the training split."""
    is_train = split == "train"
    augment = is_train if augment is None else augment
    dataset = GroceryDataset(split, train_transform() if augment else eval_transform())
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,   # avoid re-spawning workers every epoch (slow on Windows)
        pin_memory=False,                     # CPU training; also keeps RAM use low
    )
