"""MobileNetV3-Small with an ImageNet-pretrained backbone and a new 12-class output layer.

torchvision's MobileNetV3-Small:
  features   : convolutional backbone (pretrained on ImageNet) -> 576 channels
  avgpool    : global average pooling                          -> 576-d vector
  classifier : Linear(576, 1024) -> Hardswish -> Dropout(0.2) -> Linear(1024, 1000)

Only the last Linear layer is replaced (1000 ImageNet classes -> 12 products); the
pretrained Linear(576, 1024) is kept. Training is done in two stages (see src/train.py):
  1. backbone frozen: train the classifier only
  2. everything unfrozen: fine-tune the whole network with a small learning rate
"""
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src import config


def build_model(num_classes: int = len(config.CLASSES), pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features          # 1024
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(path, model: nn.Module, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "classes": list(config.CLASSES),          # index -> class name, used by predict.py
        "image_size": config.IMAGE_SIZE,
        "mean": config.IMAGENET_MEAN,
        "std": config.IMAGENET_STD,
        "torch_version": torch.__version__,
        **metadata,
    }, path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the architecture (no weight download needed) and load the saved weights."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(num_classes=len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
