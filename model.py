import os
from pathlib import Path

from torch import nn


CONVNEXT_NAME = "convnextv2_tiny.fcmae_ft_in1k"


class SmallCNN(nn.Module):
    """Binary classifier for 32x32 RGB images; returns raw class logits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))


def build_model(name="small-cnn", pretrained=False):
    if name == "small-cnn":
        return SmallCNN()
    if name != "convnextv2-tiny":
        raise ValueError(f"Unknown model: {name}")
    cache = Path(__file__).resolve().parent / "data/.cache"
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    import timm

    return timm.create_model(CONVNEXT_NAME, pretrained=pretrained, num_classes=2)
