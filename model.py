import os
from pathlib import Path

from torch import nn


CONVNEXT_NAME = "convnextv2_tiny.fcmae_ft_in1k"
MAMBA_REPO = "nvidia/MambaVision-T-1K"
MAMBA_REVISION = "b1de77e17599566d98efb701c0231b1095dc3a67"


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
    if name not in ("convnextv2-tiny", "mambavision-t"):
        raise ValueError(f"Unknown model: {name}")
    cache = Path(__file__).resolve().parent / "data/.cache"
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("HF_MODULES_CACHE", str(cache / "huggingface/modules"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    if name == "mambavision-t":
        from transformers import AutoConfig, AutoModelForImageClassification

        config = AutoConfig.from_pretrained(MAMBA_REPO, revision=MAMBA_REVISION, trust_remote_code=True)
        if pretrained:
            wrapper = AutoModelForImageClassification.from_pretrained(
                MAMBA_REPO, config=config, revision=MAMBA_REVISION, code_revision=MAMBA_REVISION,
                trust_remote_code=True, use_safetensors=True)
        else:
            wrapper = AutoModelForImageClassification.from_config(
                config, code_revision=MAMBA_REVISION, trust_remote_code=True)
        model = wrapper.model
        model.head = nn.Linear(model.head.in_features, 2)
        nn.init.trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)
        model.num_classes = 2
        model.pretrained_cfg = {"hf_hub_id": MAMBA_REPO, "revision": MAMBA_REVISION,
                                "mean": config.mean, "std": config.std,
                                "input_size": (3, 224, 224), "drop_path_rate": config.drop_path_rate}
        return model
    import timm

    return timm.create_model(CONVNEXT_NAME, pretrained=pretrained, num_classes=2)
