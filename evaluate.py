"""Evaluate a frozen checkpoint on the official test set after model selection."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import build_model
from train import CifakeDataset, make_transform, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment or execution context.")
    config = checkpoint["config"]
    torch.set_num_threads(config["threads"])
    model = build_model(config.get("model", "small-cnn"))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    transform = make_transform(config.get("input_size", 32), config["normalization_mean"], config["normalization_std"])
    data = CifakeDataset("test", transform=transform)
    amp = device.type == "cuda" and config.get("amp", False)
    metrics = run_epoch(model, DataLoader(data, batch_size=args.batch_size), device, amp=amp)
    result = {"checkpoint": str(args.checkpoint), "selected_epoch": checkpoint["epoch"],
              "model": config.get("model", "small-cnn"), "device": str(device), "amp": amp,
              "positive_class": "FAKE", "confusion_matrix_order": ["REAL", "FAKE"],
              "confusion_matrix_axes": "rows=true labels, columns=predictions", **metrics}
    destination = args.checkpoint.parent / "test_metrics.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
