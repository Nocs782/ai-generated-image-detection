"""Evaluate a frozen checkpoint on the official test set after model selection."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import SmallCNN
from train import CifakeDataset, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    torch.set_num_threads(checkpoint["config"]["threads"])
    model = SmallCNN()
    model.load_state_dict(checkpoint["model"])
    data = CifakeDataset("test")
    metrics = run_epoch(model, DataLoader(data, batch_size=64), torch.device("cpu"))
    result = {"checkpoint": str(args.checkpoint), "selected_epoch": checkpoint["epoch"],
              "positive_class": "FAKE", "confusion_matrix_order": ["REAL", "FAKE"],
              "confusion_matrix_axes": "rows=true labels, columns=predictions", **metrics}
    destination = args.checkpoint.parent / "test_metrics.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
