"""Evaluate a frozen checkpoint on CIFAKE or the fixed GenImage external sample."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from model import build_model
from prepare_genimage import pixel_digest
from train import ROOT, CifakeDataset, classification_metrics, make_transform, run_epoch


class GenImageDataset(Dataset):
    def __init__(self, transform):
        manifest = ROOT / "splits/genimage.csv"
        self.audit = json.loads((ROOT / "splits/genimage_audit.json").read_text())
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != self.audit["manifest_sha256"]:
            raise ValueError("External manifest differs from the audited sample.")
        with manifest.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows or len(self.rows) != self.audit["selected_images"]:
            raise ValueError("External manifest count does not match its audit.")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        root = (ROOT / "data/genimage-test").resolve()
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Unsafe external image path.")
        with Image.open(path) as image:
            image = image.convert("RGB")
            if pixel_digest(image) != row["digest"]:
                raise ValueError(f"External image changed after preparation: {row['path']}")
            return self.transform(image), int(row["label"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dataset", choices=("cifake", "genimage"), default="cifake")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment or execution context.")
    config = checkpoint["config"]
    if config.get("model") == "mambavision-t" and device.type != "cuda":
        parser.error("MambaVision evaluation requires CUDA; use --device cuda in the Mamba environment.")
    torch.set_num_threads(config["threads"])
    model = build_model(config.get("model", "small-cnn"))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    transform = make_transform(config.get("input_size", 32), config["normalization_mean"], config["normalization_std"])
    data = GenImageDataset(transform) if args.dataset == "genimage" else CifakeDataset("test", transform=transform)
    amp = device.type == "cuda" and config.get("amp", False)
    per_generator = {}
    if args.dataset == "genimage":
        for generator in sorted({row["generator"] for row in data.rows}):
            indices = [i for i, row in enumerate(data.rows) if row["generator"] == generator]
            per_generator[generator] = run_epoch(
                model, DataLoader(Subset(data, indices), batch_size=args.batch_size), device, amp=amp)
            print(f"{generator}: accuracy={per_generator[generator]['accuracy']:.3%}", flush=True)
        confusion = [[sum(m["confusion_matrix"][i][j] for m in per_generator.values()) for j in range(2)]
                     for i in range(2)]
        metrics = classification_metrics(confusion)
        metrics["loss"] = sum(m["loss"] * m["images"] for m in per_generator.values()) / metrics["images"]
    else:
        metrics = run_epoch(model, DataLoader(data, batch_size=args.batch_size), device, amp=amp)
    result = {"checkpoint": str(args.checkpoint), "selected_epoch": checkpoint["epoch"],
              "model": config.get("model", "small-cnn"), "device": str(device), "amp": amp,
              "positive_class": "FAKE", "confusion_matrix_order": ["REAL", "FAKE"],
              "confusion_matrix_axes": "rows=true labels, columns=predictions", **metrics}
    if args.dataset == "genimage":
        with args.checkpoint.open("rb") as handle:
            checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        result.update(dataset="genimage", per_generator=per_generator, data_audit=data.audit,
                      checkpoint_sha256=checkpoint_hash, torch=str(torch.__version__))
    destination = args.checkpoint.parent / ("genimage_metrics.json" if args.dataset == "genimage" else "test_metrics.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("data_audit", "per_generator")}, indent=2))


if __name__ == "__main__":
    main()
