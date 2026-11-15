"""Train the custom CNN, optionally with augmentation; test data is not loaded here."""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import SmallCNN


ROOT = Path(__file__).resolve().parent
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class CifakeDataset(Dataset):
    def __init__(self, split, augment=False):
        if augment and split != "train":
            raise ValueError("Augmentation is allowed only for the training split.")
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            TRANSFORM,
        ]) if augment else TRANSFORM
        with (ROOT / "splits/cifake.csv").open(newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No images in split {split!r}; run prepare_data.py first.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / "data/cifake" / row["path"]) as image:
            if image.size != (32, 32) or image.mode != "RGB":
                raise ValueError(f"Expected 32x32 RGB image: {row['path']}")
            tensor = self.transform(image)
        return tensor, int(row["label"])


def classification_metrics(confusion):
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    count = tn + fp + fn + tp
    return {
        "accuracy": (tn + tp) / count if count else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "confusion_matrix": confusion,
        "images": count,
    }


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = count = 0
    confusion = torch.zeros((2, 2), dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; training stopped.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.numel()
            count += labels.numel()
            predictions = logits.detach().argmax(dim=1)
            confusion += torch.bincount(labels * 2 + predictions, minlength=4).reshape(2, 2)
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return dict(classification_metrics(confusion.cpu().tolist()), loss=total_loss / count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--augment", dest="augmentation", action="store_true",
                        help="apply random crops and horizontal flips to training images only")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/cnn-baseline")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("epochs, batch size, threads, and learning rate must be positive")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    train_data = CifakeDataset("train", augment=args.augmentation)
    val_data = CifakeDataset("val")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "output": str(args.output), "device": str(device),
              "python": sys.version,
              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "parameters": sum(p.numel() for p in model.parameters()),
              "labels": {"REAL": 0, "FAKE": 1},
              "training_transform": repr(train_data.transform),
              "normalization_mean": [0.5] * 3, "normalization_std": [0.5] * 3,
              "weight_decay": 1e-4,
              "data_audit": json.loads((ROOT / "splits/data_audit.json").read_text())}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Device: {device}; parameters: {config['parameters']:,}; "
          f"training images: {len(train_data):,}; validation images: {len(val_data):,}", flush=True)
    best_f1 = -1.0
    with (args.output / "history.csv").open("w", newline="") as handle:
        fields = ("epoch", "train_loss", "train_accuracy", "train_f1", "val_loss", "val_accuracy", "val_f1", "seconds")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            val_metrics = run_epoch(model, val_loader, device)
            row = {"epoch": epoch, "seconds": time.perf_counter() - started,
                   **{f"train_{key}": train_metrics[key] for key in ("loss", "accuracy", "f1")},
                   **{f"val_{key}": val_metrics[key] for key in ("loss", "accuracy", "f1")}}
            writer.writerow(row)
            handle.flush()
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                temporary = args.output / "best.pt.tmp"
                torch.save({"model": model.state_dict(), "epoch": epoch, "validation": val_metrics,
                            "config": config}, temporary)
                temporary.replace(args.output / "best.pt")
            print(f"Epoch {epoch}/{args.epochs}: train loss={train_metrics['loss']:.4f}, "
                  f"val loss={val_metrics['loss']:.4f}, val accuracy={val_metrics['accuracy']:.3%}, "
                  f"val F1={val_metrics['f1']:.4f}, {row['seconds']:.1f}s", flush=True)
    print(f"Best validation F1: {best_f1:.4f}; checkpoint: {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
