import io
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image

from model import SmallCNN
from train import CifakeDataset, classification_metrics, make_transform


class ModelTests(unittest.TestCase):
    def test_can_learn_tiny_sample_and_reload_weights(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = SmallCNN()
        images = torch.zeros(8, 3, 32, 32)
        labels = torch.tensor([0, 1] * 4)
        images[labels == 0, 0] = 1
        images[labels == 1, 2] = 1
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(30):
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(images)
        self.assertEqual(expected.shape, (8, 2))
        self.assertTrue(torch.equal(expected.argmax(1), labels))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = SmallCNN().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(restored(images), expected)

    def test_metrics_use_fake_as_positive_class(self):
        metrics = classification_metrics([[8, 2], [3, 7]])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 7 / 9)
        self.assertAlmostEqual(metrics["recall"], 0.7)
        self.assertAlmostEqual(metrics["f1"], 14 / 19)
        self.assertEqual(classification_metrics([[10, 0], [10, 0]])["f1"], 0.0)

    def test_augmentation_changes_training_images_but_not_labels_or_evaluation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "splits").mkdir()
            image = Image.frombytes("RGB", (32, 32), bytes(i % 256 for i in range(32 * 32 * 3)))
            with (root / "splits/cifake.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("path", "label", "split"))
                for split in ("train", "val", "test"):
                    relative = f"{split}/FAKE/example.jpg"
                    target = root / "data/cifake" / relative
                    target.parent.mkdir(parents=True)
                    image.save(target)
                    writer.writerow((relative, 1, split))
            with patch("train.ROOT", root):
                torch.manual_seed(42)
                training = CifakeDataset("train", augment=True)
                samples = [training[0] for _ in range(6)]
                self.assertTrue(all(tensor.shape == (3, 32, 32) and label == 1 for tensor, label in samples))
                self.assertTrue(any(not torch.equal(samples[0][0], tensor) for tensor, _ in samples[1:]))
                self.assertTrue(all(torch.isfinite(tensor).all() and tensor.abs().max() <= 1 for tensor, _ in samples))
                resized = CifakeDataset("train", augment=True, transform=make_transform(224))
                self.assertEqual(resized[0][0].shape, (3, 224, 224))
                self.assertEqual(resized[0][1], 1)
                for split in ("val", "test"):
                    evaluation = CifakeDataset(split)
                    torch.testing.assert_close(evaluation[0][0], evaluation[0][0])
                    self.assertEqual(evaluation[0][1], 1)
                    with self.assertRaises(ValueError):
                        CifakeDataset(split, augment=True)

    def test_evaluation_loads_legacy_checkpoint(self):
        import evaluate

        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "splits").mkdir()
            (root / "splits/cifake.csv").write_text(
                "path,label,split\ntest/REAL/a.jpg,0,test\ntest/FAKE/b.jpg,1,test\n")
            for relative in ("test/REAL/a.jpg", "test/FAKE/b.jpg"):
                path = root / "data/cifake" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32)).save(path)
            model = SmallCNN()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            checkpoint = root / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": 1,
                        "config": {"threads": 2, "normalization_mean": [0.5] * 3,
                                   "normalization_std": [0.5] * 3}}, checkpoint)
            with patch("train.ROOT", root), patch("sys.argv", ["evaluate.py", str(checkpoint)]), patch("sys.stdout", new_callable=io.StringIO):
                evaluate.main()
            metrics = json.loads((root / "test_metrics.json").read_text())
            self.assertEqual(metrics["images"], 2)
            self.assertEqual(metrics["confusion_matrix"], [[1, 0], [1, 0]])
            self.assertEqual(metrics["model"], "small-cnn")


if __name__ == "__main__":
    unittest.main()
