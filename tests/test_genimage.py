import csv
import io
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image
import torch
from torchvision import transforms

import evaluate
from model import SmallCNN
from prepare_genimage import GENERATORS, pixel_digest, prepare_archive
from train import make_transform


class GenImageTests(unittest.TestCase):
    def make_archive(self, root, unsafe=False):
        archive = root / "genimage_test.zip"
        excluded = set()
        with ZipFile(archive, "w") as z:
            for g, generator in enumerate(GENERATORS):
                for label, folder in enumerate(("nature", "ai")):
                    for i in range(4):
                        image = Image.frombytes("RGB", (40, 24), random.Random(g * 100 + label * 10 + i).randbytes(40 * 24 * 3))
                        if g == label == i == 0:
                            excluded.add(pixel_digest(image))
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        z.writestr(f"test/{generator}/{folder}/{i}.png", buffer.getvalue())
            if unsafe:
                z.writestr("test/../../escape.png", b"unsafe")
        return archive, excluded

    def prepare(self, root, archive, excluded):
        with patch("sys.stdout", new_callable=io.StringIO):
            return prepare_archive(archive, root / "data/genimage-test", root / "splits/genimage.csv",
                                   root / "splits/genimage_audit.json", excluded, per_group=3)

    def test_sample_is_balanced_reproducible_and_excludes_overlap(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            archive, excluded = self.make_archive(root)
            first, second = root / "first", root / "second"
            audit = self.prepare(first, archive, excluded)
            self.prepare(second, archive, excluded)
            self.assertEqual(audit["selected_images"], 48)
            self.assertEqual((first / "splits/genimage.csv").read_bytes(), (second / "splits/genimage.csv").read_bytes())
            with (first / "splits/genimage.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len({row["digest"] for row in rows}), 48)
            self.assertFalse({row["digest"] for row in rows} & excluded)
            for generator in GENERATORS:
                self.assertEqual([sum(row["generator"] == generator and row["label"] == str(label) for row in rows)
                                  for label in (0, 1)], [3, 3])
            (first / "data/genimage-test").rename(first / "data/original")
            self.prepare(first, archive, excluded)
            self.assertEqual((first / "splits/genimage.csv").read_bytes(), (second / "splits/genimage.csv").read_bytes())
            with self.assertRaises(FileExistsError):
                self.prepare(first, archive, excluded)

    def test_unsafe_paths_and_insufficient_groups_are_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            archive, excluded = self.make_archive(root, unsafe=True)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                self.prepare(root / "unsafe", archive, excluded)
            empty = root / "empty.zip"
            with ZipFile(empty, "w"):
                pass
            with self.assertRaisesRegex(ValueError, "eight generators"):
                self.prepare(root / "empty", empty, excluded)
            self.assertFalse((root / "unsafe/data/genimage-test.partial").exists())
            self.assertFalse((root / "escape.png").exists())

    def test_external_evaluation_and_tamper_detection_preserve_cifake_results(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            archive, excluded = self.make_archive(root)
            self.prepare(root, archive, excluded)
            model = SmallCNN()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            checkpoint = root / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": 7,
                        "config": {"threads": 2, "normalization_mean": [0.5] * 3,
                                   "normalization_std": [0.5] * 3}}, checkpoint)
            sentinel = root / "test_metrics.json"
            sentinel.write_text("original CIFAKE results")
            with patch("evaluate.ROOT", root):
                data = evaluate.GenImageDataset(make_transform())
                self.assertEqual(data[0][0].shape, (3, 32, 32))
                torch.testing.assert_close(data[0][0], data[0][0])
                with patch("sys.argv", ["evaluate.py", str(checkpoint), "--dataset", "genimage"]), patch("sys.stdout", new_callable=io.StringIO):
                    evaluate.main()
                result = json.loads((root / "genimage_metrics.json").read_text())
                self.assertEqual(result["confusion_matrix"], [[24, 0], [24, 0]])
                self.assertEqual(len(result["per_generator"]), 8)
                self.assertTrue(all(m["confusion_matrix"] == [[3, 0], [3, 0]] for m in result["per_generator"].values()))
                self.assertEqual(result["selected_epoch"], 7)
                self.assertEqual(sentinel.read_text(), "original CIFAKE results")
                image = root / "data/genimage-test" / data.rows[0]["path"]
                Image.new("RGB", (40, 24)).save(image)
                with self.assertRaisesRegex(ValueError, "changed after preparation"):
                    data[0]
                with (root / "splits/genimage.csv").open("a") as handle:
                    handle.write("\n")
                with self.assertRaisesRegex(ValueError, "differs from the audited sample"):
                    evaluate.GenImageDataset(make_transform())

    def test_resize_preserves_original_cifake_preprocessing(self):
        image = Image.frombytes("RGB", (32, 32), random.Random(42).randbytes(32 * 32 * 3))
        original = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        torch.testing.assert_close(make_transform()(image), original(image), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
