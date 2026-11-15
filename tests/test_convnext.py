import importlib.util
import unittest

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from model import build_model
from train import make_transform, run_epoch


@unittest.skipUnless(importlib.util.find_spec("timm"), "ConvNeXt requires the optional timm dependency")
class ConvNeXtTests(unittest.TestCase):
    def test_two_class_model_backpropagates(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = build_model("convnextv2-tiny")
        logits = model(torch.rand(1, 3, 224, 224))
        self.assertEqual(logits.shape, (1, 2))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_preprocessing_matches_requested_size_and_normalization(self):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        tensor = make_transform(224, mean, std)(Image.new("RGB", (32, 32), (128, 128, 128)))
        self.assertEqual(tensor.shape, (3, 224, 224))
        expected = (torch.full((3,), 128 / 255) - torch.tensor(mean)) / torch.tensor(std)
        torch.testing.assert_close(tensor[:, 0, 0], expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is not accessible")
    def test_cuda_mixed_precision_training_step(self):
        model = build_model("convnextv2-tiny").cuda()
        loader = DataLoader(TensorDataset(torch.rand(2, 3, 224, 224), torch.tensor([0, 1])), batch_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = model.head.fc.weight.detach().clone()
        metrics = run_epoch(model, loader, torch.device("cuda"), optimizer,
                            scaler=torch.amp.GradScaler("cuda", init_scale=1024), amp=True)
        self.assertEqual(metrics["images"], 2)
        self.assertGreater(metrics["loss"], 0)
        self.assertFalse(torch.equal(before, model.head.fc.weight))


if __name__ == "__main__":
    unittest.main()
