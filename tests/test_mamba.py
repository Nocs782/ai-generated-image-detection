import importlib.util
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import build_model
from train import run_epoch


MAMBA_AVAILABLE = (importlib.util.find_spec("mamba_ssm") is not None
                   and importlib.util.find_spec("transformers") is not None
                   and torch.cuda.is_available())


class MambaDeviceTests(unittest.TestCase):
    def test_training_rejects_cpu_before_loading_model(self):
        import train

        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            with TemporaryDirectory() as folder, patch("train.build_model") as build:
                argv = ["train.py", "--model", "mambavision-t", "--device", "cpu",
                        "--output", str(Path(folder) / "run")]
                with patch("sys.argv", argv), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    with self.assertRaises(SystemExit) as error:
                        train.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("MambaVision requires CUDA", stderr.getvalue())
                build.assert_not_called()
        finally:
            torch.use_deterministic_algorithms(deterministic)

    def test_evaluation_rejects_cpu_before_loading_model(self):
        import evaluate

        with TemporaryDirectory() as folder, patch("evaluate.build_model") as build:
            checkpoint = Path(folder) / "best.pt"
            torch.save({"config": {"model": "mambavision-t"}}, checkpoint)
            with patch("sys.argv", ["evaluate.py", str(checkpoint)]), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as error:
                    evaluate.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("MambaVision evaluation requires CUDA", stderr.getvalue())
            build.assert_not_called()


@unittest.skipUnless(MAMBA_AVAILABLE, "MambaVision requires its optional dependencies and CUDA")
class MambaVisionTests(unittest.TestCase):
    def test_cuda_scan_matches_reference_and_gradients(self):
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

        torch.manual_seed(42)
        inputs = [torch.randn(2, 8, 16, device="cuda"), torch.randn(2, 8, 16, device="cuda"),
                  -torch.rand(8, 4, device="cuda"), torch.randn(2, 4, 16, device="cuda"),
                  torch.randn(2, 4, 16, device="cuda"), torch.randn(8, device="cuda")]
        actual_inputs = [value.detach().clone().requires_grad_() for value in inputs]
        reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
        actual = selective_scan_fn(*actual_inputs, delta_softplus=True)
        reference = selective_scan_ref(*reference_inputs, delta_softplus=True)
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
        actual.square().mean().backward()
        reference.square().mean().backward()
        for actual_input, reference_input in zip(actual_inputs, reference_inputs):
            torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-3, atol=1e-4)

    def test_mixed_precision_training_and_checkpoint_reload(self):
        torch.manual_seed(42)
        model = build_model("mambavision-t").cuda()
        images = torch.rand(2, 3, 224, 224)
        loader = DataLoader(TensorDataset(images, torch.tensor([0, 1])), batch_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = model.head.weight.detach().clone()
        metrics = run_epoch(model, loader, torch.device("cuda"), optimizer,
                            scaler=torch.amp.GradScaler("cuda", init_scale=1024), amp=True)
        self.assertEqual(metrics["images"], 2)
        self.assertGreater(metrics["loss"], 0)
        self.assertFalse(torch.equal(before, model.head.weight))
        model.eval()
        with torch.no_grad(), torch.autocast("cuda"):
            expected = model(images.cuda())
        self.assertEqual(expected.shape, (2, 2))
        self.assertTrue(torch.isfinite(expected).all())
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = build_model("mambavision-t").cuda().eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        with torch.no_grad(), torch.autocast("cuda"):
            torch.testing.assert_close(restored(images.cuda()), expected)


if __name__ == "__main__":
    unittest.main()
