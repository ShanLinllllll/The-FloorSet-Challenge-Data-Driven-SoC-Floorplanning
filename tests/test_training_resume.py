"""Check that a safe full checkpoint reproduces the next training update."""

from __future__ import annotations

import random
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from train_gnn_pilot_v2 import save_full, restore_full


class ResumeTests(unittest.TestCase):
    def test_full_checkpoint_reproduces_next_update(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        random.seed(13)
        np.random.seed(13)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def components():
            net = torch.nn.Sequential(
                torch.nn.Linear(8, 16),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(16, 2),
            ).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
            schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1)
            scaler = (
                torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=2)
                if device.type == "cuda"
                else None
            )
            return net, opt, schedule, scaler

        def step(items, x, y):
            net, opt, schedule, scaler = items
            net.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler is not None, dtype=torch.float16):
                prediction = net(x)
            loss = (prediction.float() - y).square().mean()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            schedule.step(float(loss.detach()))
            return float(loss.detach())

        original = components()
        x, y = torch.randn(4, 8, device=device), torch.randn(4, 2, device=device)
        step(original, x, y)
        step(original, x, y)
        temp_root = ROOT / ".cache/tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(temp_root.resolve()))
            destination = Path(directory) / "last.pt"
            save_full(
                destination,
                *original,
                {"torch": str(torch.__version__)},
                {"completed_epoch": 2},
                0.1,
            )
            payload = torch.load(destination, map_location="cpu", weights_only=True)
            expected_loss = step(original, x, y)
            expected_state = {k: v.detach().clone() for k, v in original[0].state_dict().items()}
            expected_draws = (
                random.random(),
                float(np.random.rand()),
                torch.rand(3),
                torch.rand(3, device=device),
            )
            resumed = components()
            restore_full(payload, *resumed)
            actual_loss = step(resumed, x, y)
            self.assertEqual(actual_loss, expected_loss)
            for name, value in resumed[0].state_dict().items():
                torch.testing.assert_close(value, expected_state[name], atol=0, rtol=0)
            self.assertEqual(random.random(), expected_draws[0])
            self.assertEqual(float(np.random.rand()), expected_draws[1])
            torch.testing.assert_close(torch.rand(3), expected_draws[2], atol=0, rtol=0)
            torch.testing.assert_close(
                torch.rand(3, device=device), expected_draws[3], atol=0, rtol=0
            )
            self.assertEqual(original[2].state_dict(), resumed[2].state_dict())
            if original[3]:
                self.assertEqual(original[3].state_dict(), resumed[3].state_dict())
            self.assertTrue(resumed[1].state)
            self.assertFalse(destination.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
