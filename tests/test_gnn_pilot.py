from pathlib import Path
import sys
import unittest
import tempfile
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from gnn_pilot.data import encode_inputs, add_labels, collate_graphs, fingerprint
from gnn_pilot.model import FloorplanGNN, normalized_geometry, supervised_loss
from floorset_io import solver_inputs, assert_no_label_leakage


def sample():
    area = torch.tensor([999.0, 888.0, 12.0])
    con = torch.tensor([[1, 0, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 0, 2]])
    fp = torch.tensor([[2.0, 2.0, 0.0, 0.0], [2.0, 2.0, 5.0, 0.0], [3.0, 4.0, 9.0, 0.0]])
    return {
        "input": (
            area,
            torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 2.0]]),
            torch.tensor([[0.0, 2.0, 1.0]]),
            torch.tensor([[0.0, 10.0]]),
            con,
        ),
        "label": (torch.zeros(2, 3), fp, torch.ones(8)),
    }


class GNNTests(unittest.TestCase):
    def test_inputs_are_independent_of_forbidden_labels(self):
        raw = sample()
        assert_no_label_leakage(raw, "training")
        before = encode_inputs(solver_inputs(raw, "training"))
        modified = dict(raw)
        fp = raw["label"][1].clone()
        fp[0, 2:] += 100
        fp[2] = torch.tensor([2.0, 6.0, 100.0, 200.0])
        modified["label"] = (torch.ones(2, 3), fp, torch.zeros(8))
        after = encode_inputs(solver_inputs(modified, "training"))
        self.assertEqual(fingerprint(before), fingerprint(after))

    def test_group_id_renaming_does_not_change_features(self):
        raw = sample()
        before = encode_inputs(solver_inputs(raw, "training"))
        inputs = list(raw["input"])
        inputs[4] = inputs[4].clone()
        inputs[4][inputs[4][:, 3] == 1, 3] = 123
        changed = {"input": tuple(inputs), "label": raw["label"]}
        self.assertEqual(
            fingerprint(before), fingerprint(encode_inputs(solver_inputs(changed, "training")))
        )

    def test_padding_does_not_change_predictions(self):
        graph = encode_inputs(solver_inputs(sample(), "training"))
        padded = dict(graph)
        padded["x"] = torch.cat((graph["x"], torch.zeros(8, graph["x"].shape[1])))
        model = FloorplanGNN().eval()
        single = collate_graphs([graph])
        together = collate_graphs([graph, padded])
        with torch.no_grad():
            first = model(single)[0][0]
            second = model(together)[0][0]
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)

    def test_geometry_projection_preserves_fixed_inputs(self):
        graph = encode_inputs(solver_inputs(sample(), "training"))
        batch = collate_graphs([graph])
        prediction = torch.tensor([[[0.7, 0.4, 1.0], [0.3, 0.9, -1.0], [1.0, 1.0, 0.4]]])
        lo, wh, centers = normalized_geometry(batch, prediction)
        torch.testing.assert_close(wh[0, :2], graph["locked_wh"][:2])
        torch.testing.assert_close(centers[0, 1], graph["known_centers"][1])
        self.assertAlmostEqual(float(wh[0, 2].prod()), float(graph["area_norm"][2]), places=6)

    def test_forward_backward_and_pair_labels(self):
        graph = encode_inputs(solver_inputs(sample(), "training"))
        graph = add_labels(graph, [[0, 0, 2, 2], [5, 0, 2, 2], [9, 0, 3, 4]], torch.ones(8))
        self.assertEqual(int(graph["pair_target"][0, 1]), 0)
        self.assertEqual(int(graph["pair_target"][1, 0]), 1)
        batch = collate_graphs([graph])
        model = FloorplanGNN()
        prediction, pair = model(batch)
        loss, components = supervised_loss(batch, prediction, pair, 0.1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        )
        self.assertEqual(int(components["pair_count"]), 3)

    def test_block_permutation_equivariance(self):
        graph = encode_inputs(solver_inputs(sample(), "training"))
        batch = collate_graphs([graph])
        order = torch.tensor([2, 0, 1, 3])
        changed = {key: value.clone() for key, value in batch.items()}
        changed["x"] = changed["x"][:, order]
        changed["node_mask"] = changed["node_mask"][:, order]
        changed["adjacency"] = changed["adjacency"][:, :, order][:, :, :, order]
        model = FloorplanGNN().eval()
        with torch.no_grad():
            before = model(batch)[0]
            after = model(changed)[0]
        torch.testing.assert_close(after, before[:, order[:3]], atol=1e-6, rtol=1e-6)

    def test_safe_checkpoint_and_complete_adapter(self):
        from gnn_optimizer import MyOptimizer
        from baseline_solver.search import BaselineConfig
        from baseline_solver.core import Problem, feasible

        base = ROOT / "outputs/step3/test_temp"
        base.mkdir(parents=True, exist_ok=True)
        model = FloorplanGNN()
        with tempfile.TemporaryDirectory(dir=base) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(base.resolve()))
            path = Path(directory) / "test_weights.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model.config),
                    "epoch": 1,
                    "torch": str(torch.__version__),
                },
                path,
            )
            torch.load(path, map_location="cpu", weights_only=True)
            data = solver_inputs(sample(), "training")
            solver = MyOptimizer(
                config=BaselineConfig(
                    max_seconds=0.2, lp_passes=0, sa_steps=0, second_initialization=False
                ),
                checkpoint=path,
                device="cpu",
            )
            solution = solver.solve(**data)
            self.assertTrue(feasible(Problem.from_inputs(**data), np.asarray(solution)))
            self.assertGreater(solver.last_run["gnn"]["inference_seconds"], 0)
            self.assertLessEqual(solver.last_run["gnn"]["postprocessing_budget_seconds"], 0.2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(42)
    unittest.main(verbosity=2)
