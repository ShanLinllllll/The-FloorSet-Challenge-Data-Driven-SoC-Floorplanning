from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FloorSet" / "iccad2026contest"))
import numpy as np
import torch
from scipy.optimize import OptimizeResult
from baseline_optimizer import MyOptimizer
from baseline_solver.core import Problem, feasible, metrics, soft_counts
from baseline_solver.search import BaselineConfig, compact_axis
import iccad2026_evaluate as official


def inputs(areas, constraints=None, targets=None, edges=None, pins=None, external=None):
    n = len(areas)
    return dict(
        block_count=n,
        area_targets=torch.tensor(areas, dtype=torch.float64),
        b2b_connectivity=torch.tensor(edges or [], dtype=torch.float64).reshape(-1, 3),
        p2b_connectivity=torch.tensor(external or [], dtype=torch.float64).reshape(-1, 3),
        pins_pos=torch.tensor(pins or [], dtype=torch.float64).reshape(-1, 2),
        constraints=torch.tensor(constraints if constraints is not None else [[0] * 5] * n),
        target_positions=torch.tensor(
            targets if targets is not None else [[-1.0] * 4] * n, dtype=torch.float64
        ),
    )


class BaselineTests(unittest.TestCase):
    def solve(self, data, guess=None):
        solver = MyOptimizer(config=BaselineConfig(max_seconds=10, sa_steps=25, seed=7))
        result = solver.solve_with_initial_guess(**data, initial_guess=guess)
        problem = Problem.from_inputs(**data)
        self.assertTrue(feasible(problem, np.asarray(result)))
        self.assertTrue(all(isinstance(row, tuple) and len(row) == 4 for row in result))
        return solver, result

    def test_fixed_shape_and_preplaced_are_immutable(self):
        data = inputs(
            [999, 888, 100, 200],
            [[1, 0, 0, 1, 1], [0, 1, 0, 1, 0], [0] * 5, [0] * 5],
            [[-1, -1, 10, 4], [12, 6, 6, 3], [-1] * 4, [-1] * 4],
            edges=[[0, 2, 1], [1, 3, 2]],
            pins=[[0, 0]],
            external=[[0, 2, 1]],
        )
        solver, result = self.solve(data)
        self.assertEqual(result[0][2:], (10.0, 4.0))
        self.assertEqual(result[1], (12.0, 6.0, 6.0, 3.0))
        self.assertGreater(solver.last_run["sa"]["proposals"], 0)
        self.assertLessEqual(
            solver.last_run["stage_metrics"]["sa"]["objective"],
            solver.last_run["stage_metrics"]["lp"]["objective"],
        )

    def test_all_preplaced_and_disconnected_inputs(self):
        data = inputs([1, 1], [[0, 1, 0, 0, 0]] * 2, [[-3, 0, 2, 2], [5, 4, 3, 1]])
        _, result = self.solve(data)
        self.assertEqual(result, [(-3.0, 0.0, 2.0, 2.0), (5.0, 4.0, 3.0, 1.0)])
        self.solve(inputs([1, 5, 17, 2]))

    def test_compatible_mib_and_conflicting_mib(self):
        data = inputs([100, 100, 100], [[0, 0, 1, 0, 0]] * 3)
        _, result = self.solve(data)
        self.assertEqual(len({row[2:] for row in result}), 1)
        # Different immutable area requirements must not be overwritten for MIB.
        data = inputs([25, 400], [[0, 0, 1, 0, 0]] * 2)
        _, result = self.solve(data)
        self.assertEqual(soft_counts(Problem.from_inputs(**data), np.asarray(result))["mib"], 1)

    def test_prediction_interface_projects_hard_geometry(self):
        data = inputs(
            [100, 200, 300],
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0] * 5],
            [[-1, -1, 2, 4], [10, 10, 3, 4], [-1] * 4],
        )
        _, result = self.solve(data, np.array([[1, 1, 9, 9], [1, 1, 9, 9], [1, 1, 9, 9]]))
        self.assertEqual(result[0][2:], (2.0, 4.0))
        self.assertEqual(result[1], (10.0, 10.0, 3.0, 4.0))

    def test_inconsistent_preplaced_input_is_reported(self):
        data = inputs([4, 4], [[0, 1, 0, 0, 0]] * 2, [[0, 0, 2, 2], [1, 1, 2, 2]])
        with self.assertRaisesRegex(ValueError, "Preplaced blocks overlap"):
            self.solve(data)

    def test_lp_reduces_spacing_and_keeps_anchor(self):
        data = inputs(
            [100, 100], [[0, 1, 0, 0, 0], [0] * 5], [[0, 0, 10, 10], [-1] * 4], edges=[[0, 1, 1]]
        )
        problem = Problem.from_inputs(**data)
        layout = np.array([[0.0, 0.0, 10.0, 10.0], [30.0, 0.0, 10.0, 10.0]])
        result, status = compact_axis(problem, layout, 0, 1)
        self.assertEqual(status, "accepted")
        self.assertTrue(feasible(problem, result))
        self.assertLess(metrics(problem, result)["bbox_area"], 400)
        np.testing.assert_array_equal(result[0], layout[0])

    def test_lp_failure_retains_legal_layout(self):
        data = inputs([4, 4])
        problem = Problem.from_inputs(**data)
        layout = np.array([[0.0, 0.0, 2.0, 2.0], [4.0, 0.0, 2.0, 2.0]])
        with patch(
            "baseline_solver.search.linprog", return_value=OptimizeResult(success=False, status=1)
        ):
            result, status = compact_axis(problem, layout, 0, 0.01)
        np.testing.assert_array_equal(result, layout)
        self.assertEqual(status, "solver_status_1")

    def test_internal_soft_metrics_match_official(self):
        data = inputs(
            [4, 4, 4],
            [[0, 0, 1, 1, 9], [0, 0, 1, 1, 0], [0, 0, 1, 1, 6]],
            edges=[[0, 1, 1]],
            pins=[[0, 0]],
            external=[[0, 1, 2]],
        )
        problem = Problem.from_inputs(**data)
        for layout in [
            np.array([[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0], [4.0, 2.0, 2.0, 2.0]]),
            np.array([[0.0, 0.0, 2.0, 2.0], [2.1, 0.0, 2.0, 2.0], [4.1, 2.0, 2.0, 2.0]]),
        ]:
            internal = metrics(problem, layout)
            result = official.evaluate_solution(
                {"positions": layout.tolist(), "runtime": 1},
                {"hpwl_baseline": 1.0, "area_baseline": 1.0},
                data["constraints"],
                data["b2b_connectivity"],
                data["p2b_connectivity"],
                data["pins_pos"],
                data["area_targets"],
                layout.tolist(),
            )
            self.assertEqual(
                internal["soft_counts"],
                {k: getattr(result, k + "_violations") for k in ["boundary", "grouping", "mib"]},
            )
            self.assertAlmostEqual(internal["hpwl"], result.hpwl_total)

    def test_fixed_step_seed_reproducibility(self):
        data = inputs([10, 20, 30, 40], edges=[[0, 1, 1], [1, 3, 2]])
        _, first = self.solve(data)
        _, second = self.solve(data)
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-9)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
