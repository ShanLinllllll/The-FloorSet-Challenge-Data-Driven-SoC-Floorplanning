"""Behavior tests checked against the unmodified official evaluator."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "FloorSet" / "iccad2026contest"), str(ROOT / "FloorSet")]
import torch
from iccad2026_evaluate import evaluate_solution
from my_optimizer import MyOptimizer


class OptimizerTests(unittest.TestCase):
    def check(self, areas, constraints, targets, edges=None, pins=None, pin_edges=None):
        areas = torch.tensor(areas, dtype=torch.float32)
        c = torch.tensor(constraints, dtype=torch.float32)
        t = torch.tensor(targets, dtype=torch.float32)
        edges = torch.tensor(edges or [], dtype=torch.float32).reshape(-1, 3)
        pins = torch.tensor(pins or [], dtype=torch.float32).reshape(-1, 2)
        pin_edges = torch.tensor(pin_edges or [], dtype=torch.float32).reshape(-1, 3)
        result = MyOptimizer(seconds=.5).solve(len(areas), areas, edges, pin_edges, pins, c, t)
        self.assertEqual(len(result), len(areas))
        self.assertTrue(all(isinstance(p, tuple) and len(p) == 4 for p in result))
        metrics = evaluate_solution({"positions": result, "runtime": 1},
                                    {"hpwl_baseline": 1, "area_baseline": float(areas.sum())},
                                    c, edges, pin_edges, pins, areas, t.tolist())
        self.assertTrue(metrics.is_feasible, metrics)
        self.assertEqual(metrics.overlap_violations, 0)
        self.assertEqual(metrics.dimension_violations, 0)
        self.assertEqual(metrics.area_violations, 0)
        return result, metrics

    def test_preplaced_mib_group_and_corner(self):
        result, metrics = self.check([50, 50], [[1, 1, 1, 1, 9], [0, 0, 1, 1, 0]],
                                     [[0, 0, 10, 5], [-1, -1, -1, -1]],
                                     edges=[[0, 1, 2]], pins=[[0, 0]], pin_edges=[[0, 0, 1]])
        self.assertEqual(result[0], (0., 0., 10., 5.))
        self.assertEqual(result[1][2:], (10., 5.))
        self.assertEqual(metrics.grouping_violations, 0)
        self.assertEqual(metrics.mib_violations, 0)
        self.assertEqual(metrics.boundary_violations, 0)

    def test_fixed_shape_ignores_target_area(self):
        result, _ = self.check([999, 7, 11], [[1, 0, 0, 0, 0], [0]*5, [0]*5],
                              [[-1, -1, 8, 3], [-1]*4, [-1]*4])
        self.assertEqual(result[0][2:], (8., 3.))

    def test_unused_fixed_area_may_be_zero(self):
        result, _ = self.check([0, 7], [[1, 0, 0, 0, 0], [0]*5],
                              [[-1, -1, 8, 3], [-1]*4])
        self.assertEqual(result[0][2:], (8., 3.))

    def test_multiple_preplaced_obstacles(self):
        result, _ = self.check([16, 20, 9, 11, 7],
                              [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]] + [[0]*5]*3,
                              [[-5, -3, 4, 4], [4, 2, 5, 4]] + [[-1]*4]*3,
                              edges=[[0, 2, 1], [1, 3, 1], [2, 4, 2]])
        self.assertEqual(result[:2], [(-5., -3., 4., 4.), (4., 2., 5., 4.)])

    def test_incompatible_mib_still_respects_hard_area(self):
        self.check([9, 25], [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], [[-1]*4]*2)

    def test_missing_fixed_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "target_positions"):
            MyOptimizer(seconds=.1).solve(1, [16], [], [], [], [[1, 0, 0, 0, 0]])

    def test_impossible_preplaced_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preplaced"):
            MyOptimizer(seconds=.1).solve(2, [16, 16], [], [], [],
                                          [[0, 1, 0, 0, 0]]*2,
                                          [[0, 0, 4, 4], [1, 1, 4, 4]])

    def test_empty_design(self):
        self.assertEqual(MyOptimizer().solve(0, [], [], [], [], []), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
