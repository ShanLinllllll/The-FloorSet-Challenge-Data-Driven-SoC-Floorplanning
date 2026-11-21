"""Official solve() adapter for heuristic -> legalization -> LP -> short SA."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FloorSet" / "iccad2026contest"))
from iccad2026_evaluate import FloorplanOptimizer

from baseline_solver.core import Problem, feasible, initial_geometry, legalize, metrics
from baseline_solver.search import BaselineConfig, anneal, compact


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose=False, config: BaselineConfig | None = None):
        super().__init__(verbose)
        self.config = config or BaselineConfig()
        self.last_run = {}

    def solve(
        self,
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions=None,
    ):
        return self.solve_with_initial_guess(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )

    def solve_with_initial_guess(
        self,
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions=None,
        initial_guess=None,
    ):
        """A future GNN may supply N x 4 xywh predictions; hard geometry is projected."""
        started = time.perf_counter()
        deadline = started + max(self.config.max_seconds, 0.0)
        problem = Problem.from_inputs(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )
        dims, centers, units = initial_geometry(problem, initial_guess)
        after_initialization = time.perf_counter()
        layout = legalize(problem, dims, centers)
        if (
            self.config.second_initialization
            and time.perf_counter() < started + self.config.max_seconds * 0.15
        ):
            alternative = legalize(problem, dims, centers, variant=1)
            if metrics(problem, alternative)["objective"] < metrics(problem, layout)["objective"]:
                layout = alternative
        legal_layout = layout.copy()
        after_legalization = time.perf_counter()
        # Reserve time for SA; legalization itself completes even on a tiny budget.
        lp_deadline = min(
            deadline,
            time.perf_counter() + min(0.45, max(0.0, deadline - time.perf_counter()) * 0.35),
        )
        layout, lp_statuses = compact(problem, layout, self.config, lp_deadline)
        lp_layout = layout.copy()
        after_lp = time.perf_counter()
        layout, sa_stats = anneal(problem, layout, units, self.config, deadline)
        if not feasible(problem, layout):
            raise RuntimeError("Final hard-constraint check failed")
        ended = time.perf_counter()
        self.last_run = {
            "config": asdict(self.config),
            "initial_guess_supplied": initial_guess is not None,
            "stage_seconds": {
                "initialization": after_initialization - started,
                "legalization": after_legalization - after_initialization,
                "lp": after_lp - after_legalization,
                "sa": ended - after_lp,
                "total": ended - started,
            },
            "lp_statuses": lp_statuses,
            "sa": sa_stats,
            "stage_metrics": {
                "legalization": metrics(problem, legal_layout),
                "lp": metrics(problem, lp_layout),
                "sa": metrics(problem, layout),
            },
            "layouts": {"legalization": legal_layout, "lp": lp_layout, "sa": layout.copy()},
        }
        return [tuple(float(v) for v in row) for row in layout]
