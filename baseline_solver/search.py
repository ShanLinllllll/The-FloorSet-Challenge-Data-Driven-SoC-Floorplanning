from __future__ import annotations

from dataclasses import dataclass
import math
import time
import warnings
import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import coo_matrix

from .core import Problem, bbox, feasible, metrics, overlap_pairs, place_block, repair


@dataclass
class BaselineConfig:
    max_seconds: float = 2.0
    seed: int = 42
    lp_passes: int = 2
    sa_steps: int = 2400
    stagnation_steps: int = 250
    second_initialization: bool = True


def separation_relations(layout: np.ndarray) -> list[tuple[int, int, int]]:
    result = []
    for i in range(len(layout)):
        for j in range(i + 1, len(layout)):
            candidates = []
            for axis in (0, 1):
                for first, second in ((i, j), (j, i)):
                    gap = layout[second, axis] - layout[first, axis] - layout[first, axis + 2]
                    if gap >= -1e-10:
                        candidates.append(
                            (
                                max(gap, 0.0) / (layout[i, axis + 2] + layout[j, axis + 2]),
                                axis,
                                first,
                                second,
                            )
                        )
            if not candidates:
                raise ValueError("LP input layout has overlapping rectangles")
            _, axis, first, second = min(candidates)
            result.append((axis, first, second))
    return result


class Rows:
    def __init__(self):
        self.r, self.c, self.v, self.b = [], [], [], []

    def add(self, terms, rhs):
        row = len(self.b)
        for col, value in terms:
            self.r.append(row)
            self.c.append(col)
            self.v.append(value)
        self.b.append(rhs)

    def matrix(self, n):
        if not self.b:
            return None, None
        return coo_matrix((self.v, (self.r, self.c)), shape=(len(self.b), n)).tocsr(), np.asarray(
            self.b
        )


def group_contacts(problem: Problem, layout: np.ndarray):
    contacts = []
    for group in problem.groups:
        for p, i in enumerate(group):
            for j in group[p + 1 :]:
                for axis in (0, 1):
                    other = 1 - axis
                    depth = min(
                        layout[i, other] + layout[i, other + 2],
                        layout[j, other] + layout[j, other + 2],
                    ) - max(layout[i, other], layout[j, other])
                    if depth <= 0:
                        continue
                    for first, second in ((i, j), (j, i)):
                        if layout[first, axis] + layout[first, axis + 2] == layout[second, axis]:
                            contacts.append((axis, int(first), int(second), depth))
    return contacts


def compact_axis(problem: Problem, layout: np.ndarray, axis: int, time_limit: float, deadline=None):
    n = problem.n
    lo, hi = bbox(layout)
    wire_edges = [(int(i), int(j), float(w)) for i, j, w in problem.b2b if w > 0]
    pin_edges = [(int(p), int(i), float(w)) for p, i, w in problem.p2b if w > 0]
    variable_count = n + 2 + len(wire_edges) + len(pin_edges)
    c = np.zeros(variable_count)
    c[n] = -0.5 * (hi[1 - axis] - lo[1 - axis]) / problem.area_scale
    c[n + 1] = -c[n]
    bounds = (
        [(0.0, None)] * n + [(None, None), (None, None)] + [(0.0, None)] * (variable_count - n - 2)
    )
    for i in np.flatnonzero(problem.preplaced):
        coordinate = float(problem.targets[i, axis])
        bounds[i] = (coordinate, coordinate)
    ub, eq = Rows(), Rows()
    for i in range(n):
        ub.add([(n, 1.0), (i, -1.0)], 0.0)
        ub.add([(i, 1.0), (n + 1, -1.0)], -layout[i, axis + 2])
    for relation_axis, first, second in separation_relations(layout):
        if relation_axis == axis:
            ub.add([(first, 1.0), (second, -1.0)], -layout[first, axis + 2])
    contacts = group_contacts(problem, layout)
    for contact_axis, first, second, depth in contacts:
        if contact_axis == axis:
            eq.add([(second, 1.0), (first, -1.0)], layout[first, axis + 2])
        else:
            # Preserve a positive-length intersection along the shared edge.
            margin = min(depth * 0.25, problem.length_scale * 1e-5)
            ub.add([(first, 1.0), (second, -1.0)], layout[second, axis + 2] - margin)
            ub.add([(second, 1.0), (first, -1.0)], layout[first, axis + 2] - margin)
    low_bit, high_bit = (1, 2) if axis == 0 else (8, 4)
    for i, code in enumerate(problem.constraints[:, 4]):
        if code & low_bit and abs(layout[i, axis] - lo[axis]) < 1e-6:
            eq.add([(i, 1.0), (n, -1.0)], 0.0)
        if code & high_bit and abs(layout[i, axis] + layout[i, axis + 2] - hi[axis]) < 1e-6:
            eq.add([(i, 1.0), (n + 1, -1.0)], -layout[i, axis + 2])
    variable = n + 2
    for i, j, weight in wire_edges:
        offset = (layout[i, axis + 2] - layout[j, axis + 2]) / 2
        ub.add([(i, 1.0), (j, -1.0), (variable, -1.0)], -offset)
        ub.add([(i, -1.0), (j, 1.0), (variable, -1.0)], offset)
        c[variable] = 0.5 * weight / problem.wire_scale
        variable += 1
    for pin, i, weight in pin_edges:
        offset = problem.pins[pin, axis] - layout[i, axis + 2] / 2
        ub.add([(i, 1.0), (variable, -1.0)], offset)
        ub.add([(i, -1.0), (variable, -1.0)], -offset)
        c[variable] = 0.5 * weight / problem.wire_scale
        variable += 1
    a_ub, b_ub = ub.matrix(variable_count)
    a_eq, b_eq = eq.matrix(variable_count)
    if deadline is not None:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.005:
            return layout, "budget_exhausted"
        time_limit = min(time_limit, remaining)
    with warnings.catch_warnings():
        # SciPy forwards this documented HiGHS option to its native solver.
        warnings.filterwarnings(
            "ignore", message="Unrecognized options detected:.*threads", category=OptimizeWarning
        )
        result = linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={
                "time_limit": max(time_limit, 0.005),
                "threads": 1,
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
    if not result.success:
        return layout, f"solver_status_{result.status}"
    candidate = layout.copy()
    candidate[:, axis] = result.x[:n]
    candidate[problem.preplaced] = problem.targets[problem.preplaced]
    # Restore exact floating-point edge contacts when the LP equality drifted.
    for contact_axis, first, second, _ in contacts:
        if contact_axis != axis:
            continue
        if not problem.preplaced[second]:
            candidate[second, axis] = candidate[first, axis] + candidate[first, axis + 2]
        elif not problem.preplaced[first]:
            candidate[first, axis] = candidate[second, axis] - candidate[first, axis + 2]
    try:
        if len(overlap_pairs(candidate)):
            candidate = repair(problem, candidate)
    except (ValueError, RuntimeError):
        return layout, "numerical_repair_failed"
    if not feasible(problem, candidate):
        return layout, "hard_check_failed"
    if metrics(problem, candidate)["objective"] < metrics(problem, layout)["objective"] - 1e-10:
        return candidate, "accepted"
    return layout, "no_improvement"


def compact(problem: Problem, layout: np.ndarray, config: BaselineConfig, deadline: float):
    statuses = []
    for _ in range(config.lp_passes):
        changed = False
        for axis in (0, 1):
            remaining = deadline - time.perf_counter()
            if remaining < 0.025:
                return layout, statuses
            candidate, status = compact_axis(
                problem, layout, axis, min(0.2, remaining * 0.5), deadline
            )
            statuses.append(status)
            changed |= status == "accepted"
            layout = candidate
        if not changed:
            break
    return layout, statuses


def relocate(
    problem: Problem,
    layout: np.ndarray,
    blocks: np.ndarray,
    ideals: np.ndarray,
    new_dims: np.ndarray | None = None,
) -> np.ndarray:
    candidate = layout.copy()
    if new_dims is not None:
        candidate[blocks, 2:] = new_dims
    removed = set(int(i) for i in blocks)
    placed = [i for i in range(problem.n) if i not in removed]
    for block, ideal in zip(blocks, ideals):
        i = int(block)
        if problem.preplaced[i]:
            raise ValueError("SA cannot relocate a preplaced block")
        candidate[i] = place_block(
            problem, candidate, i, np.asarray(placed, int), ideal, grid_size=10
        )
        placed.append(i)
    return candidate


def anneal(
    problem: Problem,
    layout: np.ndarray,
    shape_units: list[np.ndarray],
    config: BaselineConfig,
    deadline: float,
):
    rng = np.random.default_rng(config.seed)
    current, best = layout.copy(), layout.copy()
    current_score = best_score = metrics(problem, layout)["objective"]
    stats = {
        "proposals": 0,
        "accepted": 0,
        "uphill_accepted": 0,
        "improvements": 0,
        "invalid_rejected": 0,
        "move_counts": {},
        "stop_reason": "no_movable_blocks",
    }
    if not len(problem.movable) or config.sa_steps <= 0:
        return best, stats
    started = time.perf_counter()
    budget = max(deadline - started, 1e-9)
    initial_temperature = max(current_score * 0.06, 0.01)
    deltas = []
    stale = 0
    for step in range(config.sa_steps):
        now = time.perf_counter()
        if now >= deadline:
            stats["stop_reason"] = "time_budget"
            break
        if stale >= config.stagnation_steps:
            stats["stop_reason"] = "stagnation"
            break
        centers = current[:, :2] + current[:, 2:] / 2
        move = float(rng.random())
        try:
            if move < 0.2 and len(shape_units):
                name = "reshape"
                blocks = shape_units[int(rng.integers(len(shape_units)))]
                area = float(np.prod(current[blocks[0], 2:]))
                log_ratio = float(
                    np.log(current[blocks[0], 2] / current[blocks[0], 3])
                ) + rng.normal(0, 0.35)
                ratio = float(np.exp(np.clip(log_ratio, math.log(0.05), math.log(20))))
                dims = np.tile([math.sqrt(area * ratio), math.sqrt(area / ratio)], (len(blocks), 1))
                candidate = relocate(problem, current, blocks, centers[blocks], dims)
            elif move < 0.35 and len(problem.movable) > 1:
                name = "swap"
                blocks = rng.choice(problem.movable, 2, replace=False)
                candidate = relocate(problem, current, blocks, centers[blocks[::-1]])
            elif move < 0.5 and len(problem.groups):
                name = "group_repair"
                group = problem.groups[int(rng.integers(len(problem.groups)))]
                blocks = group[~problem.preplaced[group]]
                if not len(blocks):
                    continue
                anchor = centers[group].mean(axis=0)
                ideals = np.tile(anchor, (len(blocks), 1))
                candidate = relocate(problem, current, blocks, ideals)
            else:
                name = "relocate"
                probabilities = (
                    1.0
                    + (problem.constraints[problem.movable, 4] != 0) * 0.5
                    + (problem.constraints[problem.movable, 3] > 0) * 0.5
                )
                i = int(rng.choice(problem.movable, p=probabilities / probabilities.sum()))
                fraction = min(1.0, (now - started) / budget)
                noise = rng.normal(0, problem.length_scale * (0.08 * (1 - fraction) + 0.01), size=2)
                ideal = centers[i] + noise
                if problem.constraints[i, 4] and rng.random() < 0.5:
                    lo, hi = bbox(current)
                    code = problem.constraints[i, 4]
                    if code & 1:
                        ideal[0] = lo[0] + current[i, 2] / 2
                    if code & 2:
                        ideal[0] = hi[0] - current[i, 2] / 2
                    if code & 4:
                        ideal[1] = hi[1] - current[i, 3] / 2
                    if code & 8:
                        ideal[1] = lo[1] + current[i, 3] / 2
                candidate = relocate(problem, current, np.array([i]), np.array([ideal]))
        except (ValueError, RuntimeError):
            stats["invalid_rejected"] += 1
            continue
        stats["proposals"] += 1
        stats["move_counts"][name] = stats["move_counts"].get(name, 0) + 1
        if not feasible(problem, candidate):
            stats["invalid_rejected"] += 1
            continue
        score = metrics(problem, candidate)["objective"]
        delta = score - current_score
        if step < 20 and abs(delta) > 1e-9:
            deltas.append(abs(delta))
            initial_temperature = max(float(np.median(deltas)) / -math.log(0.65), best_score * 0.01)
        fraction = max(step / max(config.sa_steps, 1), min(1.0, (now - started) / budget))
        temperature = initial_temperature * (0.01**fraction)
        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
            current, current_score = candidate, score
            stats["accepted"] += 1
            stats["uphill_accepted"] += int(delta > 1e-12)
        stale += 1
        if score < best_score - 1e-10:
            best, best_score = candidate.copy(), score
            stats["improvements"] += 1
            stale = 0
    else:
        stats["stop_reason"] = "step_limit"
    return best, stats
