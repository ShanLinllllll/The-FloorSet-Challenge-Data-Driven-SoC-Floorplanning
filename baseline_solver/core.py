from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def array(value, dtype=float) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).copy()


@dataclass
class Problem:
    n: int
    areas: np.ndarray
    constraints: np.ndarray
    targets: np.ndarray
    b2b: np.ndarray
    p2b: np.ndarray
    pins: np.ndarray

    @classmethod
    def from_inputs(
        cls,
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions=None,
    ):
        n = int(block_count)
        if n <= 0:
            raise ValueError("block_count must be positive")
        areas = array(area_targets)[:n]
        con = array(constraints, int)[:n]
        target = np.full((n, 4), -1.0) if target_positions is None else array(target_positions)[:n]
        pins = array(pins_pos).reshape(-1, 2)
        pins = pins[pins[:, 0] != -1]
        edges = []
        for raw in (b2b_connectivity, p2b_connectivity):
            edge = array(raw).reshape(-1, 3)
            edge = edge[edge[:, 0] >= 0]
            if not np.isfinite(edge).all() or (edge[:, 2] < 0).any():
                raise ValueError("Invalid connectivity")
            if not np.equal(edge[:, :2], np.round(edge[:, :2])).all():
                raise ValueError("Connectivity indices must be integers")
            edges.append(edge)
        if areas.shape != (n,) or con.shape != (n, 5) or target.shape != (n, 4):
            raise ValueError("Invalid block input shape")
        if not np.isfinite(areas).all() or (areas <= 0).any() or not np.isfinite(pins).all():
            raise ValueError("Areas and coordinates must be finite; areas must be positive")
        if len(edges[0]) and ((edges[0][:, :2] < 0).any() or (edges[0][:, :2] >= n).any()):
            raise ValueError("Block edge index out of range")
        if len(edges[1]) and (
            (edges[1][:, :2] < 0).any()
            or (edges[1][:, 0] >= len(pins)).any()
            or (edges[1][:, 1] >= n).any()
        ):
            raise ValueError("Terminal edge index out of range")
        obj = cls(n, areas, con, target, *edges, pins)
        obj.preplaced = con[:, 1] != 0
        obj.fixed = (con[:, 0] != 0) | obj.preplaced
        obj.movable = np.flatnonzero(~obj.preplaced)
        if not np.isfinite(target[obj.fixed, 2:]).all() or (target[obj.fixed, 2:] <= 0).any():
            raise ValueError("Fixed/preplaced dimensions must be supplied")
        if not np.isfinite(target[obj.preplaced, :2]).all():
            raise ValueError("Preplaced coordinates must be supplied")
        obj.effective_areas = areas.copy()
        obj.effective_areas[obj.fixed] = np.prod(target[obj.fixed, 2:], axis=1)
        obj.area_scale = float(obj.effective_areas.sum())
        obj.length_scale = math.sqrt(obj.area_scale)
        obj.wire_scale = max(
            float(sum(edge[:, 2].sum() for edge in edges)) * obj.length_scale * 0.25, 1e-12
        )
        obj.groups = [np.flatnonzero(con[:, 3] == g) for g in sorted(set(con[:, 3]) - {0, -1})]
        obj.mib = [np.flatnonzero(con[:, 2] == g) for g in sorted(set(con[:, 2]) - {0, -1})]
        obj.n_soft = max(
            1,
            int((con[:, 4] != 0).sum())
            + sum(len(g) - 1 for g in obj.groups)
            + sum(len(g) - 1 for g in obj.mib),
        )
        obj.weights = np.zeros((n, n))
        for i, j, weight in obj.b2b:
            i, j = int(i), int(j)
            obj.weights[i, j] += weight
            obj.weights[j, i] += weight
        obj.pin_weights = np.zeros((n, len(pins)))
        for pin, block, weight in obj.p2b:
            obj.pin_weights[int(block), int(pin)] += weight
        return obj


def bbox(layout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return layout[:, :2].min(axis=0), (layout[:, :2] + layout[:, 2:]).max(axis=0)


def overlap_pairs(layout: np.ndarray) -> np.ndarray:
    start, end = layout[:, :2], layout[:, :2] + layout[:, 2:]
    depth = np.minimum(end[:, None], end[None]) - np.maximum(start[:, None], start[None])
    return np.argwhere(np.triu((depth[:, :, 0] > 0) & (depth[:, :, 1] > 0), 1))


def feasible(problem: Problem, layout: np.ndarray) -> bool:
    if (
        layout.shape != (problem.n, 4)
        or not np.isfinite(layout).all()
        or (layout[:, 2:] <= 0).any()
    ):
        return False
    if not np.array_equal(layout[problem.fixed, 2:], problem.targets[problem.fixed, 2:]):
        return False
    if not np.array_equal(layout[problem.preplaced], problem.targets[problem.preplaced]):
        return False
    free = ~problem.fixed
    if (
        np.abs(np.prod(layout[free, 2:], axis=1) - problem.areas[free]) / problem.areas[free] > 0.01
    ).any():
        return False
    return not len(overlap_pairs(layout))


def soft_counts(problem: Problem, layout: np.ndarray) -> dict:
    lo, hi = bbox(layout)
    start, end = layout[:, :2], layout[:, :2] + layout[:, 2:]
    boundary = 0
    touches = np.column_stack(
        (
            np.abs(start[:, 0] - lo[0]) < 1e-6,
            np.abs(end[:, 0] - hi[0]) < 1e-6,
            np.abs(end[:, 1] - hi[1]) < 1e-6,
            np.abs(start[:, 1] - lo[1]) < 1e-6,
        )
    )
    for i, code in enumerate(problem.constraints[:, 4]):
        if code and any(code & bit and not touches[i, k] for k, bit in enumerate((1, 2, 4, 8))):
            boundary += 1
    grouping = 0
    for group in problem.groups:
        st, en = start[group], end[group]
        depth = np.minimum(en[:, None], en[None]) - np.maximum(st[:, None], st[None])
        # Exact shared edges match Shapely union semantics; corners do not connect.
        adj = ((depth[:, :, 0] >= 0) & (depth[:, :, 1] > 0)) | (
            (depth[:, :, 1] >= 0) & (depth[:, :, 0] > 0)
        )
        remaining = set(range(len(group)))
        components = 0
        while remaining:
            components += 1
            stack = [remaining.pop()]
            while stack:
                neighbors = set(np.flatnonzero(adj[stack.pop()])) & remaining
                remaining -= neighbors
                stack.extend(neighbors)
        grouping += max(0, components - 1)
    mib = sum(
        len({(round(float(layout[i, 2]), 4), round(float(layout[i, 3]), 4)) for i in g}) - 1
        for g in problem.mib
    )
    return {"boundary": boundary, "grouping": grouping, "mib": mib}


def metrics(problem: Problem, layout: np.ndarray) -> dict:
    centers = layout[:, :2] + layout[:, 2:] / 2
    wire = 0.0
    if len(problem.b2b):
        i, j = problem.b2b[:, 0].astype(int), problem.b2b[:, 1].astype(int)
        wire += float((np.abs(centers[i] - centers[j]).sum(axis=1) * problem.b2b[:, 2]).sum())
    if len(problem.p2b):
        pin, block = problem.p2b[:, 0].astype(int), problem.p2b[:, 1].astype(int)
        wire += float(
            (np.abs(problem.pins[pin] - centers[block]).sum(axis=1) * problem.p2b[:, 2]).sum()
        )
    lo, hi = bbox(layout)
    area = float(np.prod(hi - lo))
    violations = soft_counts(problem, layout)
    relative = sum(violations.values()) / problem.n_soft
    # Input-only search objective. Ground-truth baselines are deliberately unavailable.
    objective = (
        0.5 * (wire / problem.wire_scale + area / problem.area_scale) * math.exp(2 * relative)
    )
    return {
        "objective": objective,
        "hpwl": wire,
        "bbox_area": area,
        "soft_counts": violations,
        "violations_relative": relative,
    }


def initial_geometry(
    problem: Problem, initial_guess=None
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    n = problem.n
    if len(problem.pins):
        span = np.ptp(problem.pins, axis=0)
        aspect = float(np.clip((span[0] + 1e-6) / (span[1] + 1e-6), 0.25, 4))
    else:
        aspect = 1.0
    outline = np.sqrt(problem.area_scale * np.array([aspect, 1 / aspect]))
    ratios = np.full(n, aspect**0.35)
    # Allocate perimeter lengths before packing, instead of placing square edge
    # blocks in arbitrarily long strips. This is a heuristic, not a fixed outline.
    for bit, axis in ((1, 1), (2, 1), (4, 0), (8, 0)):
        side = (problem.constraints[:, 4] & bit) != 0
        single = (problem.constraints[:, 4] == bit) & ~problem.fixed
        if single.any():
            fixed_length = problem.targets[side & problem.fixed, axis + 2].sum()
            available = max(outline[axis] - fixed_length, outline[axis] * 0.3)
            length_sum = np.sqrt(problem.areas[side & ~problem.fixed]).sum()
            side_ratio = (available / max(length_sum, 1e-12)) ** 2
            ratios[single] = np.clip(side_ratio if axis == 0 else 1 / side_ratio, 0.1, 10)
    guess = None
    if initial_guess is not None:
        guess = array(initial_guess)
        if guess.shape != (n, 4) or not np.isfinite(guess).all() or (guess[:, 2:] <= 0).any():
            raise ValueError("initial_guess must contain finite positive (x,y,w,h) rectangles")
        ratios = np.clip(guess[:, 2] / guess[:, 3], 0.05, 20)
    dims = np.column_stack((np.sqrt(problem.areas * ratios), np.sqrt(problem.areas / ratios)))
    dims[problem.fixed] = problem.targets[problem.fixed, 2:]
    units = []
    covered = set()
    for group in problem.mib:
        fixed = group[problem.fixed[group]]
        free = group[~problem.fixed[group]]
        if not len(free):
            continue
        if len(fixed):
            shape = dims[fixed[0]].copy()
            compatible = (
                np.array_equal(dims[fixed], np.tile(shape, (len(fixed), 1)))
                and (
                    np.abs(np.prod(shape) - problem.areas[free]) / problem.areas[free] <= 0.01
                ).all()
            )
            if compatible:
                dims[free] = shape
                covered.update(free.tolist())  # A shared fixed master has immutable shape.
        else:
            low, high = max(problem.areas[group] * 0.99), min(problem.areas[group] * 1.01)
            if low <= high:
                common_area = float(np.clip(np.median(problem.areas[group]), low, high))
                ratio = float(np.exp(np.log(ratios[group]).mean()))
                dims[group] = [math.sqrt(common_area * ratio), math.sqrt(common_area / ratio)]
                units.append(group)
                covered.update(group.tolist())
    units.extend(np.array([i]) for i in range(n) if not problem.fixed[i] and i not in covered)
    if guess is not None:
        centers = guess[:, :2] + guess[:, 2:] / 2
    else:
        weights = problem.weights.copy()
        nonzero = weights[weights > 0]
        typical = float(np.mean(nonzero)) if len(nonzero) else 1.0
        for group in problem.groups:
            if len(group) > 1:
                for i, j in zip(group[:-1], group[1:]):
                    weights[i, j] += typical
                    weights[j, i] += typical
        degree = weights.sum(axis=1)
        pin_degree = problem.pin_weights.sum(axis=1)
        ridge = max(typical * 0.1, float(np.mean(degree + pin_degree)) * 0.03)
        matrix = np.diag(degree + pin_degree + ridge) - weights
        index = np.arange(n)
        prior = (
            outline / 2
            + np.column_stack((np.sin(index * 2.39996), np.cos(index * 2.39996))) * outline * 0.12
        )
        rhs = problem.pin_weights @ problem.pins + ridge * prior
        boundary_weight = np.maximum(degree + pin_degree, typical) * 0.2
        for i, code in enumerate(problem.constraints[:, 4]):
            for axis, low_bit, high_bit in ((0, 1, 2), (1, 8, 4)):
                if code & (low_bit | high_bit):
                    target = (
                        dims[i, axis] / 2 if code & low_bit else outline[axis] - dims[i, axis] / 2
                    )
                    # Axis-specific anchors are solved in separate systems below.
                    rhs[i, axis] += boundary_weight[i] * target
        centers = np.empty((n, 2))
        for axis, bits in ((0, 3), (1, 12)):
            axis_matrix = matrix.copy()
            active = (problem.constraints[:, 4] & bits) != 0
            axis_matrix[np.arange(n), np.arange(n)] += boundary_weight * active
            for i in np.flatnonzero(problem.preplaced):
                axis_matrix[i] = 0
                axis_matrix[i, i] = 1
                rhs[i, axis] = problem.targets[i, axis] + dims[i, axis] / 2
            centers[:, axis] = np.linalg.solve(axis_matrix, rhs[:, axis])
    centers[problem.preplaced] = (
        problem.targets[problem.preplaced, :2] + dims[problem.preplaced] / 2
    )
    return dims, centers, units


def placement_order(problem: Problem, dims: np.ndarray, variant=0) -> list[int]:
    degree = problem.weights.sum(axis=1) + problem.pin_weights.sum(axis=1)
    codes = problem.constraints[:, 4]
    score = np.log1p(np.prod(dims, axis=1)) + degree / max(degree.max(initial=1), 1e-12)
    if variant == 0:
        score += np.array([int(code).bit_count() * 4 for code in codes])
    else:
        score += (codes != 0) * 1.5
    pending = set(problem.movable.tolist())
    placed = set(np.flatnonzero(problem.preplaced).tolist())
    order = []
    while pending:

        def priority(i):
            group = problem.constraints[i, 3]
            friends = sum(problem.constraints[j, 3] == group for j in placed) if group > 0 else 0
            return score[i] + min(friends, 2) * 3, -i

        i = max(pending, key=priority)
        pending.remove(i)
        placed.add(i)
        order.append(i)
    return order


def place_block(
    problem: Problem,
    layout: np.ndarray,
    i: int,
    placed: np.ndarray,
    ideal: np.ndarray,
    grid_size=18,
    dims=None,
) -> np.ndarray:
    wh = layout[i, 2:] if dims is None else dims
    desired = np.maximum(ideal - wh / 2, 0)
    if not len(placed):
        return np.r_[desired, wh]
    others = layout[placed]
    lo, hi = bbox(others)
    xs = np.r_[
        desired[0],
        lo[0],
        hi[0] - wh[0],
        hi[0],
        others[:, 0],
        others[:, 0] + others[:, 2],
        others[:, 0] - wh[0],
        others[:, 0] + others[:, 2] - wh[0],
        0,
    ]
    ys = np.r_[
        desired[1],
        lo[1],
        hi[1] - wh[1],
        hi[1],
        others[:, 1],
        others[:, 1] + others[:, 3],
        others[:, 1] - wh[1],
        others[:, 1] + others[:, 3] - wh[1],
        0,
    ]
    xs, ys = np.unique(xs[xs >= 0]), np.unique(ys[ys >= 0])
    all_xs = xs.copy()
    if grid_size <= 12 and len(all_xs) > 48:
        nearby = np.argsort(np.abs(all_xs - desired[0]), kind="stable")[:32]
        spread = np.linspace(0, len(all_xs) - 1, 16, dtype=int)
        all_xs = all_xs[np.unique(np.r_[nearby, spread])]
    xs = xs[np.argsort(np.abs(xs - desired[0]), kind="stable")[:grid_size]]
    ys = ys[np.argsort(np.abs(ys - desired[1]), kind="stable")[:grid_size]]
    xg, yg = np.meshgrid(xs, ys)
    candidates = np.column_stack((xg.ravel(), yg.ravel()))
    # Scan every x frontier for its lowest legal y. A nearest-coordinate grid
    # alone can miss interior vacancies and repeatedly append exterior shelves.
    x_overlap = (all_xs[:, None] < others[None, :, 0] + others[None, :, 2]) & (
        all_xs[:, None] + wh[0] > others[None, :, 0]
    )
    pushed_y = np.zeros(len(all_xs))
    for _ in range(len(others) + 1):
        clashes = (
            x_overlap
            & (pushed_y[:, None] < others[None, :, 1] + others[None, :, 3])
            & (pushed_y[:, None] + wh[1] > others[None, :, 1])
        )
        affected = clashes.any(axis=1)
        if not affected.any():
            break
        tops = np.max(np.where(clashes, others[None, :, 1] + others[None, :, 3], 0.0), axis=1)
        pushed_y[affected] = tops[affected]
    candidates = np.vstack((candidates, np.column_stack((all_xs, pushed_y))))
    extra = [
        [hi[0], max(0.0, lo[1])],
        [max(0.0, lo[0]), hi[1]],
        [lo[0] - wh[0], desired[1]],
        [desired[0], lo[1] - wh[1]],
        [lo[0], lo[1]],
        [hi[0] - wh[0], lo[1]],
        [lo[0], hi[1] - wh[1]],
        [hi[0] - wh[0], hi[1] - wh[1]],
    ]
    group = problem.constraints[i, 3]
    friends = placed[problem.constraints[placed, 3] == group] if group > 0 else np.array([], int)
    for j in friends:
        x, y, w, h = layout[j]
        extra.extend([[x + w, y], [x - wh[0], y], [x, y + h], [x, y - wh[1]]])
    candidates = np.vstack((candidates, np.asarray(extra)))
    candidates = candidates[(candidates >= 0).all(axis=1)]
    starts = candidates[:, None, :]
    ends = starts + wh
    overlap = np.minimum(ends, others[None, :, :2] + others[None, :, 2:]) - np.maximum(
        starts, others[None, :, :2]
    )
    candidates = candidates[~((overlap[:, :, 0] > 0) & (overlap[:, :, 1] > 0)).any(axis=1)]
    if not len(candidates):
        # There is no fixed outline. An exterior shelf is always available.
        return np.r_[np.nextafter(max(0.0, hi[0]), np.inf), max(0.0, lo[1]), wh]
    centers = candidates + wh / 2
    other_centers = others[:, :2] + others[:, 2:] / 2
    wire = (
        np.abs(centers[:, None] - other_centers[None]).sum(axis=2) * problem.weights[i, placed]
    ).sum(axis=1)
    if len(problem.pins):
        wire += (
            np.abs(centers[:, None] - problem.pins[None]).sum(axis=2) * problem.pin_weights[i]
        ).sum(axis=1)
    new_lo, new_hi = np.minimum(candidates, lo), np.maximum(candidates + wh, hi)
    area = np.prod(new_hi - new_lo, axis=1)
    rank = 0.5 * (wire / problem.wire_scale + area / problem.area_scale)
    rank += 0.04 * np.abs(centers - ideal).sum(axis=1) / problem.length_scale
    code = problem.constraints[i, 4]
    touches = np.column_stack(
        (
            np.abs(candidates[:, 0] - new_lo[:, 0]) < 1e-6,
            np.abs(candidates[:, 0] + wh[0] - new_hi[:, 0]) < 1e-6,
            np.abs(candidates[:, 1] + wh[1] - new_hi[:, 1]) < 1e-6,
            np.abs(candidates[:, 1] - new_lo[:, 1]) < 1e-6,
        )
    )
    if code:
        required = [k for k, bit in enumerate((1, 2, 4, 8)) if code & bit]
        rank += 2 / problem.n_soft * (~touches[:, required].all(axis=1))
    if len(friends):
        fr = layout[friends]
        depth = np.minimum(
            candidates[:, None] + wh, fr[None, :, :2] + fr[None, :, 2:]
        ) - np.maximum(candidates[:, None], fr[None, :, :2])
        connected = ((depth[:, :, 0] >= 0) & (depth[:, :, 1] > 0)) | (
            (depth[:, :, 1] >= 0) & (depth[:, :, 0] > 0)
        )
        rank += 2 / problem.n_soft * (~connected.any(axis=1))
    best = int(np.argmin(rank))
    return np.r_[candidates[best], wh]


def legalize(problem: Problem, dims: np.ndarray, centers: np.ndarray, variant=0) -> np.ndarray:
    layout = np.column_stack((centers - dims / 2, dims))
    layout[problem.preplaced] = problem.targets[problem.preplaced]
    placed = list(np.flatnonzero(problem.preplaced))
    if len(placed) and len(overlap_pairs(layout[placed])):
        raise ValueError("Preplaced blocks overlap; immutable inputs cannot be legalized")
    for i in placement_order(problem, dims, variant):
        layout[i] = place_block(problem, layout, i, np.asarray(placed, int), centers[i])
        placed.append(i)
    if not feasible(problem, layout):
        raise RuntimeError("Legalization failed a hard constraint")
    return layout


def repair(problem: Problem, layout: np.ndarray) -> np.ndarray:
    layout = layout.copy()
    layout[problem.preplaced] = problem.targets[problem.preplaced]
    for _ in range(problem.n * 2):
        pairs = overlap_pairs(layout)
        if not len(pairs):
            return layout
        i, j = pairs[0]
        block = int(j if not problem.preplaced[j] else i)
        if problem.preplaced[block]:
            raise ValueError("Preplaced overlap cannot be repaired")
        other = np.delete(np.arange(problem.n), block)
        ideal = layout[block, :2] + layout[block, 2:] / 2
        layout[block] = place_block(problem, layout, block, other, ideal, grid_size=12)
    raise RuntimeError("Overlap repair did not converge")
