"""Constraint-aware FloorSet solver, using only the public solve() inputs.

Analytical placement -> legal rectangle packing -> linear-program compaction
-> legal local refinement. Fixed/preplaced geometry is never resized/moved.
Set FLOORSET_SECONDS to change the per-design search budget (default 3 s).
No reference layouts, dataset files, or generator internals are read here.
"""
import math
import os
import time

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

from iccad2026_evaluate import FloorplanOptimizer


GRID = 2.0 ** -20


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _snap(value):
    # Dyadic coordinates keep abutting edges equal under binary arithmetic.
    return np.rint(np.asarray(value) / GRID) * GRID


def _components(rects):
    n = len(rects)
    if n < 2:
        return n
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lo = rects[:, :2]
    hi = lo + rects[:, 2:]
    for i in range(n):
        for j in range(i):
            overlap = np.minimum(hi[i], hi[j]) - np.maximum(lo[i], lo[j])
            touch = ((abs(overlap[0]) < 1e-10 and overlap[1] > 1e-10) or
                     (abs(overlap[1]) < 1e-10 and overlap[0] > 1e-10))
            if touch:
                parent[root(i)] = root(j)
    return len({root(i) for i in range(n)})


class _Problem:
    def __init__(self, n, areas, b2b, p2b, pins, constraints, targets):
        self.n = n
        self.areas = _array(areas).reshape(-1)[:n].copy()
        self.constraints = np.zeros((n, 5), dtype=int)
        if constraints is not None:
            c = _array(constraints)
            self.constraints[:, :min(c.shape[1], 5)] = c[:n, :5].astype(int)
        self.fixed = np.any(self.constraints[:, :2] != 0, axis=1)
        self.pre = self.constraints[:, 1] != 0
        self.targets = np.full((n, 4), -1.0) if targets is None else _array(targets)[:n].copy()
        if np.any(self.fixed) and (targets is None or
                                  np.any(~np.isfinite(self.targets[self.fixed, 2:])) or
                                  np.any(self.targets[self.fixed, 2:] <= 0)):
            raise ValueError("Fixed/preplaced blocks require target_positions dimensions")
        if np.any(self.pre) and np.any(~np.isfinite(self.targets[self.pre])):
            raise ValueError("Preplaced blocks require finite target_positions")
        self.areas[self.fixed] = np.prod(self.targets[self.fixed, 2:], axis=1)
        if np.any(~np.isfinite(self.areas)) or np.any(self.areas <= 0):
            raise ValueError("Soft blocks must have a finite positive target area")
        self.total_area = float(self.areas.sum())
        self.scale = math.sqrt(self.total_area)
        self.pins = _array(pins).reshape(-1, 2)
        if len(self.pins):
            frame_lo, frame_hi = self.pins.min(0), self.pins.max(0)
        else:
            frame_lo, frame_hi = np.zeros(2), np.full(2, self.scale)
        if np.any(self.pre):
            frame_lo = np.minimum(frame_lo, self.targets[self.pre, :2].min(0))
            frame_hi = np.maximum(frame_hi, (self.targets[self.pre, :2] + self.targets[self.pre, 2:]).max(0))
        frame_span = np.maximum(frame_hi - frame_lo, self.scale * .2)
        frame_span *= max(1., math.sqrt(self.total_area * 1.15 / np.prod(frame_span)))
        self.frame = np.array([_snap(frame_lo), _snap(frame_lo + frame_span)])
        # An immutable block with a boundary requirement supplies a useful
        # frame edge. This is an input constraint, not reference-layout data.
        for axis, bits in enumerate(((1, 2), (8, 4))):
            for side, bit in enumerate(bits):
                indices = np.flatnonzero(self.pre & ((self.constraints[:, 4] & bit) != 0))
                if len(indices):
                    values = self.targets[indices, axis] + side * self.targets[indices, axis + 2]
                    if np.ptp(values) < 1e-8:
                        value = float(values[0])
                        immutable_lo = self.targets[self.pre, axis].min()
                        immutable_hi = (self.targets[self.pre, axis] + self.targets[self.pre, axis + 2]).max()
                        if (side == 0 and value <= immutable_lo) or (side == 1 and value >= immutable_hi):
                            self.frame[side, axis] = value
        self.edges = self._edges(b2b, n, n)
        self.pin_edges = self._edges(p2b, len(self.pins), n)
        self.W = np.zeros((n, n))
        for i, j, w in self.edges:
            self.W[int(i), int(j)] += w
            self.W[int(j), int(i)] += w
        self.P = np.zeros((n, len(self.pins)))
        for pin, block, w in self.pin_edges:
            self.P[int(block), int(pin)] += w
        self.groups = [np.flatnonzero(self.constraints[:, 3] == g)
                       for g in np.unique(self.constraints[:, 3]) if g > 0]
        self.mibs = [np.flatnonzero(self.constraints[:, 2] == g)
                     for g in np.unique(self.constraints[:, 2]) if g > 0]
        self.nsoft = max(1, int(np.count_nonzero(self.constraints[:, 4])) +
                         sum(len(g) - 1 for g in self.groups + self.mibs))
        total_weight = float(self.edges[:, 2].sum() + self.pin_edges[:, 2].sum())
        # A scale estimate derived from inputs, not from hidden reference metrics.
        self.wl_scale = max(1e-6, total_weight * self.scale / math.sqrt(n) * 1.7)
        self.rng = np.random.default_rng(20261008 + n)

    @staticmethod
    def _edges(value, first, second):
        if value is None:
            return np.empty((0, 3))
        a = _array(value).reshape(-1, 3)
        mask = ((a[:, 0] >= 0) & (a[:, 0] < first) & (a[:, 1] >= 0) &
                (a[:, 1] < second) & (a[:, 2] > 0) & np.all(np.isfinite(a), axis=1))
        return a[mask]

    def shapes(self, restart):
        ratios = np.zeros(self.n) if restart == 0 else self.rng.normal(0, .45, self.n)
        dims = np.empty((self.n, 2))
        dims[:, 0] = _snap(np.sqrt(self.areas) * np.exp(ratios / 2))
        dims[:, 0] = np.maximum(dims[:, 0], GRID)
        dims[:, 1] = np.maximum(_snap(self.areas / dims[:, 0]), GRID)
        dims[self.fixed] = self.targets[self.fixed, 2:]
        for group in self.mibs:
            locked = group[self.fixed[group]]
            if len(locked):
                shape = dims[locked[0]].copy()
            else:
                a = float(np.median(self.areas[group]))
                w = max(GRID, float(_snap(math.sqrt(a) * math.exp(ratios[group[0]] / 2))))
                shape = np.array([w, max(GRID, float(_snap(a / w)))])
            fits = abs(np.prod(shape) - self.areas[group]) / self.areas[group] <= .009
            dims[group[fits & ~self.fixed[group]]] = shape
        return dims

    def legal(self, rects):
        if rects.shape != (self.n, 4) or not np.all(np.isfinite(rects)) or np.any(rects[:, 2:] <= 0):
            return False
        if np.any(rects[self.fixed, 2:] != self.targets[self.fixed, 2:]):
            return False
        if np.any(rects[self.pre] != self.targets[self.pre]):
            return False
        if np.any(abs(np.prod(rects[~self.fixed, 2:], axis=1) - self.areas[~self.fixed]) /
                  self.areas[~self.fixed] > .01):
            return False
        overlap = (np.minimum((rects[:, :2] + rects[:, 2:])[:, None],
                              (rects[:, :2] + rects[:, 2:])[None, :]) -
                   np.maximum(rects[:, None, :2], rects[None, :, :2]))
        return not np.any(np.triu(np.all(overlap > 0, axis=2), 1))

    def violations(self, rects):
        lower = rects[:, :2].min(0)
        upper = (rects[:, :2] + rects[:, 2:]).max(0)
        contacts = np.column_stack((abs(rects[:, 0] - lower[0]) < 1e-6,
                                    abs(rects[:, 0] + rects[:, 2] - upper[0]) < 1e-6,
                                    abs(rects[:, 1] + rects[:, 3] - upper[1]) < 1e-6,
                                    abs(rects[:, 1] - lower[1]) < 1e-6))
        codes = self.constraints[:, 4]
        violations = int(np.count_nonzero(np.any(
            ((codes[:, None] & np.array([1, 2, 4, 8])) != 0) & ~contacts, axis=1)))
        for group in self.groups:
            violations += _components(rects[group]) - 1
        for group in self.mibs:
            violations += len({tuple(np.round(d, 4)) for d in rects[group, 2:]}) - 1
        return violations

    def score(self, rects):
        centers = rects[:, :2] + rects[:, 2:] / 2
        e = self.edges
        hpwl = float(np.sum(e[:, 2] * np.abs(centers[e[:, 0].astype(int)] -
                                              centers[e[:, 1].astype(int)]).sum(1)))
        e = self.pin_edges
        hpwl += float(np.sum(e[:, 2] * np.abs(self.pins[e[:, 0].astype(int)] -
                                               centers[e[:, 1].astype(int)]).sum(1)))
        span = (rects[:, :2] + rects[:, 2:]).max(0) - rects[:, :2].min(0)
        quality = .5 * hpwl / self.wl_scale + .5 * np.prod(span) / self.total_area
        return math.log(max(quality, .5)) + 2 * self.violations(rects) / self.nsoft

    def analytical(self, dims, restart, deadline):
        n, s = self.n, self.scale
        W = self.W.copy()
        typical = max(float(W.sum()) / max(n, 1), 1e-3)
        for group in self.groups:
            W[np.ix_(group, group)] += typical / max(len(group), 1) * .35
        np.fill_diagonal(W, 0)
        P = self.P
        pin_mean = self.pins.mean(0) if len(self.pins) else np.array([s / 2, s / 2])
        L = np.diag(W.sum(1) + P.sum(1) + typical * .015) - W
        rhs = P @ self.pins + typical * .015 * pin_mean
        movable = np.flatnonzero(~self.pre)
        pinned = np.flatnonzero(self.pre)
        desired = np.tile(pin_mean, (n, 1))
        desired[pinned] = self.targets[pinned, :2] + dims[pinned] / 2
        if len(movable):
            rhs_m = rhs[movable] - L[np.ix_(movable, pinned)] @ desired[pinned]
            desired[movable] = np.linalg.solve(L[np.ix_(movable, movable)], rhs_m)
        desired[movable] += self.rng.normal(0, s * (.03 if restart == 0 else .12), (len(movable), 2))
        if not len(movable):
            return desired
        centers = desired / s
        normalized_dims = dims / s
        weight_sum = max(self.W.sum() / 2 + self.P.sum(), 1e-9)
        edges = self.edges
        ei, ej, ew = edges[:, 0].astype(int), edges[:, 1].astype(int), edges[:, 2] / weight_sum
        pe = self.pin_edges
        pi, pb, pw = pe[:, 0].astype(int), pe[:, 1].astype(int), pe[:, 2] / weight_sum
        pins_n = self.pins / s
        half = (normalized_dims[:, None] + normalized_dims[None, :]) / 2

        def objective(vector, penalty):
            if time.perf_counter() >= deadline:
                raise TimeoutError
            c = centers.copy()
            c[movable] = vector.reshape(-1, 2)
            grad = np.zeros_like(c)
            diff = c[ei] - c[ej]
            smooth = np.sqrt(diff * diff + .002 ** 2)
            value = float(np.sum(ew[:, None] * smooth))
            force = ew[:, None] * diff / smooth
            np.add.at(grad, ei, force)
            np.add.at(grad, ej, -force)
            diff = c[pb] - pins_n[pi]
            smooth = np.sqrt(diff * diff + .002 ** 2)
            value += float(np.sum(pw[:, None] * smooth))
            np.add.at(grad, pb, pw[:, None] * diff / smooth)
            delta = c[:, None] - c[None, :]
            penetration = np.maximum(half - np.abs(delta), 0)
            penetration[np.arange(n), np.arange(n)] = 0
            value += penalty * float(np.prod(penetration, axis=2).sum()) / 2
            force = -np.sign(delta) * penetration[:, :, ::-1] * (penetration > 0)
            grad += penalty * force.sum(1)
            # A compactness term keeps disconnected blocks near the design.
            centered = c - pin_mean / s
            value += .04 * float(np.sum(centered * centered)) / n
            grad += .08 * centered / n
            return value, grad[movable].ravel()

        for penalty in (8., 40., 180.):
            try:
                result = minimize(objective, centers[movable].ravel(), args=(penalty,), jac=True,
                                  method="L-BFGS-B", options={"maxiter": 65, "ftol": 1e-7, "maxls": 15})
                centers[movable] = result.x.reshape(-1, 2)
            except TimeoutError:
                break
        return centers * s

    def candidates(self, i, dims, rects, placed, desired, max_axis=13):
        w, h = dims[i]
        obstacles = rects[placed]
        center = desired[i]
        xy0 = _snap(center - dims[i] / 2)
        if len(obstacles):
            lo, hi = obstacles[:, :2], obstacles[:, :2] + obstacles[:, 2:]
            axes = []
            for axis, size in enumerate((w, h)):
                options = np.unique(np.r_[lo[:, axis], hi[:, axis], lo[:, axis] - size,
                                         hi[:, axis] - size, xy0[axis]])
                selected = options[np.argsort(abs(options - xy0[axis]))[:max_axis]]
                axes.append(np.unique(np.r_[selected, lo[:, axis].min(), hi[:, axis].max() - size,
                                            self.frame[0, axis], self.frame[1, axis] - size]))
            xx, yy = np.meshgrid(*axes)
            candidates = np.column_stack((xx.ravel(), yy.ravel()))
            # Include every side contact, even when it is far from the target.
            contacts = np.vstack((np.column_stack((hi[:, 0], lo[:, 1])),
                                  np.column_stack((lo[:, 0] - w, lo[:, 1])),
                                  np.column_stack((lo[:, 0], hi[:, 1])),
                                  np.column_stack((lo[:, 0], lo[:, 1] - h))))
            candidates = np.unique(np.vstack((candidates, contacts,
                                              [hi[:, 0].max(), xy0[1]],
                                              [xy0[0], hi[:, 1].max()])), axis=0)
            intersect = np.minimum(candidates[:, None] + [w, h], hi[None]) - np.maximum(candidates[:, None], lo[None])
            candidates = candidates[~np.any(np.all(intersect > 0, axis=2), axis=1)]
        else:
            candidates = np.array([xy0, _snap(center), [0., 0.]])
        return candidates

    def candidate_costs(self, i, dims, candidates, rects, placed, desired):
        centers = candidates + dims[i] / 2
        neighbor_centers = desired.copy()
        neighbor_centers[placed] = rects[placed, :2] + rects[placed, 2:] / 2
        weights = self.W[i].copy()
        weights[~placed] *= .45
        cost = .5 * (np.abs(centers[:, None] - neighbor_centers[None]).sum(2) @ weights +
                      np.abs(centers[:, None] - self.pins[None]).sum(2) @ self.P[i]) / self.wl_scale
        obs = rects[placed]
        if len(obs):
            lo = np.minimum(candidates, obs[:, :2].min(0))
            hi = np.maximum(candidates + dims[i], (obs[:, :2] + obs[:, 2:]).max(0))
        else:
            lo, hi = candidates, candidates + dims[i]
        cost += .5 * np.prod(hi - lo, axis=1) / self.total_area
        cost += .015 * np.sum((centers - desired[i]) ** 2, axis=1) / self.total_area
        indices = np.r_[np.flatnonzero(placed), i]
        codes = self.constraints[indices, 4]
        if np.any(codes):
            required = (codes[:, None] & np.array([1, 2, 4, 8])) != 0
            lower = np.broadcast_to(rects[indices, :2], (len(candidates), len(indices), 2)).copy()
            upper = lower + dims[indices]
            lower[:, -1], upper[:, -1] = candidates, candidates + dims[i]
            touched = np.stack((abs(lower[:, :, 0] - lo[:, None, 0]) < 1e-6,
                                abs(upper[:, :, 0] - hi[:, None, 0]) < 1e-6,
                                abs(upper[:, :, 1] - hi[:, None, 1]) < 1e-6,
                                abs(lower[:, :, 1] - lo[:, None, 1]) < 1e-6), axis=2)
            cost += 2 * np.any(required[None] & ~touched, axis=2).sum(1) / self.nsoft
        group_id = self.constraints[i, 3]
        same = placed & (self.constraints[:, 3] == group_id) if group_id > 0 else np.zeros(self.n, bool)
        if np.any(same):
            other = rects[same]
            overlap = (np.minimum(candidates[:, None] + dims[i], (other[:, :2] + other[:, 2:])[None]) -
                       np.maximum(candidates[:, None], other[None, :, :2]))
            connected = (((abs(overlap[:, :, 0]) < 1e-10) & (overlap[:, :, 1] > 1e-10)) |
                         ((abs(overlap[:, :, 1]) < 1e-10) & (overlap[:, :, 0] > 1e-10)))
            cost -= 2 * np.any(connected, axis=1) / self.nsoft
        return cost

    def pack(self, dims, desired, framed=False):
        rects = np.column_stack((np.zeros((self.n, 2)), dims)).copy()
        rects[self.pre] = self.targets[self.pre]
        placed = self.pre.copy()
        while not np.all(placed):
            priorities = np.sqrt(self.areas) / self.scale + self.W[:, placed].sum(1) / max(self.W.sum(), 1)
            codes = self.constraints[:, 4]
            priorities += (codes > 0) * 1.5
            priorities += np.array([int(int(c).bit_count() >= 2) for c in codes]) * 2
            for group in self.groups:
                if np.any(placed[group]):
                    priorities[group] += 5
            priorities[placed] = -np.inf
            i = int(np.argmax(priorities))
            candidates = self.candidates(i, dims, rects, placed, desired)
            if framed:
                inside = np.all((candidates >= self.frame[0]) &
                                (candidates + dims[i] <= self.frame[1]), axis=1)
                if np.any(inside):
                    candidates = candidates[inside]
                    code = int(codes[i])
                    boundary = np.ones(len(candidates), dtype=bool)
                    for axis, bits in enumerate(((1, 2), (8, 4))):
                        if code & bits[0]:
                            boundary &= candidates[:, axis] == self.frame[0, axis]
                        if code & bits[1]:
                            boundary &= candidates[:, axis] + dims[i, axis] == self.frame[1, axis]
                    if np.any(boundary):
                        candidates = candidates[boundary]
            values = self.candidate_costs(i, dims, candidates, rects, placed, desired)
            rects[i, :2] = candidates[int(np.argmin(values))]
            placed[i] = True
        return rects

    def compact(self, rects, deadline):
        """Optimize coordinates with separation topology and satisfied contacts held."""
        original = rects.copy()
        n = self.n
        lo, hi = rects[:, :2], rects[:, :2] + rects[:, 2:]
        separators = [[], []]
        group_links = [[], []]
        for i in range(n):
            for j in range(i):
                possibilities = []
                for axis in (0, 1):
                    if hi[i, axis] <= lo[j, axis]:
                        possibilities.append((axis, i, j, lo[j, axis] - hi[i, axis]))
                    if hi[j, axis] <= lo[i, axis]:
                        possibilities.append((axis, j, i, lo[i, axis] - hi[j, axis]))
                if not possibilities:
                    return original
                same_group = (self.constraints[i, 3] > 0 and
                              self.constraints[i, 3] == self.constraints[j, 3])
                touching = [p for p in possibilities if p[3] == 0 and
                            min(hi[i, 1 - p[0]], hi[j, 1 - p[0]]) > max(lo[i, 1 - p[0]], lo[j, 1 - p[0]])]
                selected = touching[0] if touching else max(possibilities, key=lambda p: p[3] / max(rects[i, p[0] + 2], rects[j, p[0] + 2]))
                axis, a, b, _ = selected
                separators[axis].append((a, b))
                if same_group and touching:
                    axis, a, b, _ = touching[0]
                    group_links[axis].append((a, b))
        for axis in (0, 1):
            remaining = deadline - time.perf_counter()
            if remaining < .03:
                break
            size = rects[:, axis + 2]
            min_index, max_index = n, n + 1
            edge_start, pin_start = n + 2, n + 2 + len(self.edges)
            nv = pin_start + len(self.pin_edges)
            costs = np.zeros(nv)
            other_span = (rects[:, 1 - axis] + rects[:, 3 - axis]).max() - rects[:, 1 - axis].min()
            costs[min_index], costs[max_index] = -.5 * other_span / self.total_area, .5 * other_span / self.total_area
            costs[edge_start:pin_start] = .5 * self.edges[:, 2] / self.wl_scale
            costs[pin_start:] = .5 * self.pin_edges[:, 2] / self.wl_scale
            rows, cols, vals, rhs = [], [], [], []
            erows, ecols, evals, erhs = [], [], [], []

            def add(terms, bound, equality=False):
                rr, cc, vv, bb = (erows, ecols, evals, erhs) if equality else (rows, cols, vals, rhs)
                r = len(bb)
                for k, v in terms:
                    rr.append(r); cc.append(k); vv.append(v)
                bb.append(float(bound))

            for i in range(n):
                add([(min_index, 1), (i, -1)], 0)
                add([(i, 1), (max_index, -1)], -size[i])
            for a, b in separators[axis]:
                add([(a, 1), (b, -1)], -size[a])
            for a, b in group_links[axis]:
                add([(a, 1), (b, -1)], -size[a], True)
            for a, b in group_links[1 - axis]:
                contact_length = min(hi[a, axis], hi[b, axis]) - max(lo[a, axis], lo[b, axis])
                contact_length = min(contact_length, min(size[a], size[b]) * .1)
                add([(a, 1), (b, -1)], size[b] - contact_length)
                add([(b, 1), (a, -1)], size[a] - contact_length)
            for k, (i0, j0, _) in enumerate(self.edges):
                i, j, d = int(i0), int(j0), edge_start + k
                add([(i, 1), (j, -1), (d, -1)], (size[j] - size[i]) / 2)
                add([(j, 1), (i, -1), (d, -1)], (size[i] - size[j]) / 2)
            for k, (pin0, block0, _) in enumerate(self.pin_edges):
                pin, block, d = int(pin0), int(block0), pin_start + k
                coordinate = self.pins[pin, axis] - size[block] / 2
                add([(block, 1), (d, -1)], coordinate)
                add([(block, -1), (d, -1)], -coordinate)
            for i, code in enumerate(self.constraints[:, 4]):
                lower_bit, upper_bit = (1, 2) if axis == 0 else (8, 4)
                if code & lower_bit and lo[i, axis] == lo[:, axis].min():
                    add([(i, 1), (min_index, -1)], 0, True)
                if code & upper_bit and hi[i, axis] == hi[:, axis].max():
                    add([(i, 1), (max_index, -1)], -size[i], True)
            bounds = [(None, None)] * (n + 2) + [(0, None)] * (nv - n - 2)
            for i in np.flatnonzero(self.pre):
                bounds[i] = (self.targets[i, axis], self.targets[i, axis])
            if not np.any(self.pre) and not len(self.pin_edges):
                bounds[min_index] = (0, 0)
            A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), nv)).tocsr()
            E = coo_matrix((evals, (erows, ecols)), shape=(len(erhs), nv)).tocsr() if erhs else None
            result = linprog(costs, A_ub=A, b_ub=rhs, A_eq=E, b_eq=erhs or None,
                             bounds=bounds, method="highs", options={"time_limit": min(1., remaining)})
            if result.success:
                trial = rects.copy()
                trial[:, axis] = _snap(result.x[:n])
                trial[self.pre] = self.targets[self.pre]
                if self.legal(trial):
                    rects = trial
        return rects if self.score(rects) <= self.score(original) + 1e-10 else original

    def refine(self, rects, deadline):
        best = rects.copy()
        best_score = self.score(best)
        for sweep in range(2):
            order = self.rng.permutation(np.flatnonzero(~self.pre))
            improved = False
            for i in order:
                if time.perf_counter() >= deadline:
                    return best
                dims = best[:, 2:].copy()
                placed = np.ones(self.n, bool); placed[i] = False
                desired = best[:, :2] + dims / 2
                connected = self.W[i]
                weight = connected.sum() + self.P[i].sum()
                if weight > 0:
                    desired[i] = (connected @ desired + self.P[i] @ self.pins) / weight
                shapes = [dims[i].copy()]
                if not self.fixed[i] and self.constraints[i, 2] == 0:
                    for factor in (.78, 1.28):
                        width = max(GRID, float(_snap(dims[i, 0] * factor)))
                        shapes.append(np.array([width, max(GRID, float(_snap(self.areas[i] / width)))]))
                for shape in shapes:
                    if time.perf_counter() >= deadline:
                        return best
                    dims[i] = shape
                    candidates = self.candidates(i, dims, best, placed, desired, 10)
                    values = self.candidate_costs(i, dims, candidates, best, placed, desired)
                    for choice in np.argsort(values)[:4]:
                        trial = best.copy(); trial[i] = np.r_[candidates[choice], shape]
                        score = self.score(trial)
                        if score < best_score - 1e-9:
                            best, best_score = trial, score
                            improved = True
            if not improved:
                break
        return best


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose=False, seconds=None):
        super().__init__(verbose)
        self.seconds = max(.1, float(seconds if seconds is not None else os.environ.get("FLOORSET_SECONDS", "3")))

    def solve(self, block_count, area_targets, b2b_connectivity,
              p2b_connectivity, pins_pos, constraints, target_positions=None):
        if block_count == 0:
            return []
        start = time.perf_counter()
        deadline = start + self.seconds
        problem = _Problem(block_count, area_targets, b2b_connectivity,
                           p2b_connectivity, pins_pos, constraints, target_positions)
        # Construct a legal answer before starting time-bounded optimization.
        dims = problem.shapes(0)
        desired = problem.analytical(dims, 0, min(deadline, start + self.seconds * .24))
        best = problem.pack(dims, desired)
        if not problem.legal(best):
            raise ValueError("No valid packing: check immutable preplaced rectangles for overlap")
        best_score = problem.score(best)
        if time.perf_counter() < start + self.seconds * .6:
            framed = problem.pack(dims, desired, framed=True)
            if problem.legal(framed) and problem.score(framed) < best_score:
                best, best_score = framed, problem.score(framed)
        for restart in range(2):
            if restart:
                if time.perf_counter() > start + self.seconds * .60:
                    break
                dims = problem.shapes(restart)
                desired = problem.analytical(dims, restart, min(deadline, time.perf_counter() + self.seconds * .14))
                current = problem.pack(dims, desired)
            else:
                current = best.copy()
            current = problem.compact(current, min(deadline, start + self.seconds * .76))
            if problem.legal(current):
                score = problem.score(current)
                if score < best_score:
                    best, best_score = current, score
        if time.perf_counter() < deadline:
            best = problem.refine(best, deadline)
        if not problem.legal(best):
            raise RuntimeError("Internal error: final hard-constraint check failed")
        return [tuple(map(float, row)) for row in best]
