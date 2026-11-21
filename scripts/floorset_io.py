"""Separate solver-visible inputs from reference labels, matching the contest API."""

from __future__ import annotations

import torch


def reference_positions(sample: dict, kind: str) -> list[tuple[float, float, float, float]]:
    count = int((sample["input"][0] != -1).sum())
    if kind == "training":
        # Official training fp_sol is (width, height, x, y).
        layout = sample["label"][1][:count]
        return [tuple(float(v) for v in row) for row in layout[:, [2, 3, 0, 1]]]
    positions = []
    for polygon in sample["label"][0][:count]:
        valid = polygon[polygon[:, 0] != -1]
        if not len(valid):
            raise ValueError("Reference polygon contains no valid vertices")
        lo = valid.min(dim=0).values
        hi = valid.max(dim=0).values
        positions.append((float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])))
    return positions


def solver_inputs(sample: dict, kind: str) -> dict:
    area, b2b, p2b, pins, constraints = sample["input"]
    count = int((area != -1).sum())
    if not bool((area[:count] > 0).all()) or bool((area[count:] != -1).any()):
        raise ValueError("Unexpected block padding or nonpositive area")
    reference = torch.tensor(reference_positions(sample, kind), dtype=torch.float64)
    constraints = constraints[:count].clone()
    allowed_positions = torch.full((count, 4), -1.0, dtype=torch.float64)
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    allowed_positions[fixed, 2:] = reference[fixed, 2:]
    allowed_positions[preplaced] = reference[preplaced]
    return {
        "block_count": count,
        "area_targets": area[:count].clone(),
        "b2b_connectivity": b2b[b2b[:, 0] >= 0].clone(),
        "p2b_connectivity": p2b[p2b[:, 0] >= 0].clone(),
        "pins_pos": pins[pins[:, 0] != -1].clone(),
        "constraints": constraints,
        "target_positions": allowed_positions,
    }


def assert_no_label_leakage(sample: dict, kind: str) -> None:
    """Changing every forbidden reference field must leave solver inputs unchanged."""
    before = solver_inputs(sample, kind)
    count = before["block_count"]
    constraints = before["constraints"]
    positions = reference_positions(sample, kind)
    changed = []
    for i, (x, y, w, h) in enumerate(positions):
        fixed, preplaced = bool(constraints[i, 0]), bool(constraints[i, 1])
        if not preplaced:
            x, y = x + 123.0, y + 321.0
        if not fixed and not preplaced:
            w, h = w * 1.25, h / 1.25
        changed.append((x, y, w, h))
    modified = dict(sample)
    if kind == "training":
        tree, layout, metrics = sample["label"]
        changed_layout = layout.clone()
        changed_layout[:count] = torch.tensor(changed, dtype=layout.dtype)[:, [2, 3, 0, 1]]
        modified["label"] = (torch.zeros_like(tree), changed_layout, torch.zeros_like(metrics))
    else:
        _, metrics = sample["label"]
        polygons = [
            torch.tensor(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]], dtype=torch.float64
            )
            for x, y, w, h in changed
        ]
        modified["label"] = (polygons, torch.zeros_like(metrics))
    after = solver_inputs(modified, kind)
    for key in before:
        if key == "block_count":
            assert before[key] == after[key]
        else:
            assert torch.equal(before[key], after[key]), f"Label leakage through {key}"
    targets = before["target_positions"]
    free = (constraints[:, 0] == 0) & (constraints[:, 1] == 0)
    assert bool((targets[free] == -1).all())
    assert bool((targets[~(constraints[:, 1] != 0), :2] == -1).all())
