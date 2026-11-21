from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

FEATURE_DIM = 20
RELATIONS = 4
FEATURE_NAMES = [
    "block",
    "terminal",
    "log_area_to_mean",
    "area_fraction",
    "fixed",
    "preplaced",
    "left",
    "right",
    "top",
    "bottom",
    "fixed_width",
    "fixed_height",
    "known_center_x",
    "known_center_y",
    "known_position",
    "group_fraction",
    "mib_fraction",
    "block_count",
    "terminal_count",
    "terminal_aspect",
]


def _numpy(value, dtype=float):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def encode_inputs(inputs: dict) -> dict:
    """Only accepts solve()-visible inputs; no reference layout/metrics arguments."""
    n = int(inputs["block_count"])
    area = _numpy(inputs["area_targets"])[:n].copy()
    constraints = _numpy(inputs["constraints"], int)[:n]
    targets = _numpy(inputs["target_positions"])[:n]
    pins = _numpy(inputs["pins_pos"]).reshape(-1, 2)
    b2b = _numpy(inputs["b2b_connectivity"]).reshape(-1, 3)
    p2b = _numpy(inputs["p2b_connectivity"]).reshape(-1, 3)
    b2b, p2b = b2b[b2b[:, 0] >= 0], p2b[p2b[:, 0] >= 0]
    pins = pins[pins[:, 0] != -1]
    fixed = (constraints[:, 0] != 0) | (constraints[:, 1] != 0)
    preplaced = constraints[:, 1] != 0
    area[fixed] = np.prod(targets[fixed, 2:], axis=1)
    if (area <= 0).any() or not np.isfinite(area).all():
        raise ValueError("Invalid effective areas")
    scale = math.sqrt(float(area.sum()))
    known = np.vstack((pins, targets[preplaced, :2], np.zeros((1, 2))))
    origin = known.min(axis=0)
    normalized_pins = (pins - origin) / scale
    total = n + len(pins)
    x = np.zeros((total, FEATURE_DIM), np.float32)
    x[:n, 0], x[n:, 1] = 1, 1
    x[:n, 2] = np.log1p(area / (area.sum() / n))
    x[:n, 3] = area / area.sum()
    x[:n, 4], x[:n, 5] = fixed, preplaced
    for k, bit in enumerate((1, 2, 4, 8)):
        x[:n, 6 + k] = (constraints[:, 4] & bit) != 0
    locked_wh = np.zeros((n, 2), np.float32)
    locked_wh[fixed] = targets[fixed, 2:] / scale
    x[:n, 10:12] = locked_wh
    known_centers = np.zeros((n, 2), np.float32)
    known_centers[preplaced] = (
        targets[preplaced, :2] + targets[preplaced, 2:] / 2 - origin
    ) / scale
    x[:n, 12:14] = known_centers
    x[n:, 12:14] = normalized_pins
    x[:n, 14], x[n:, 14] = preplaced, 1
    for column, feature in ((3, 15), (2, 16)):
        for group in set(constraints[:, column]) - {0, -1}:
            mask = constraints[:, column] == group
            x[:n, feature][mask] = mask.sum() / n
    x[:, 17], x[:, 18] = n / 120, len(pins) / 512
    span = np.ptp(pins, axis=0) if len(pins) else np.ones(2)
    x[:, 19] = np.clip(np.log((span[0] + 1) / (span[1] + 1)), -3, 3) / 3
    edge_chunks, type_chunks, weight_chunks = [], [], []

    def add(first, second, weights, relation):
        if not len(first):
            return
        first, second, weights = (
            np.asarray(first, int),
            np.asarray(second, int),
            np.asarray(weights, np.float32),
        )
        edge_chunks.append(np.vstack((np.r_[first, second], np.r_[second, first])))
        type_chunks.append(np.full(len(first) * 2, relation, np.int8))
        weight_chunks.append(np.r_[weights, weights])

    add(b2b[:, 0], b2b[:, 1], b2b[:, 2], 0)
    add(p2b[:, 1], p2b[:, 0] + n, p2b[:, 2], 1)
    for column, relation in ((3, 2), (2, 3)):
        for group in sorted(set(constraints[:, column]) - {0, -1}):
            members = np.flatnonzero(constraints[:, column] == group)
            a, b = np.triu_indices(len(members), 1)
            add(members[a], members[b], np.ones(len(a)), relation)
    edges = (
        np.concatenate(edge_chunks, axis=1).astype(np.int32)
        if edge_chunks
        else np.zeros((2, 0), np.int32)
    )
    types = np.concatenate(type_chunks) if type_chunks else np.zeros(0, np.int8)
    weights = np.concatenate(weight_chunks) if weight_chunks else np.zeros(0, np.float32)
    return {
        "x": torch.from_numpy(x),
        "edges": torch.from_numpy(edges),
        "edge_types": torch.from_numpy(types),
        "edge_weights": torch.from_numpy(weights),
        "n_blocks": n,
        "n_pins": len(pins),
        "scale": scale,
        "origin": torch.tensor(origin, dtype=torch.float32),
        "area_norm": torch.tensor(area / scale**2, dtype=torch.float32),
        "fixed": torch.tensor(fixed),
        "preplaced": torch.tensor(preplaced),
        "locked_wh": torch.from_numpy(locked_wh),
        "known_centers": torch.from_numpy(known_centers),
        "boundary": torch.tensor(constraints[:, 4], dtype=torch.int16),
        "b2b": torch.tensor(b2b, dtype=torch.float32),
        "p2b": torch.tensor(p2b, dtype=torch.float32),
        "pins_norm": torch.tensor(normalized_pins, dtype=torch.float32),
    }


def add_labels(graph: dict, reference_xywh, metrics=None) -> dict:
    reference = torch.as_tensor(reference_xywh, dtype=torch.float32)
    n, scale, origin = graph["n_blocks"], graph["scale"], graph["origin"]
    if (
        reference.shape != (n, 4)
        or not torch.isfinite(reference).all()
        or (reference[:, 2:] <= 0).any()
    ):
        raise ValueError("Invalid reference rectangles")
    centers = (reference[:, :2] + reference[:, 2:] / 2 - origin) / scale
    ratios = torch.log(reference[:, 2] / reference[:, 3])
    graph["target"] = torch.cat((centers, ratios[:, None]), dim=1)
    lo, hi = reference[:, :2], reference[:, :2] + reference[:, 2:]
    gaps = torch.stack(
        (
            lo[None, :, 0] - hi[:, None, 0],
            lo[:, None, 0] - hi[None, :, 0],
            lo[None, :, 1] - hi[:, None, 1],
            lo[:, None, 1] - hi[None, :, 1],
        ),
        dim=-1,
    )
    denom = torch.stack(
        (reference[:, None, 2] + reference[None, :, 2],) * 2
        + (reference[:, None, 3] + reference[None, :, 3],) * 2,
        dim=-1,
    )
    values = torch.where(
        gaps >= -1e-5, gaps.clamp_min(0) / denom, torch.full_like(gaps, float("inf"))
    )
    diagonal = torch.eye(n, dtype=torch.bool)
    if not torch.isfinite(values.min(dim=-1).values[~diagonal]).all():
        raise ValueError("Reference blocks overlap")
    graph["pair_target"] = values.argmin(dim=-1).to(torch.int8)
    if metrics is not None:
        raw = torch.as_tensor(metrics)
        graph["wire_baseline_norm"] = float((raw[6] + raw[7]) / scale)
    return graph


def fingerprint(graph: dict) -> str:
    h = hashlib.sha256()
    for key in (
        "x",
        "edges",
        "edge_types",
        "edge_weights",
        "area_norm",
        "locked_wh",
        "known_centers",
    ):
        h.update(graph[key].numpy().tobytes())
    return h.hexdigest()


def collate_graphs(graphs: list[dict]) -> dict:
    batch = len(graphs)
    nodes = max(len(g["x"]) for g in graphs)
    blocks = max(g["n_blocks"] for g in graphs)
    x = torch.zeros(batch, nodes, FEATURE_DIM)
    adjacency = np.zeros((batch, RELATIONS, nodes, nodes), np.float32)
    result = {
        "x": x,
        "node_mask": torch.zeros(batch, nodes, dtype=torch.bool),
        "block_mask": torch.zeros(batch, blocks, dtype=torch.bool),
        "fixed": torch.zeros(batch, blocks, dtype=torch.bool),
        "preplaced": torch.zeros(batch, blocks, dtype=torch.bool),
        "area_norm": torch.ones(batch, blocks),
        "locked_wh": torch.zeros(batch, blocks, 2),
        "known_centers": torch.zeros(batch, blocks, 2),
        "boundary": torch.zeros(batch, blocks, dtype=torch.int64),
        "scale": torch.tensor([g["scale"] for g in graphs]),
        "origin": torch.stack([g["origin"] for g in graphs]),
        "wire_b2b": torch.zeros(batch, blocks, blocks),
        "wire_pin_weight": torch.zeros(batch, blocks, nodes),
        "wire_pin_xy": torch.zeros(batch, nodes, 2),
        "wire_baseline_norm": torch.ones(batch),
        "n_blocks": torch.tensor([g["n_blocks"] for g in graphs]),
    }
    labeled = all("target" in g for g in graphs)
    if labeled:
        result["target"] = torch.zeros(batch, blocks, 3)
        result["pair_target"] = torch.full((batch, blocks, blocks), -100, dtype=torch.long)
    for row, graph in enumerate(graphs):
        n, count = graph["n_blocks"], len(graph["x"])
        x[row, :count] = graph["x"]
        result["node_mask"][row, :count] = True
        result["block_mask"][row, :n] = True
        for key in ("fixed", "preplaced", "area_norm", "locked_wh", "known_centers", "boundary"):
            result[key][row, :n] = graph[key]
        e, t, w = graph["edges"].numpy(), graph["edge_types"].numpy(), graph["edge_weights"].numpy()
        np.add.at(adjacency[row], (t, e[1], e[0]), w)
        if len(graph["b2b"]):
            values = graph["b2b"]
            result["wire_b2b"][row].index_put_(
                (values[:, 0].long(), values[:, 1].long()), values[:, 2], accumulate=True
            )
        if len(graph["p2b"]):
            values = graph["p2b"]
            result["wire_pin_weight"][row].index_put_(
                (values[:, 1].long(), values[:, 0].long()), values[:, 2], accumulate=True
            )
        result["wire_pin_xy"][row, : graph["n_pins"]] = graph["pins_norm"]
        result["wire_baseline_norm"][row] = graph.get("wire_baseline_norm", 1.0)
        if labeled:
            result["target"][row, :n] = graph["target"]
            pair_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            result["pair_target"][row, :n, :n][pair_mask] = graph["pair_target"][pair_mask].long()
    degree = adjacency.sum(axis=-1, keepdims=True)
    adjacency /= np.maximum(degree, 1e-12)
    result["adjacency"] = torch.from_numpy(adjacency)
    return result


class ShardDataset(Dataset):
    def __init__(self, root, manifest, split, cache_files=3):
        self.root, self.cache_files = root, cache_files
        self.shards = manifest["splits"][split]
        self.offsets = np.r_[0, np.cumsum([s["samples"] for s in self.shards])]
        self.cache = OrderedDict()

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        file_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        if file_index not in self.cache:
            self.cache[file_index] = torch.load(self.root / self.shards[file_index]["cache"])
            if len(self.cache) > self.cache_files:
                self.cache.popitem(last=False)
        self.cache.move_to_end(file_index)
        return self.cache[file_index][int(index - self.offsets[file_index])]


class ShardBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=8, shuffle=True, seed=42):
        self.dataset, self.batch_size, self.shuffle, self.seed, self.epoch = (
            dataset,
            batch_size,
            shuffle,
            seed,
            0,
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shards = np.arange(len(self.dataset.shards))
        if self.shuffle:
            rng.shuffle(shards)
        for shard in shards:
            values = np.arange(self.dataset.offsets[shard], self.dataset.offsets[shard + 1])
            if self.shuffle:
                rng.shuffle(values)
            for start in range(0, len(values), self.batch_size):
                yield values[start : start + self.batch_size].tolist()

    def __len__(self):
        return sum(math.ceil(s["samples"] / self.batch_size) for s in self.dataset.shards)
