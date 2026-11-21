from __future__ import annotations

from dataclasses import dataclass, asdict  # noqa: F401 - retain the historical module export
import torch
from torch import nn
import torch.nn.functional as F
from .data import FEATURE_DIM, RELATIONS


@dataclass
class ModelConfig:
    hidden: int = 128
    layers: int = 4
    dropout: float = 0.1


class GraphLayer(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.self_map = nn.Linear(hidden, hidden)
        self.messages = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(RELATIONS)
        )
        self.global_map = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adjacency, node_mask):
        mask = node_mask.unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        message = self.self_map(h) + self.global_map(pooled)[:, None]
        for relation, layer in enumerate(self.messages):
            message = message + torch.bmm(adjacency[:, relation], layer(h))
        return self.norm(h + self.dropout(F.silu(message))) * mask


class FloorplanGNN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        h = self.config.hidden
        self.embedding = nn.Sequential(nn.Linear(FEATURE_DIM, h), nn.LayerNorm(h), nn.SiLU())
        self.layers = nn.ModuleList(
            GraphLayer(h, self.config.dropout) for _ in range(self.config.layers)
        )
        self.position_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.pair_head = nn.Sequential(nn.Linear(h * 2, 64), nn.SiLU(), nn.Linear(64, 4))

    def forward(self, batch, pair_output=True):
        h = self.embedding(batch["x"]) * batch["node_mask"].unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, batch["adjacency"], batch["node_mask"])
        n = batch["block_mask"].shape[1]
        blocks = h[:, :n]
        raw = self.position_head(blocks)
        # Positive center predictions and a broad bounded shape head keep the
        # first pilot numerically stable. Source labels lie inside this range.
        prediction = torch.cat((F.softplus(raw[:, :, :2]), 1.5 * torch.tanh(raw[:, :, 2:])), dim=-1)
        pair = None
        if pair_output:
            first = blocks[:, :, None].expand(-1, -1, n, -1)
            second = blocks[:, None].expand(-1, n, -1, -1)
            pair = self.pair_head(torch.cat((first, second), dim=-1))
        return prediction, pair


def normalized_geometry(batch, prediction):
    prediction = prediction.float()
    centers = torch.where(
        batch["preplaced"].unsqueeze(-1), batch["known_centers"], prediction[:, :, :2]
    )
    ratios = prediction[:, :, 2].clamp(-3, 3).exp()
    w = (batch["area_norm"] * ratios).sqrt()
    h = (batch["area_norm"] / ratios).sqrt()
    wh = torch.stack((w, h), dim=-1)
    wh = torch.where(batch["fixed"].unsqueeze(-1), batch["locked_wh"], wh)
    return centers - wh / 2, wh, centers


def supervised_loss(batch, prediction, pair_logits, geometric_weight=0.0):
    prediction = prediction.float()
    valid = batch["block_mask"]
    moving = valid & ~batch["preplaced"]
    reshaping = valid & ~batch["fixed"]
    target = batch["target"]
    position_loss = (
        F.smooth_l1_loss(prediction[:, :, :2][moving], target[:, :, :2][moving], beta=0.05)
        if moving.any()
        else prediction.sum() * 0
    )
    ratio_loss = (
        F.smooth_l1_loss(prediction[:, :, 2][reshaping], target[:, :, 2][reshaping], beta=0.2)
        if reshaping.any()
        else prediction.sum() * 0
    )
    pair_loss = F.cross_entropy(
        pair_logits.float().reshape(-1, 4), batch["pair_target"].reshape(-1)
    )
    selection = position_loss + 0.15 * ratio_loss + 0.05 * pair_loss
    lo, wh, centers = normalized_geometry(batch, prediction)
    hi = lo + wh
    depth = (
        torch.minimum(hi[:, :, None], hi[:, None]) - torch.maximum(lo[:, :, None], lo[:, None])
    ).clamp_min(0)
    pair_valid = valid[:, :, None] & valid[:, None]
    upper = torch.triu(pair_valid, diagonal=1)
    overlaps = depth[:, :, :, 0] * depth[:, :, :, 1] * upper
    overlap_loss = (overlaps.sum(dim=(1, 2)) / valid.sum(dim=1).clamp_min(1)).mean()
    bb_lo = lo.masked_fill(~valid.unsqueeze(-1), float("inf")).amin(dim=1)
    bb_hi = hi.masked_fill(~valid.unsqueeze(-1), -float("inf")).amax(dim=1)
    area_loss = ((bb_hi - bb_lo).prod(dim=-1) - 1).clamp_min(0).mean()
    boundary_gaps = torch.stack(
        (
            (lo[:, :, 0] - bb_lo[:, None, 0]).abs(),
            (hi[:, :, 0] - bb_hi[:, None, 0]).abs(),
            (hi[:, :, 1] - bb_hi[:, None, 1]).abs(),
            (lo[:, :, 1] - bb_lo[:, None, 1]).abs(),
        ),
        dim=-1,
    )
    bits = torch.tensor([1, 2, 4, 8], device=prediction.device)
    required = ((batch["boundary"].unsqueeze(-1) & bits) != 0) & valid.unsqueeze(-1)
    boundary_loss = (boundary_gaps * required).sum() / required.sum().clamp_min(1)
    distance = (centers[:, :, None] - centers[:, None]).abs().sum(dim=-1)
    wire = (distance * batch["wire_b2b"]).sum(dim=(1, 2))
    pin_distance = (centers[:, :, None] - batch["wire_pin_xy"][:, None]).abs().sum(dim=-1)
    wire = wire + (pin_distance * batch["wire_pin_weight"]).sum(dim=(1, 2))
    wire_loss = (wire / batch["wire_baseline_norm"].clamp_min(1e-8) - 1).clamp_min(0).mean()
    geometry = overlap_loss + 0.1 * boundary_loss + 0.05 * wire_loss + 0.05 * area_loss
    total = selection + geometric_weight * geometry
    squared = ((prediction[:, :, :2] - target[:, :, :2]).square().sum(dim=-1) * moving).sum()
    ratio_error = ((prediction[:, :, 2] - target[:, :, 2]).abs() * reshaping).sum()
    pairs = batch["pair_target"] != -100
    correct = ((pair_logits.argmax(dim=-1) == batch["pair_target"]) & pairs).sum()
    return total, {
        "loss": total.detach(),
        "selection_loss": selection.detach(),
        "position_loss": position_loss.detach(),
        "ratio_loss": ratio_loss.detach(),
        "pair_loss": pair_loss.detach(),
        "overlap_loss": overlap_loss.detach(),
        "boundary_loss": boundary_loss.detach(),
        "wire_loss": wire_loss.detach(),
        "area_loss": area_loss.detach(),
        "position_squared_sum": squared.detach(),
        "position_count": moving.sum().detach(),
        "ratio_absolute_sum": ratio_error.detach(),
        "ratio_count": reshaping.sum().detach(),
        "pair_correct": correct.detach(),
        "pair_count": pairs.sum().detach(),
    }


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
