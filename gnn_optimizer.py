"""Trained GNN initializer with the same baseline postprocessing and total budget."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from baseline_optimizer import MyOptimizer as _Baseline
from baseline_solver.search import BaselineConfig
from gnn_pilot.data import encode_inputs, collate_graphs
from gnn_pilot.model import FloorplanGNN, ModelConfig, move_batch, normalized_geometry
from gnn_pilot.checkpoints import load_default_checkpoint


class MyOptimizer(_Baseline):
    def __init__(self, verbose=False, config=None, checkpoint=None, device=None):
        super().__init__(verbose, config or BaselineConfig())
        if checkpoint is None:
            path, self.default_checkpoint_config = load_default_checkpoint()
        else:
            path, self.default_checkpoint_config = Path(checkpoint), None
        stored = torch.load(path, map_location="cpu", weights_only=True)
        if stored["epoch"] <= 0:
            raise RuntimeError("The selected checkpoint has not completed a training epoch")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = FloorplanGNN(ModelConfig(**stored["model_config"])).to(self.device)
        self.model.load_state_dict(stored["model_state"])
        self.model.eval()
        self.checkpoint_epoch = int(stored["epoch"])
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.checkpoint_path = str(path)
        # Model loading and one-time CUDA setup are outside per-design solve()
        # timing, just as baseline imports are outside it. Each actual prediction
        # and input encoding is included in solve() timing below.
        if self.device.type == "cuda":
            warm = {
                "x": torch.zeros(1, 8, 20, device=self.device),
                "adjacency": torch.zeros(1, 4, 8, 8, device=self.device),
                "node_mask": torch.ones(1, 8, dtype=torch.bool, device=self.device),
                "block_mask": torch.ones(1, 2, dtype=torch.bool, device=self.device),
            }
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                self.model(warm, pair_output=False)
            torch.cuda.synchronize()

    def predict(self, inputs):
        graph = encode_inputs(inputs)
        batch = move_batch(collate_graphs([graph]), self.device)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction, _ = self.model(batch, pair_output=False)
            else:
                prediction, _ = self.model(batch, pair_output=False)
            lo, wh, _ = normalized_geometry(batch, prediction)
            positions = (
                torch.cat((lo, wh), dim=-1)[0, : inputs["block_count"]].float().cpu().numpy()
            )
        positions[:, :2] = positions[:, :2] * graph["scale"] + graph["origin"].numpy()
        positions[:, 2:] *= graph["scale"]
        return positions

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
        started = time.perf_counter()
        if target_positions is None:
            target_positions = torch.full((block_count, 4), -1.0)
        inputs = dict(
            block_count=block_count,
            area_targets=area_targets,
            b2b_connectivity=b2b_connectivity,
            p2b_connectivity=p2b_connectivity,
            pins_pos=pins_pos,
            constraints=constraints,
            target_positions=target_positions,
        )
        prediction = self.predict(inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        requested = self.config
        self.config = replace(
            requested, max_seconds=max(0.0, requested.max_seconds - inference_seconds)
        )
        try:
            solution = super().solve_with_initial_guess(**inputs, initial_guess=prediction)
        finally:
            self.config = requested
        self.last_run["config"] = {**self.last_run["config"], "max_seconds": requested.max_seconds}
        self.last_run["stage_seconds"]["gnn_inference"] = inference_seconds
        self.last_run["stage_seconds"]["total"] = time.perf_counter() - started
        self.last_run["gnn"] = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "epoch": self.checkpoint_epoch,
            "device": str(self.device),
            "inference_seconds": inference_seconds,
            "postprocessing_budget_seconds": max(0.0, requested.max_seconds - inference_seconds),
        }
        self.last_run["raw_prediction_xywh"] = prediction.tolist()
        return solution
