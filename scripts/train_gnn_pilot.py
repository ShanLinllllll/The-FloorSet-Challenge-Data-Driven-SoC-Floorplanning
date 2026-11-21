"""Train the small GNN pilot and select a checkpoint on internal development data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from gnn_pilot.data import ShardDataset, ShardBatchSampler, collate_graphs
from gnn_pilot.model import FloorplanGNN, ModelConfig, move_batch, supervised_loss


class Meter:
    def __init__(self):
        self.samples = 0
        self.values = {}
        self.sums = {
            key: 0.0
            for key in (
                "position_squared_sum",
                "position_count",
                "ratio_absolute_sum",
                "ratio_count",
                "pair_correct",
                "pair_count",
            )
        }

    def add(self, components, count):
        self.samples += count
        for key, tensor in components.items():
            value = float(tensor)
            if key in self.sums:
                self.sums[key] += value
            else:
                self.values[key] = self.values.get(key, 0.0) + value * count

    def result(self):
        output = {key: value / max(self.samples, 1) for key, value in self.values.items()}
        output.update(
            center_rmse_normalized=math.sqrt(
                self.sums["position_squared_sum"] / max(self.sums["position_count"], 1)
            ),
            log_ratio_mae=self.sums["ratio_absolute_sum"] / max(self.sums["ratio_count"], 1),
            pair_accuracy=self.sums["pair_correct"] / max(self.sums["pair_count"], 1),
            samples=self.samples,
        )
        return output


def run_epoch(model, loader, device, geometry_weight, optimizer=None, scaler=None, epoch=0):
    training = optimizer is not None
    model.train(training)
    meter = Meter()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for step, raw in enumerate(loader):
        batch = move_batch(raw, device)
        count = len(batch["x"])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast("cuda", dtype=torch.float16):
                prediction, pair = model(batch)
            # Geometry, reductions and loss evaluation use FP32.
            loss, components = supervised_loss(batch, prediction, pair, geometry_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, step {step}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        meter.add(components, count)
        if training and (step + 1) % 250 == 0:
            print(
                f"epoch={epoch:02d}, step={step + 1}/{len(loader)}, elapsed={time.perf_counter() - started:.1f}s, loss={float(loss):.5f}",
                flush=True,
            )
    torch.cuda.synchronize()
    result = meter.result()
    result.update(
        seconds=time.perf_counter() - started,
        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default="outputs/step3/training")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot expects the verified CUDA environment")
    device = torch.device("cuda")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError(
            "Existing pilot checkpoint preserved; use a new output directory for another run"
        )
    cache = ROOT / "outputs/step3/data"
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = {
        split: ShardDataset(cache, manifest, split) for split in ("train", "development", "holdout")
    }
    train_sampler = ShardBatchSampler(datasets["train"], args.batch_size, True, args.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_sampler=train_sampler,
        collate_fn=collate_graphs,
        num_workers=0,
        pin_memory=True,
    )
    loaders = {
        split: DataLoader(
            datasets[split],
            batch_sampler=ShardBatchSampler(datasets[split], args.batch_size, False),
            collate_fn=collate_graphs,
            num_workers=0,
            pin_memory=True,
        )
        for split in ("development", "holdout")
    }
    model_config = ModelConfig()
    model = FloorplanGNN(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, min_lr=1e-5
    )
    sources = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            ROOT / "gnn_pilot/data.py",
            ROOT / "gnn_pilot/model.py",
            Path(__file__).resolve(),
        ]
    }
    config = {
        **vars(args),
        "model": asdict(model_config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "amp": "float16 forward, FP32 loss and geometry",
        "cache_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_hashes": sources,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    started = time.perf_counter()
    initial = run_epoch(model, loaders["development"], device, 0.0)
    best = initial["selection_loss"]
    best_epoch = 0

    def checkpoint(epoch, metric):
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": asdict(model_config),
                "epoch": epoch,
                "development_selection_loss": metric,
                "training_config": config,
            },
            output / "best.pt",
        )

    checkpoint(0, best)
    torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    print(f"untrained_development={json.dumps(initial)}", flush=True)
    history, stale = [], 0
    (output / "initial_development.json").write_text(
        json.dumps(initial, indent=2), encoding="utf-8"
    )
    for epoch in range(1, args.epochs + 1):
        train_sampler.epoch = epoch
        weight = 0.0 if epoch <= 5 else 0.1
        train = run_epoch(model, train_loader, device, weight, optimizer, scaler, epoch)
        development = run_epoch(model, loaders["development"], device, weight)
        scheduler.step(development["selection_loss"])
        improved = development["selection_loss"] < best - 1e-5
        if improved:
            best, best_epoch, stale = development["selection_loss"], epoch, 0
            checkpoint(epoch, best)
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "geometry_weight": weight,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train,
            "development": development,
            "best_epoch": best_epoch,
        }
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(
            f"EPOCH {epoch:02d}: train={train['selection_loss']:.5f}, dev={development['selection_loss']:.5f}, center_rmse={development['center_rmse_normalized']:.4f}, ratio_mae={development['log_ratio_mae']:.4f}, pair_acc={development['pair_accuracy']:.3f}, train_s={train['seconds']:.1f}, peak_GiB={train['peak_allocated_GiB']:.2f}, best_epoch={best_epoch}",
            flush=True,
        )
        if stale >= args.patience and epoch >= 10:
            print(f"Early stopping after {stale} non-improving epochs", flush=True)
            break
    stored = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(stored["model_state"])
    development = run_epoch(model, loaders["development"], device, 0.1)
    # Holdout is evaluated once after model selection.
    holdout = run_epoch(model, loaders["holdout"], device, 0.1)
    result = {
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "initial_development": initial,
        "selected_development": development,
        "holdout": holdout,
        "training_wall_seconds": time.perf_counter() - started,
        "maximum_training_allocated_GiB": max(r["train"]["peak_allocated_GiB"] for r in history),
        "maximum_training_reserved_GiB": max(r["train"]["peak_reserved_GiB"] for r in history),
        "checkpoint_sha256": hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"TRAINING COMPLETE: {json.dumps(result, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
