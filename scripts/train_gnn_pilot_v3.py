"""Resumable pilot training with configurable cache and strict patience stopping."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_gnn_pilot import run_epoch
from gnn_pilot.data import ShardDataset, ShardBatchSampler, collate_graphs
from gnn_pilot.model import FloorplanGNN, ModelConfig


def capture_rng():
    value = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [value[0], value[1].tolist(), int(value[2]), int(value[3]), float(value[4])],
    }


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state(
        (value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4])
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def save_full(destination, model, optimizer, scheduler, scaler, config, progress, seconds):
    atomic_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler else None,
            "training_config": config,
            "progress": progress,
            "rng": capture_rng(),
            "execution_wall_seconds": seconds,
        },
        destination,
    )


def restore_full(payload, model, optimizer, scheduler, scaler):
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    if scaler and payload["scaler_state"] is not None:
        scaler.load_state_dict(payload["scaler_state"])
    restore_rng(payload["rng"])


def derived_files(output, config, progress):
    # last.pt is authoritative. These files can be regenerated after a crash.
    atomic_save(
        {
            "model_state": progress["best_state"],
            "model_config": config["model"],
            "epoch": progress["best_epoch"],
            "development_selection_loss": progress["best_value"],
            "training_config": config,
        },
        output / "best.pt",
    )
    (output / "history.json").write_text(
        json.dumps(progress["history"], indent=2), encoding="utf-8"
    )
    (output / "initial_development.json").write_text(
        json.dumps(progress["initial"], indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cache", default="outputs/step3/data")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, batch-size and patience must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot recipe")
    device = torch.device("cuda")
    output = ROOT / args.output
    if (output / "result.json").exists():
        raise FileExistsError("Completed training result preserved")
    if not args.resume and any(
        (output / name).exists() for name in ("best.pt", "last.pt", "config.json")
    ):
        raise FileExistsError("Existing run preserved; use --resume or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / args.cache
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = {
        split: ShardDataset(cache, manifest, split) for split in ("train", "development", "holdout")
    }
    sampler = ShardBatchSampler(datasets["train"], args.batch_size, True, args.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_sampler=sampler,
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
    sources = [
        ROOT / "gnn_pilot/data.py",
        ROOT / "gnn_pilot/model.py",
        ROOT / "scripts/train_gnn_pilot.py",
        Path(__file__),
    ]
    requested = {k: v for k, v in vars(args).items() if k != "resume"}
    config = {
        **requested,
        "generation": 3,
        "model": asdict(model_config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "amp": "float16 forward, FP32 loss and geometry",
        "threads": 1,
        "cache_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "checkpoint_note": "last.pt atomically stores optimizer, scheduler, scaler, RNG, history and best weights; best.pt is an inference checkpoint.",
    }
    print(json.dumps(config, indent=2), flush=True)
    started = time.perf_counter()
    offset = 0.0
    if args.resume:
        stored = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        assert stored["training_config"] == config, "Resume configuration or source changed"
        restore_full(stored, model, optimizer, scheduler, scaler)
        progress = stored["progress"]
        offset = stored["execution_wall_seconds"]
        print(
            f"Resumed {len(progress['history'])} completed epochs from full checkpoint", flush=True
        )
    else:
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        initial = run_epoch(model, loaders["development"], device, 0.0)
        progress = {
            "initial": initial,
            "history": [],
            "stale": 0,
            "best_epoch": 0,
            "best_value": initial["selection_loss"],
            "best_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        }
        save_full(
            output / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            progress,
            time.perf_counter() - started,
        )
        torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    derived_files(output, config, progress)
    torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    first_epoch = len(progress["history"]) + 1
    already_stopped = progress["stale"] >= args.patience
    for epoch in [] if already_stopped else range(first_epoch, args.epochs + 1):
        sampler.epoch = epoch
        weight = 0.0 if epoch <= 5 else 0.1
        train = run_epoch(model, train_loader, device, weight, optimizer, scaler, epoch)
        development = run_epoch(model, loaders["development"], device, weight)
        scheduler.step(development["selection_loss"])
        if development["selection_loss"] < progress["best_value"] - 1e-5:
            progress.update(
                best_value=development["selection_loss"],
                best_epoch=epoch,
                stale=0,
                best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        else:
            progress["stale"] += 1
        progress["history"].append(
            {
                "epoch": epoch,
                "geometry_weight": weight,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train,
                "development": development,
                "best_epoch": progress["best_epoch"],
            }
        )
        save_full(
            output / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            progress,
            offset + time.perf_counter() - started,
        )
        derived_files(output, config, progress)
        print(
            f"EPOCH {epoch:02d}: seed={args.seed}, dev={development['selection_loss']:.5f}, center={development['center_rmse_normalized']:.4f}, pair={development['pair_accuracy']:.3f}, train_s={train['seconds']:.1f}, best={progress['best_epoch']}, stale={progress['stale']}",
            flush=True,
        )
        if progress["stale"] >= args.patience:
            print(f"Early stopping after {progress['stale']} non-improving epochs", flush=True)
            break
    assert progress["best_epoch"] > 0
    model.load_state_dict(progress["best_state"])
    development = run_epoch(model, loaders["development"], device, 0.1)
    holdout = run_epoch(model, loaders["holdout"], device, 0.1)
    history = progress["history"]
    result = {
        "completed_epochs": len(history),
        "best_epoch": progress["best_epoch"],
        "stopping_reason": "patience" if progress["stale"] >= args.patience else "epoch_limit",
        "consecutive_non_improving_epochs": progress["stale"],
        "initial_development": progress["initial"],
        "selected_development": development,
        "holdout": holdout,
        "training_wall_seconds": offset + time.perf_counter() - started,
        "timing_note": "Execution wall seconds accumulate active sessions; a resumed run excludes downtime and any work after the last committed checkpoint.",
        "resumed_from_full_checkpoint": args.resume,
        "maximum_training_allocated_GiB": max(r["train"]["peak_allocated_GiB"] for r in history),
        "maximum_training_reserved_GiB": max(r["train"]["peak_reserved_GiB"] for r in history),
        "checkpoint_sha256": hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(),
        "last_checkpoint_sha256": hashlib.sha256((output / "last.pt").read_bytes()).hexdigest(),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"TRAINING COMPLETE: {json.dumps(result, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
