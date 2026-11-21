"""Frozen-checkpoint paired comparison against the baseline, including inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "FloorSet"))
sys.path.insert(0, str(ROOT / "FloorSet/iccad2026contest"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import iccad2026_evaluate as official
from baseline_optimizer import MyOptimizer as Baseline
from baseline_solver.core import Problem, feasible, metrics
from baseline_solver.search import BaselineConfig
from gnn_optimizer import MyOptimizer as GNN
from floorset_io import solver_inputs, reference_positions
from experiment_artifacts import comparison_output, snapshot_source, write_new_json


def evaluate_layout(inputs, positions, baseline, reference):
    return asdict(
        official.evaluate_solution(
            {"positions": positions, "runtime": 1.0},
            baseline,
            inputs["constraints"],
            inputs["b2b_connectivity"],
            inputs["p2b_connectivity"],
            inputs["pins_pos"],
            inputs["area_targets"],
            reference,
            median_runtime=1.0,
        )
    )


def plot_case(path, record, inputs, reference):
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    colors = plt.get_cmap("tab20")
    choices = [("Reference", reference, None)] + [
        (name, record["methods"][name]["positions"], record["methods"][name])
        for name in ("baseline", "gnn")
    ]
    for ax, (name, positions, result) in zip(axes, choices):
        for i, (x, y, w, h) in enumerate(positions):
            group = int(inputs["constraints"][i, 3])
            locked = bool(inputs["constraints"][i, 0] or inputs["constraints"][i, 1])
            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    h,
                    facecolor=colors((group if group else i) % 20),
                    edgecolor="#222",
                    linewidth=0.5,
                    alpha=0.7,
                    hatch="//" if locked else None,
                )
            )
            if inputs["block_count"] <= 60:
                ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=5)
        pins = inputs["pins_pos"].numpy()
        if len(pins):
            ax.scatter(pins[:, 0], pins[:, 1], marker="x", c="#c82127", s=9)
        ax.autoscale()
        ax.margins(0.06)
        ax.set_aspect("equal")
        title = (
            name
            if result is None
            else f"{name}: cost={result['metrics']['cost']:.3f}\n{result['runtime_seconds']:.3f}s incl. initialization"
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(
        f"Case {record['test_id']}: {record['block_count']} blocks | identical total time budget",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="all")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        help="New result path in an empty directory; default creates a unique outputs/comparisons run",
    )
    parser.add_argument(
        "--checkpoint", help="Completed checkpoint; default reads config/gnn_default.json"
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    with comparison_output(args.output, ROOT) as output:
        print(f"Comparison output: {output}", flush=True)
        run_comparison(args, output)


def run_comparison(args, output):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = BaselineConfig(max_seconds=args.seconds, seed=args.seed)
    checkpoint = ROOT / args.checkpoint if args.checkpoint else None
    if checkpoint is not None:
        completion = json.loads((checkpoint.parent / "result.json").read_text(encoding="utf-8"))
        assert (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest() == completion["checkpoint_sha256"]
        )
    sources = [
        ROOT / name
        for name in (
            "baseline_optimizer.py",
            "baseline_solver/core.py",
            "baseline_solver/search.py",
            "gnn_optimizer.py",
            "gnn_pilot/data.py",
            "gnn_pilot/model.py",
            "gnn_pilot/checkpoints.py",
            "scripts/floorset_io.py",
            "scripts/experiment_artifacts.py",
        )
    ]
    source_hashes = {str(p.relative_to(ROOT)): snapshot_source(p) for p in sources}
    benchmark_digest = snapshot_source(Path(__file__))
    constructor_start = time.perf_counter()
    solvers = {
        "baseline": Baseline(config=config),
        "gnn": GNN(config=config, checkpoint=checkpoint),
    }
    constructor_seconds = time.perf_counter() - constructor_start
    indices = list(range(100)) if args.cases == "all" else [int(x) for x in args.cases.split(",")]
    dataset = official.FloorplanDatasetLiteTest(str(ROOT / "FloorSet"))
    evaluator = official.ContestEvaluator(str(ROOT / "FloorSet"), verbose=False)
    started = time.perf_counter()
    records = []
    for index in indices:
        sample = dataset[index]
        inputs = solver_inputs(sample, "validation")
        reference = reference_positions(sample, "validation")
        baseline, _ = evaluator._extract_baseline(
            index,
            sample["label"],
            inputs["b2b_connectivity"],
            inputs["p2b_connectivity"],
            inputs["pins_pos"],
            inputs["block_count"],
        )
        problem = Problem.from_inputs(**inputs)
        methods = {}
        order = ("baseline", "gnn") if index % 2 == 0 else ("gnn", "baseline")
        for name in order:
            solver = solvers[name]
            begin = time.perf_counter()
            try:
                positions = solver.solve(**inputs)
                runtime = time.perf_counter() - begin
                assert feasible(problem, np.asarray(positions))
                stages = {}
                for stage, layout in solver.last_run["layouts"].items():
                    exact = evaluate_layout(inputs, layout.tolist(), baseline, reference)
                    assert exact["is_feasible"]
                    internal = metrics(problem, layout)
                    assert np.isclose(internal["hpwl"], exact["hpwl_total"], rtol=1e-10, atol=1e-10)
                    assert internal["soft_counts"] == {
                        k: exact[k + "_violations"] for k in ("boundary", "grouping", "mib")
                    }
                    stages[stage] = exact
                methods[name] = {
                    "positions": positions,
                    "runtime_seconds": runtime,
                    "metrics": stages["sa"],
                    "stages": stages,
                    "telemetry": {k: v for k, v in solver.last_run.items() if k != "layouts"},
                    "error": None,
                }
            except Exception as exc:
                methods[name] = {
                    "runtime_seconds": time.perf_counter() - begin,
                    "metrics": {"is_feasible": False, "cost": 10.0},
                    "error": f"{type(exc).__name__}: {exc}",
                }
        record = {
            "test_id": index,
            "block_count": inputs["block_count"],
            "execution_order": order,
            "methods": methods,
        }
        records.append(record)
        if index in (0, 39, 99) and all(m["error"] is None for m in methods.values()):
            plot_case(
                output.parent / f"comparison_{inputs['block_count']:03d}.png",
                record,
                inputs,
                reference,
            )
        print(
            f"case={index:02d} blocks={inputs['block_count']:3d} baseline={methods['baseline']['metrics']['cost']:.4f} gnn={methods['gnn']['metrics']['cost']:.4f} errors={[m['error'] for m in methods.values() if m['error']]}",
            flush=True,
        )
    summaries = {}
    for name in ("baseline", "gnn"):
        values = [r["methods"][name] for r in records]
        good = [r for r in values if r["error"] is None]
        summaries[name] = {
            "num_cases": len(values),
            "num_feasible": sum(r["metrics"]["is_feasible"] for r in values),
            "num_errors": len(values) - len(good),
            "weighted_cost_runtime_factor_1": official.compute_total_score(
                [r["metrics"]["cost"] for r in values], [r["block_count"] for r in records]
            ),
            "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in values])),
            "p95_runtime_seconds": float(np.percentile([r["runtime_seconds"] for r in values], 95)),
            "total_soft_counts": {
                k: sum(r["metrics"][k + "_violations"] for r in good)
                for k in ("boundary", "grouping", "mib")
            },
            "stage_weighted_costs": {
                stage: official.compute_total_score(
                    [
                        r["methods"][name]["stages"][stage]["cost"]
                        for r in records
                        if r["methods"][name]["error"] is None
                    ],
                    [r["block_count"] for r in records if r["methods"][name]["error"] is None],
                )
                for stage in ("legalization", "lp", "sa")
            },
        }
    delta = (
        summaries["baseline"]["weighted_cost_runtime_factor_1"]
        - summaries["gnn"]["weighted_cost_runtime_factor_1"]
    )
    summaries["paired"] = {
        "gnn_lower_cost_cases": sum(
            r["methods"]["gnn"]["metrics"]["cost"]
            < r["methods"]["baseline"]["metrics"]["cost"] - 1e-9
            for r in records
        ),
        "gnn_higher_cost_cases": sum(
            r["methods"]["gnn"]["metrics"]["cost"]
            > r["methods"]["baseline"]["metrics"]["cost"] + 1e-9
            for r in records
        ),
        "weighted_cost_improvement_percent": 100
        * delta
        / summaries["baseline"]["weighted_cost_runtime_factor_1"],
        "mean_gnn_inference_seconds": float(
            np.mean(
                [
                    r["methods"]["gnn"]["telemetry"]["gnn"]["inference_seconds"]
                    for r in records
                    if r["methods"]["gnn"]["error"] is None
                ]
            )
        ),
    }
    for path, digest in source_hashes.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during comparison: {path}")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != benchmark_digest:
        raise RuntimeError("Benchmark script changed during comparison")
    result = {
        "config": asdict(config),
        "selected_epoch": solvers["gnn"].checkpoint_epoch,
        "checkpoint_sha256": solvers["gnn"].checkpoint_sha256,
        "threads": 1,
        "checkpoint_path": str(Path(solvers["gnn"].checkpoint_path).relative_to(ROOT)),
        "benchmark_script_sha256": benchmark_digest,
        "default_checkpoint_config": solvers["gnn"].default_checkpoint_config,
        "source_snapshot_directory": "outputs/source_snapshots",
        "timing_notes": "Both methods receive the same total budget. GNN solve() includes encoding, inference, transfer, synchronization and postprocessing. Constructor and dataset/scoring/plotting excluded.",
        "constructor_seconds": constructor_seconds,
        "comparison_wall_seconds": time.perf_counter() - started,
        "source_hashes": source_hashes,
        "summary": summaries,
        "cases": records,
    }
    write_new_json(output, result)
    print(json.dumps(summaries, indent=2), flush=True)
    if any(summaries[name]["num_errors"] for name in ("baseline", "gnn")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
