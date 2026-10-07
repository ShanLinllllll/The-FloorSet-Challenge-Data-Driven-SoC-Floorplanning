"""Produce plots and independently recheck saved layouts with official metrics."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "FloorSet" / "iccad2026contest"), str(ROOT / "FloorSet")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
from iccad2026_evaluate import ContestEvaluator, evaluate_solution, compute_cost


def main():
    result_dir = ROOT / "results"
    report = json.loads((result_dir / "validation.json").read_text(encoding="utf-8"))
    records = report["results"]
    assert len(records) == 100, "Only deliver a complete validation report"
    evaluator = ContestEvaluator(str(ROOT / "FloorSet"), verbose=False)
    evaluator._load_dataset()
    hard_counts = {"overlap": 0, "area": 0, "dimension": 0}
    soft_counts = {"grouping": 0, "boundary": 0, "mib": 0}
    checked = []
    for record in records:
        sample = evaluator.dataset[record["test_id"]]
        areas, b2b, p2b, pins, c = sample["input"]
        baseline, truth = evaluator._extract_baseline(record["test_id"], sample["label"], b2b, p2b, pins, record["block_count"])
        m = evaluate_solution({"positions": record["positions"], "runtime": 1}, baseline, c, b2b, p2b, pins, areas, truth)
        assert m.is_feasible, (record["test_id"], m)
        rescored = compute_cost(m.hpwl_gap, m.area_gap, m.violations_relative, 1., m.is_feasible)
        assert abs(rescored - record["cost"]) < 1e-8, (record["test_id"], rescored, record["cost"])
        hard_counts["overlap"] += m.overlap_violations
        hard_counts["area"] += m.area_violations
        hard_counts["dimension"] += m.dimension_violations
        soft_counts["grouping"] += m.grouping_violations
        soft_counts["boundary"] += m.boundary_violations
        soft_counts["mib"] += m.mib_violations
        checked.append({"test_id": record["test_id"], "grouping": m.grouping_violations,
                        "boundary": m.boundary_violations, "mib": m.mib_violations})
    verification = {"verified_cases": len(records), "hard_violations": hard_counts,
                    "soft_violations": soft_counts, "cases": checked}
    (result_dir / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    n = np.array([r["block_count"] for r in records])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    earlier = result_dir / "validation_before_boundary.json"
    if earlier.exists():
        previous = json.loads(earlier.read_text(encoding="utf-8"))["results"]
        axes[0, 0].plot([r["block_count"] for r in previous], [r["cost"] for r in previous],
                        color="#b7bbc1", lw=1., label="Before boundary packing")
    axes[0, 0].plot(n, [r["cost"] for r in records], color="#2455a4", lw=1.5, label="Final solver")
    axes[0, 0].axhline(1, color="#999999", ls="--", label="Reference quality")
    axes[0, 0].set(ylabel="Local cost (lower is better)", title=f"100/100 feasible | weighted cost {report['total_score']:.4f}")
    axes[0, 0].legend()
    axes[0, 1].plot(n, [100*r["hpwl_gap"] for r in records], label="Wirelength", color="#2455a4")
    axes[0, 1].plot(n, [100*r["area_gap"] for r in records], label="Bounding-box area", color="#dc863a")
    axes[0, 1].set(ylabel="Gap against reference (%)", title="Remaining quality gap")
    axes[0, 1].legend()
    axes[1, 0].plot(n, [100*r["violations_relative"] for r in records], color="#428761")
    axes[1, 0].set(ylabel="Soft violation rate (%)", title="Grouping, boundary and MIB")
    axes[1, 1].plot(n, [r["runtime_seconds"] for r in records], color="#8755a3")
    axes[1, 1].set(ylabel="Solve time (s)", title="CPU solve time; runtime factor neutral in local score")
    for ax in axes.flat:
        ax.set_xlabel("Number of blocks")
        ax.grid(alpha=.2)
    fig.savefig(result_dir / "validation_overview.png", dpi=170)
    plt.close(fig)

    record = records[-1]
    sample = evaluator.dataset[record["test_id"]]
    areas, b2b, p2b, pins, c = sample["input"]
    rects = np.asarray(record["positions"])
    colors = {0: "#d9e0e7", 1: "#96c5e4", 2: "#aecfa8", 3: "#d5b4db"}
    fig, ax = plt.subplots(figsize=(9, 10), layout="constrained")
    for i, (x, y, w, h) in enumerate(rects):
        pre = bool(c[i, 1]); fixed = bool(c[i, 0])
        edge = "#b53a35" if pre else "#725431" if fixed else "#5a6672"
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colors.get(int(c[i, 3]), "#d9e0e7"),
                               edgecolor=edge, lw=2.3 if pre else 1.5 if fixed else .6,
                               hatch="//" if c[i, 2] else None))
        ax.text(x + w/2, y + h/2, str(i), ha="center", va="center", fontsize=6.5)
    ax.scatter(pins[:, 0], pins[:, 1], s=12, marker="x", color="#d16b27", label="Fixed terminals")
    lo, hi = rects[:, :2].min(0), (rects[:, :2] + rects[:, 2:]).max(0)
    ax.add_patch(Rectangle(lo, *(hi-lo), fill=False, ec="#222222", lw=1.3, ls="--"))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set(xlabel="x", ylabel="y", title=f"120-block validation layout | cost {record['cost']:.4f}\n"
           f"Feasible; wire gap {record['hpwl_gap']:.1%}, area gap {record['area_gap']:.1%}")
    legend = [Patch(facecolor=colors[g], label=f"Grouping {g}") for g in (1, 2, 3)]
    legend += [Patch(facecolor="white", edgecolor="#b53a35", lw=2, label="Preplaced"),
               Patch(facecolor="white", edgecolor="#725431", lw=2, label="Fixed shape"),
               Patch(facecolor="white", hatch="//", label="MIB")]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.savefig(result_dir / "layout_120.png", dpi=170)
    fig.savefig(result_dir / "layout_120.svg")
    plt.close(fig)
    print(json.dumps({k:v for k,v in verification.items() if k != "cases"}))
    print("Written validation_overview.png, layout_120.png, layout_120.svg")


if __name__ == "__main__":
    main()
