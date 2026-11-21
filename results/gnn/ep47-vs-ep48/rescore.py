"""Re-score the saved final layouts and summarize the paired run."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
import compare_gnn_pilot as bench
from experiment_artifacts import sha256, write_new_json

folder = Path(sys.argv[1])
config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (folder / "layouts.jsonl").read_text(encoding="utf-8").splitlines()]
assert [r["test_id"] for r in records] == list(range(100))
assert sha256(root / "FloorSet/iccad2026contest/iccad2026_evaluate.py") == config["source_sha256"]["FloorSet/iccad2026contest/iccad2026_evaluate.py"]
bench.torch.set_num_threads(1)
dataset = bench.official.FloorplanDatasetLiteTest(str(root / "FloorSet"))
evaluator = bench.official.ContestEvaluator(str(root / "FloorSet"), verbose=False)
score = bench.official.compute_total_score
names = ("full47", "subset48")
rows = []
for record in records:
    i = record["test_id"]
    assert record["order"] == list(names if i % 2 == 0 else reversed(names))
    sample = dataset[i]
    inputs = bench.solver_inputs(sample, "validation")
    reference = bench.reference_positions(sample, "validation")
    baseline, _ = evaluator._extract_baseline(i, sample["label"], inputs["b2b_connectivity"],
                                              inputs["p2b_connectivity"], inputs["pins_pos"], inputs["block_count"])
    assert record["block_count"] == inputs["block_count"]
    row = dict(test_id=i, blocks=record["block_count"], first=record["order"][0])
    for name in names:
        m = record["methods"][name]
        assert m["gnn"]["checkpoint_sha256"] == config["checkpoints"][name]["sha256"]
        metrics = bench.evaluate_layout(inputs, m["positions"], baseline, reference)
        assert metrics == m["metrics"], (i, name)
        row.update({name+"_cost": metrics["cost"], name+"_feasible": metrics["is_feasible"],
                    name+"_runtime_s": m["runtime_seconds"], name+"_inference_ms": m["gnn"]["inference_seconds"]*1000})
    row["cost_drop_vs_subset48"] = row["subset48_cost"]-row["full47_cost"]
    rows.append(row)

blocks = [r["blocks"] for r in rows]
summary = dict(cases=len(rows), rechecked_layouts=2*len(rows), rescore_mismatches=0)
for name in names:
    runtimes = [r[name+"_runtime_s"] for r in rows]
    summary[name] = dict(
        weighted_cost=score([r[name+"_cost"] for r in rows],blocks),
        feasible=sum(r[name+"_feasible"] for r in rows),
        mean_runtime_s=float(np.mean(runtimes)), p95_runtime_s=float(np.percentile(runtimes,95)),
        max_runtime_s=max(runtimes), over_2s=sum(x>2.0 for x in runtimes),
        mean_inference_ms=float(np.mean([r[name+"_inference_ms"] for r in rows])),
        stage_costs={stage:score([r["methods"][name]["stage_costs"][stage] for r in records],blocks)
                     for stage in ("legalization","lp","sa")})
drops = [row["cost_drop_vs_subset48"] for row in rows]
summary["wins_full47"] = sum(d>1e-9 for d in drops)
summary["wins_subset48"] = sum(d< -1e-9 for d in drops)
summary["ties"] = len(rows)-summary["wins_full47"]-summary["wins_subset48"]
before = summary["subset48"]["weighted_cost"]
after = summary["full47"]["weighted_cost"]
summary["weighted_cost_drop"] = before-after
summary["improvement_percent"] = 100*(before-after)/before
summary["by_block_count"] = []
for lo in range(21,121,20):
    selected = [r for r in rows if lo <= r["blocks"] < lo+20]
    sizes = [r["blocks"] for r in selected]
    a,b = (score([r[name+"_cost"] for r in selected],sizes) for name in names)
    summary["by_block_count"].append(dict(blocks=[lo,lo+19],full47_wins=sum(r["cost_drop_vs_subset48"]>1e-9 for r in selected),
                                           cases=len(selected),full47_cost=a,subset48_cost=b,improvement_percent=100*(b-a)/b))
with (folder / "cases.csv").open("x",newline="",encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f,fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
write_new_json(folder/"summary.json",summary)
print(json.dumps(summary,indent=2))
