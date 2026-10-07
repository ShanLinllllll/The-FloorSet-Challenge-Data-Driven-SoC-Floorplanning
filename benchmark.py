"""Evaluate with the official engine and save reviewable metrics and solutions."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "FloorSet" / "iccad2026contest"), str(ROOT / "FloorSet")]
from iccad2026_evaluate import ContestEvaluator, compute_total_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="my_optimizer.py")
    parser.add_argument("--ids", default="all")
    parser.add_argument("--output", default="results/validation.json")
    args = parser.parse_args()
    evaluator = ContestEvaluator(str(ROOT / "FloorSet"), verbose=False)
    evaluator._load_dataset()
    optimizer = ROOT / "FloorSet" / "iccad2026contest" / args.optimizer
    optimizer_digest = hashlib.sha256(optimizer.read_bytes()).hexdigest()
    ids = range(len(evaluator.dataset)) if args.ids == "all" else [int(v) for v in args.ids.split(",")]
    records = []
    start = time.perf_counter()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    for index in ids:
        result = evaluator.evaluate(str(optimizer), [index])
        record = asdict(result.test_results[0])
        records.append(record)
        score = compute_total_score([r["cost"] for r in records], [r["block_count"] for r in records])
        report = {"optimizer": args.optimizer,
                  "upstream_commit": "aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa",
                  "optimizer_sha256": optimizer_digest,
                  "search_budget_seconds": os.environ.get("FLOORSET_SECONDS", "3"),
                  "runtime_factor": 1.0, "cases": len(records),
                  "feasible": sum(r["is_feasible"] for r in records),
                  "total_score": score, "elapsed_seconds": time.perf_counter() - start,
                  "mean_runtime_seconds": sum(r["runtime_seconds"] for r in records)/len(records),
                  "results": records}
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"id={index:02} n={record['block_count']:3} feasible={record['is_feasible']} "
              f"cost={record['cost']:.4f} wl_gap={record['hpwl_gap']:.3f} "
              f"area_gap={record['area_gap']:.3f} soft={record['violations_relative']:.3f} "
              f"time={record['runtime_seconds']:.2f}s error={record.get('error')}", flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
