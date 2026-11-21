"""Solve a public FloorSet-Lite case and print the block rectangles as JSON."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gnn_optimizer import MyOptimizer
from scripts.floorset_io import solver_inputs
from iccad2026_evaluate import FloorplanDatasetLiteTest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=0, help="public case index (default: 0)")
    args = parser.parse_args()
    dataset = FloorplanDatasetLiteTest(str(ROOT / "FloorSet"))
    if not 0 <= args.case < len(dataset):
        parser.error(f"--case must be between 0 and {len(dataset) - 1}")

    inputs = solver_inputs(dataset[args.case], "validation")
    solver = MyOptimizer()
    rectangles = solver.solve(**inputs)
    result = {"case": args.case, "block_count": inputs["block_count"], "positions": rectangles}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
