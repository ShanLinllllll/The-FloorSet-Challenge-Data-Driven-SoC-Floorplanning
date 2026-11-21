"""Create stratified, source-file-disjoint pilot shards from official training data."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch
from floorset_io import solver_inputs, reference_positions, assert_no_label_leakage
from gnn_pilot.data import encode_inputs, add_labels, fingerprint, FEATURE_NAMES


def main():
    torch.set_num_threads(1)
    seed = 42
    rng = random.Random(seed)
    output = ROOT / "outputs/step3/data"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        print("Pilot cache exists; preserving fixed split.", flush=True)
        print((output / "manifest.json").read_text(encoding="utf-8")[:1500])
        return
    started = time.perf_counter()
    pools = collections.defaultdict(list)
    workers = sorted((ROOT / "FloorSet/floorset_lite").glob("worker_*"))
    inventory = []
    for worker_index, worker in enumerate(workers):
        files = sorted(worker.glob("layouts*"))
        for source in rng.sample(files, min(5, len(files))):
            raw = torch.load(source)
            counts = [int((value[:, 0] != -1).sum()) for value in raw[0]]
            if len(set(counts)) != 1:
                raise ValueError("Pilot stratification expects one block size per source file")
            count = counts[0]
            if not 21 <= count <= 120:
                raise ValueError(f"Unexpected size {count}")
            entry = {
                "source": str(source.relative_to(ROOT)),
                "blocks": count,
                "samples": len(counts),
            }
            inventory.append(entry)
            pools[(count - 21) // 10].append(entry)
        if (worker_index + 1) % 20 == 0:
            print(f"Inventoried {(worker_index + 1) * 5} files", flush=True)
    selected = {"train": [], "development": [], "holdout": []}
    for bucket in range(10):
        values = pools[bucket]
        rng.shuffle(values)
        if len(values) < 11:
            raise ValueError(f"Insufficient files for size bucket {bucket}: {len(values)}")
        selected["train"].extend(values[:9])
        selected["development"].append(values[9])
        selected["holdout"].append(values[10])
    seen = {}
    manifest = {
        "seed": seed,
        "source_dataset_revision": "7dc7c1bec07bcae583a274f5ed95e5be399cb8db",
        "split_policy": "90/10/10 source files; 9/1/1 files in each 10-block-size bucket; exact input duplicates across splits removed",
        "feature_names": FEATURE_NAMES,
        "splits": {},
        "inventory": inventory,
        "duplicate_inputs_removed": collections.Counter(),
        "label_leakage_checks": 0,
    }
    for split, entries in selected.items():
        manifest["splits"][split] = []
        for file_index, entry in enumerate(entries):
            source = ROOT / entry["source"]
            raw = torch.load(source)
            graphs = []
            for index in range(len(raw[0])):
                sample = {
                    "input": (
                        raw[0][index][:, 0],
                        raw[1][index],
                        raw[2][index],
                        raw[3][index],
                        raw[0][index][:, 1:],
                    ),
                    "label": (raw[4][index], raw[5][index], raw[6][index]),
                }
                if index == 0:
                    assert_no_label_leakage(sample, "training")
                    manifest["label_leakage_checks"] += 1
                graph = encode_inputs(solver_inputs(sample, "training"))
                digest = fingerprint(graph)
                if digest in seen and seen[digest] != split:
                    manifest["duplicate_inputs_removed"][split] += 1
                    continue
                seen[digest] = split
                reference = reference_positions(sample, "training")
                graph = add_labels(graph, reference, raw[6][index])
                graph["sample_id"] = f"{entry['source']}:{index}"
                graphs.append(graph)
            name = f"{split}_{file_index:03d}.pt"
            torch.save(graphs, output / name)
            manifest["splits"][split].append(
                {
                    **entry,
                    "samples": len(graphs),
                    "cache": name,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
            if (file_index + 1) % 10 == 0:
                print(f"{split}: prepared {file_index + 1}/{len(entries)} files", flush=True)
    sources = [s["source"] for entries in manifest["splits"].values() for s in entries]
    assert len(sources) == len(set(sources))
    manifest["summary"] = {
        split: {
            "samples": sum(s["samples"] for s in entries),
            "source_files": len(entries),
            "block_counts": sorted(set(s["blocks"] for s in entries)),
            "size_bucket_samples": {
                str(bucket): sum(
                    s["samples"] for s in entries if (s["blocks"] - 21) // 10 == bucket
                )
                for bucket in range(10)
            },
        }
        for split, entries in manifest["splits"].items()
    }
    manifest["prepare_seconds"] = time.perf_counter() - started
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    print(f"Data prepared in {manifest['prepare_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
