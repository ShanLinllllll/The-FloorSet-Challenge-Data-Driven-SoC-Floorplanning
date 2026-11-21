"""Independently re-read and verify the full FloorSet-Lite shard cache.

The verifier never trusts the preparation ledger.  It re-reads every shard,
re-hashes it, recomputes the ordered input fingerprints, replays a random
subset of raw source files through the encoder, and re-checks the split
isolation against development, holdout and the 100 public validation cases.
"""

from __future__ import annotations

import argparse
import collections
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
for directory in (ROOT, ROOT / "scripts", ROOT / "FloorSet", ROOT / "FloorSet/iccad2026contest"):
    sys.path.insert(0, str(directory))

import torch
from floorset_io import solver_inputs
from gnn_pilot.data import encode_inputs, fingerprint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_of(digests) -> str:
    digest = hashlib.sha256()
    for value in digests:
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def build_public_hashes():
    import iccad2026_evaluate as official

    dataset = official.FloorplanDatasetLiteTest(str(ROOT / "FloorSet"))
    hashes = {
        fingerprint(encode_inputs(solver_inputs(dataset[i], "validation")))
        for i in range(len(dataset))
    }
    return len(dataset), hashes


def verify(args):
    started = time.time()
    cache = ROOT / args.cache
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8-sig"))
    report = {
        "cache": str(cache.relative_to(ROOT)),
        "manifest_sha256": sha256_file(cache / "manifest.json"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": {},
        "failures": [],
    }
    failures = report["failures"]

    public_cases, public = build_public_hashes()
    report["checks"]["public_validation_cases"] = public_cases
    if public_cases != 100:
        failures.append(f"expected 100 public validation cases, found {public_cases}")

    protected = set(public)
    for split in ("development", "holdout"):
        for entry in manifest["splits"][split]:
            path = cache / entry["cache"]
            if sha256_file(path) != entry["cache_sha256"]:
                failures.append(f"{split} cache sha mismatch: {entry['cache']}")
                continue
            pilot = ROOT / manifest["parent_pilot_manifest"]["path"]
            pilot_dir = pilot.parent
            if sha256_file(pilot_dir / entry["cache"]) != entry["cache_sha256"]:
                failures.append(f"{split} cache differs from pilot copy: {entry['cache']}")
            graphs = torch.load(path, map_location="cpu", weights_only=True)
            digests = [fingerprint(graph) for graph in graphs]
            if digest_of(digests) != entry["input_digests_sha256"]:
                failures.append(f"{split} fingerprint digest mismatch: {entry['cache']}")
            if len(graphs) != entry["samples"] or any(
                g["n_blocks"] != entry["blocks"] for g in graphs
            ):
                failures.append(f"{split} sample/block mismatch: {entry['cache']}")
            if len(set(digests)) != len(digests):
                failures.append(f"{split} duplicate inputs inside shard: {entry['cache']}")
            protected.update(digests)
    report["checks"]["protected_inputs"] = len(protected)

    replay = {}
    for entry in manifest["splits"]["train"]:
        replay[entry["source"]] = entry["cache"]
    rng = random.Random(args.seed)
    sample_sources = sorted(rng.sample(sorted(replay), min(args.reencode_sources, len(replay))))
    report["checks"]["reencoded_source_files"] = len(sample_sources)

    total, unique, duplicates, collision_protected, collision_public = 0, set(), 0, 0, 0
    max_nodes = 0
    distribution = collections.Counter()
    checked = 0
    reencoded_ok = 0
    for entry in manifest["splits"]["train"]:
        path = cache / entry["cache"]
        if sha256_file(path) != entry["cache_sha256"]:
            failures.append(f"cache sha mismatch: {entry['cache']}")
            continue
        graphs = torch.load(path, map_location="cpu", weights_only=True)
        digests = [fingerprint(graph) for graph in graphs]
        if digest_of(digests) != entry["input_digests_sha256"]:
            failures.append(f"fingerprint digest mismatch: {entry['cache']}")
        if len(graphs) != entry["samples"]:
            failures.append(f"sample count mismatch: {entry['cache']}")
        if any(graph["n_blocks"] != entry["blocks"] for graph in graphs):
            failures.append(f"block count mismatch: {entry['cache']}")
        for graph in graphs:
            total += 1
            if len(graph["x"]) > max_nodes:
                max_nodes = len(graph["x"])
            target = graph["target"]
            if target.shape != (graph["n_blocks"], 3) or not torch.isfinite(target).all():
                failures.append(f"invalid target: {graph.get('sample_id')}")
            if not graph["sample_id"].startswith(entry["source"] + ":"):
                failures.append(f"sample_id does not match source: {graph['sample_id']}")
            if not bool((graph["area_norm"] > 0).all()):
                failures.append(f"nonpositive area: {graph['sample_id']}")
        for digest in digests:
            if digest in protected:
                collision_protected += 1
            if digest in public:
                collision_public += 1
            if digest in unique:
                duplicates += 1
            unique.add(digest)
        distribution[entry["blocks"]] += len(graphs)
        if entry["source"] in sample_sources:
            raw = torch.load(ROOT / entry["source"], map_location="cpu", weights_only=True)
            expected = {}
            for index in range(len(graphs)):
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
                expected[index] = fingerprint(encode_inputs(solver_inputs(sample, "training")))
            actual = {}
            for graph in graphs:
                index = int(graph["sample_id"].rsplit(":", 1)[1])
                actual[index] = fingerprint(graph)
            same = len(actual) == len(graphs) and all(
                expected.get(index) == digest for index, digest in actual.items()
            )
            if same:
                reencoded_ok += 1
            else:
                failures.append(f"re-encoded inputs differ: {entry['source']}")
        checked += 1
        if checked % 500 == 0:
            print(
                f"verified {checked}/{len(manifest['splits']['train'])} shards, "
                f"{total} samples, {(time.time() - started) / 60:.1f} min",
                flush=True,
            )

    report["checks"].update(
        {
            "train_shards": checked,
            "train_samples": total,
            "unique_train_inputs": len(unique),
            "duplicate_train_inputs_retained": duplicates,
            "cross_split_or_public_collisions": collision_protected,
            "public_collisions": collision_public,
            "max_nodes": max_nodes,
            "reencoded_sources_matching": reencoded_ok,
            "train_size_bucket_samples": {
                str(bucket): sum(v for k, v in distribution.items() if (k - 21) // 10 == bucket)
                for bucket in range(10)
            },
        }
    )
    if collision_protected or collision_public:
        failures.append(
            f"isolation violated: protected={collision_protected}, public={collision_public}"
        )
    if reencoded_ok != len(sample_sources):
        failures.append("re-encoding replay failed")
    expected_train = sum(entry["samples"] for entry in manifest["splits"]["train"])
    if total != expected_train or expected_train != manifest["summary"]["train"]["samples"]:
        failures.append("train sample totals disagree")
    if manifest["duplicates"].get("unique_training_inputs") != len(unique):
        failures.append("manifest unique_training_inputs disagrees with re-read")
    if manifest["duplicates"].get("retained_duplicate_training_inputs") != duplicates:
        failures.append("manifest duplicate count disagrees with re-read")
    if max_nodes != manifest.get("n_nodes_max"):
        failures.append("manifest n_nodes_max disagrees with re-read")

    report["checks"]["dataset_totals"] = {
        "raw_source_files": manifest["inventory"]["raw_source_files_total"],
        "raw_layouts": manifest["inventory"]["raw_layouts_total"],
        "train_samples": total,
        "development_samples": manifest["summary"]["development"]["samples"],
        "holdout_samples": manifest["summary"]["holdout"]["samples"],
        "expected_train_without_protected_duplicates": 1008000 - 2240,
    }
    report["passed"] = not failures
    report["seconds"] = time.time() - started
    output = ROOT / args.output if args.output else cache / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"passed": report["passed"], "failures": failures[:20]}, indent=2), flush=True)
    print(json.dumps(report["checks"], indent=2), flush=True)
    print(f"verification finished in {report['seconds'] / 60:.1f} min", flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="outputs/full_dataset/data")
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reencode-sources", type=int, default=25)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raise SystemExit(verify(args))


if __name__ == "__main__":
    main()
