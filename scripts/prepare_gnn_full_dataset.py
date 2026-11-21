"""Prepare the official FloorSet-Lite training split as a resumable shard cache.

Every ``FloorSet/floorset_lite/worker_*`` source file is inventoried; the
development and holdout source files are copied from the frozen pilot split,
and every remaining source file is encoded into its own shard.  Exact input
duplicates are removed when they collide with the protected splits, the public
validation inputs, or an earlier training sample.

The cache is resumable: ``progress.jsonl`` records, per source file, the source
SHA-256, the shard SHA-256, the ordered input fingerprints and every dropped
sample.  On restart a shard is reused only if its bytes still match the ledger.

This script only reads solver-visible inputs (``solver_inputs``) plus reference
labels used for supervision; it never feeds reference geometry or metrics into
the model features.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts", ROOT / "FloorSet", ROOT / "FloorSet/iccad2026contest"):
    sys.path.insert(0, str(directory))

import torch
from floorset_io import assert_no_label_leakage, reference_positions, solver_inputs
from gnn_pilot.data import FEATURE_NAMES, add_labels, encode_inputs, fingerprint

SCHEMA_VERSION = 1
SPLITS = ("train", "development", "holdout")


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


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_write_json(path: Path, value) -> None:
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_save(payload, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_ledger(path: Path) -> dict:
    entries = {}
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"ledger: ignoring truncated line {line_number}", flush=True)
                continue
            entries[entry["source"]] = entry
    return entries


def append_ledger(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sample_of(raw, index):
    return {
        "input": (
            raw[0][index][:, 0],
            raw[1][index],
            raw[2][index],
            raw[3][index],
            raw[0][index][:, 1:],
        ),
        "label": (raw[4][index], raw[5][index], raw[6][index]),
    }


def encode_source(source: Path, protected: set[str], seen: dict[str, str]):
    """Encode one source file, dropping inputs that collide with a protected split.

    Identical inputs inside the official training split are kept (the official
    file may contain several reference layouts for the same input); they are
    counted and reported instead of silently shrinking the dataset.
    """
    raw = torch.load(source, map_location="cpu", weights_only=True)
    counts = [int((value[:, 0] != -1).sum()) for value in raw[0]]
    if len(counts) != 112:
        raise ValueError(f"{source}: expected 112 layouts, found {len(counts)}")
    if len(set(counts)) != 1:
        raise ValueError(f"{source}: mixed block counts {sorted(set(counts))}")
    blocks = counts[0]
    if not 21 <= blocks <= 120:
        raise ValueError(f"{source}: block count {blocks} outside 21--120")
    graphs, digests, dropped, duplicates, leakage_checks = [], [], [], [], 0
    label = str(source.relative_to(ROOT))
    step = max(1, len(counts) // 8)
    for index in range(len(counts)):
        sample = sample_of(raw, index)
        if index % step == 0:
            assert_no_label_leakage(sample, "training")
            leakage_checks += 1
        graph = encode_inputs(solver_inputs(sample, "training"))
        digest = fingerprint(graph)
        if digest in protected:
            dropped.append({"index": index, "reason": "protected_split_or_public_input"})
            continue
        if digest in seen:
            duplicates.append(index)
        seen[digest] = label
        graph = add_labels(graph, reference_positions(sample, "training"), raw[6][index])
        graph["sample_id"] = f"{label}:{index}"
        graphs.append(graph)
        digests.append(digest)
    if any(graph["n_blocks"] != blocks for graph in graphs):
        raise ValueError(f"{source}: inconsistent encoded block counts")
    return blocks, graphs, digests, dropped, duplicates, leakage_checks


def build_public_hashes() -> set[str]:
    import iccad2026_evaluate as official

    dataset = official.FloorplanDatasetLiteTest(str(ROOT / "FloorSet"))
    if len(dataset) != 100:
        raise ValueError(f"Expected 100 public validation cases, found {len(dataset)}")
    hashes = set()
    for index in range(len(dataset)):
        hashes.add(fingerprint(encode_inputs(solver_inputs(dataset[index], "validation"))))
    return hashes


def prepare_development_holdout(output: Path, pilot_cache: Path, pilot: dict, public: set[str]):
    entries = collections.defaultdict(list)
    seen = {}
    protected = set()
    for split in ("development", "holdout"):
        for entry in pilot["splits"][split]:
            source = ROOT / entry["source"]
            if sha256_file(source) != entry["source_sha256"]:
                raise ValueError(f"Pilot source changed: {entry['source']}")
            source_cache = pilot_cache / entry["cache"]
            destination = output / entry["cache"]
            digest = sha256_file(source_cache)
            if destination.exists() and sha256_file(destination) == digest:
                pass
            else:
                destination.write_bytes(source_cache.read_bytes())
            if sha256_file(destination) != digest:
                raise ValueError(f"Copied cache differs: {destination}")
            graphs = torch.load(destination, map_location="cpu", weights_only=True)
            if len(graphs) != entry["samples"]:
                raise ValueError(f"Pilot cache sample count changed: {entry['cache']}")
            digests = [fingerprint(graph) for graph in graphs]
            if len(set(digests)) != len(digests):
                raise ValueError(f"Duplicated inputs inside pilot cache {entry['cache']}")
            for graph_digest, graph in zip(digests, graphs):
                if graph_digest in public:
                    raise ValueError(
                        f"Pilot cache collides with public validation: {entry['cache']}"
                    )
                if graph_digest in seen:
                    raise ValueError(f"Development/holdout collision: {entry['cache']}")
                seen[graph_digest] = split
                if graph["n_blocks"] != entry["blocks"]:
                    raise ValueError(f"Pilot block count changed: {entry['cache']}")
            protected.update(digests)
            entries[split].append(
                {
                    "source": entry["source"],
                    "blocks": entry["blocks"],
                    "samples": entry["samples"],
                    "cache": entry["cache"],
                    "source_sha256": entry["source_sha256"],
                    "cache_sha256": digest,
                    "input_digests_sha256": digest_of(digests),
                    "copied_from_pilot_cache": str(source_cache.relative_to(ROOT)),
                }
            )
    return entries, seen, protected


def verify_ledger_entry(entry: dict, output: Path) -> bool:
    cache = output / entry["cache"]
    if not cache.is_file():
        return False
    if sha256_file(cache) != entry.get("cache_sha256"):
        return False
    source = ROOT / entry["source"]
    if not source.is_file() or sha256_file(source) != entry.get("source_sha256"):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/full_dataset/data")
    parser.add_argument("--pilot-cache", default="outputs/step3/data")
    parser.add_argument("--limit", type=int, default=0, help="debug only: stop after N new files")
    parser.add_argument(
        "--reuse-ledger-without-hashing",
        action="store_true",
        help="resume trust: skip byte verification (recorded in the manifest)",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    pilot_cache = ROOT / args.pilot_cache
    pilot = read_json(pilot_cache / "manifest.json")
    ledger_path = output / "progress.jsonl"
    started = time.time()

    print("Reading public validation fingerprints...", flush=True)
    public = build_public_hashes()
    print(f"public validation cases: 100, unique inputs: {len(public)}", flush=True)

    development, holdout = ("development", "holdout")
    split_entries, seen, protected = prepare_development_holdout(output, pilot_cache, pilot, public)
    protected |= public
    print(
        f"development={len(split_entries[development])} files, "
        f"holdout={len(split_entries[holdout])} files",
        flush=True,
    )

    workers = sorted(
        (ROOT / "FloorSet/floorset_lite").glob("worker_*"),
        key=lambda path: int(path.name.split("_")[1]),
    )
    if len(workers) != 100:
        raise ValueError(f"Expected 100 workers, found {len(workers)}")
    excluded = {
        entry["source"] for split in (development, holdout) for entry in split_entries[split]
    }
    sources = []
    for worker in workers:
        for path in sorted(worker.glob("layouts*.th"), key=lambda p: int(p.stem.split("_")[1])):
            relative = str(path.relative_to(ROOT))
            if relative not in excluded:
                sources.append((worker.name, path))
    print(f"training source files to inventory: {len(sources)}", flush=True)

    ledger = load_ledger(ledger_path)
    reusable = {}
    checked = 0
    for entry in ledger.values():
        if entry.get("cache") is None:
            reusable[entry["source"]] = entry
            continue
        if args.reuse_ledger_without_hashing or verify_ledger_entry(entry, output):
            reusable[entry["source"]] = entry
            checked += 1
    print(f"verified reusable shards: {checked}/{len(ledger)}", flush=True)
    for source in excluded:
        if source in ledger:
            del ledger[source]
    # Rebuild duplicate-detection state from the verified ledger (order preserved).
    seen.clear()
    for split in (development, holdout):
        for entry in split_entries[split]:
            cache = torch.load(output / entry["cache"], map_location="cpu", weights_only=True)
            for graph in cache:
                seen[fingerprint(graph)] = split
            protected.update(fingerprint(graph) for graph in cache)
    ordered_entries = sorted(reusable.values(), key=lambda entry: entry["source"])
    for entry in ordered_entries:
        if entry.get("cache") is None:
            continue
        for digest in entry.get("fingerprints", []):
            if digest in protected:
                raise ValueError(f"Ledger shard collides with protected split: {entry['cache']}")
            seen[digest] = "train"
    print(f"rebuilt {len(seen)} unique input fingerprints from ledger", flush=True)

    processed = 0
    new_files = 0
    for worker_name, source in sources:
        relative = str(source.relative_to(ROOT))
        previous = reusable.get(relative)
        if previous is not None and previous.get("cache") is not None:
            ledger[relative] = previous
            continue
        if args.limit and new_files >= args.limit:
            print(f"limit reached after {new_files} new files", flush=True)
            break
        source_digest = sha256_file(source)
        if previous is not None and previous.get("source_sha256") == source_digest:
            # All samples of this file were dropped last time: reuse the decision.
            ledger[relative] = previous
            continue
        blocks, graphs, digests, dropped, duplicates, leakage_checks = encode_source(
            source, protected, seen
        )
        entry = {
            "source": relative,
            "worker": worker_name,
            "blocks": blocks,
            "samples_total": 112,
            "samples": len(graphs),
            "cache": None,
            "source_sha256": source_digest,
            "cache_sha256": None,
            "input_digests_sha256": digest_of(digests),
            "fingerprints": digests,
            "dropped": dropped,
            "duplicate_training_inputs": duplicates,
            "label_leakage_checks": leakage_checks,
        }
        if graphs:
            cache_name = f"train_{worker_name.split('_')[1]:>03}_{source.stem}.pt"
            destination = output / cache_name
            atomic_save(graphs, destination)
            entry["cache"] = cache_name
            entry["cache_sha256"] = sha256_file(destination)
            entry["n_nodes_max"] = max(len(graph["x"]) for graph in graphs)
        else:
            entry["n_nodes_max"] = 0
        append_ledger(ledger_path, entry)
        ledger[relative] = entry
        new_files += 1
        processed += len(dropped)
        if new_files % 25 == 0:
            elapsed = time.time() - started
            print(
                f"prepared {new_files} new files, {len(ledger)} ledger entries, "
                f"dropped={processed}, elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    train_entries = [entry for entry in ledger.values() if entry.get("cache")]
    train_entries.sort(key=lambda entry: entry["source"])
    total_train = sum(entry["samples"] for entry in train_entries)
    dropped_total = sum(len(entry.get("dropped", [])) for entry in ledger.values())
    dropped_by_reason = collections.Counter(
        item["reason"] for entry in ledger.values() for item in entry.get("dropped", [])
    )
    retained_duplicates = sum(
        len(entry.get("duplicate_training_inputs", [])) for entry in ledger.values()
    )
    distribution = collections.Counter()
    for entry in train_entries:
        distribution[entry["blocks"]] += entry["samples"]
    blocks_by_split = {
        development: sorted({entry["blocks"] for entry in split_entries[development]}),
        holdout: sorted({entry["blocks"] for entry in split_entries[holdout]}),
        "train": sorted(distribution),
    }

    def samples_by_block_count(entries):
        counts = collections.Counter()
        for entry in entries:
            counts[entry["blocks"]] += entry["samples"]
        return {str(k): v for k, v in sorted(counts.items())}

    summary = {
        split: {
            "samples": sum(entry["samples"] for entry in entries),
            "source_files": len(entries),
            "block_counts": blocks_by_split[split],
            "samples_by_block_count": samples_by_block_count(entries),
        }
        for split, entries in (
            (development, split_entries[development]),
            (holdout, split_entries[holdout]),
            ("train", train_entries),
        )
    }
    summary["train"]["size_bucket_samples"] = {
        str(bucket): sum(
            value for count, value in distribution.items() if (count - 21) // 10 == bucket
        )
        for bucket in range(10)
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "FloorSet-Lite official training split (all 9,000 source files)",
        "seed": 42,
        "source_dataset_revision": pilot.get("source_dataset_revision"),
        "source_root": str(Path("FloorSet/floorset_lite")),
        "parent_pilot_manifest": {
            "path": str((pilot_cache / "manifest.json").relative_to(ROOT)),
            "sha256": sha256_file(pilot_cache / "manifest.json"),
        },
        "split_policy": (
            "Development and holdout keep the exact pilot source files and shards. Every "
            "other source file contributes its 112 layouts to train. A source file never "
            "crosses splits. Training inputs duplicated in development, holdout, public "
            "validation, or an earlier training shard are dropped."
        ),
        "feature_names": FEATURE_NAMES,
        "input_isolation": {
            "statement": (
                "Features come only from solve()-visible inputs: area targets, constraints, "
                "connectivity, pins and allowed fixed/preplaced geometry. Reference "
                "coordinates of free blocks, reference sizes of free blocks and reference "
                "metrics are never encoded; they are used only as supervision labels."
            ),
            "label_leakage_checks": sum(
                entry.get("label_leakage_checks", 0) for entry in ledger.values()
            ),
            "public_validation_cases": 100,
        },
        "splits": {
            "train": [
                {
                    key: entry[key]
                    for key in (
                        "source",
                        "blocks",
                        "samples",
                        "cache",
                        "source_sha256",
                        "cache_sha256",
                        "input_digests_sha256",
                        "n_nodes_max",
                    )
                }
                for entry in train_entries
            ],
            "development": split_entries[development],
            "holdout": split_entries[holdout],
        },
        "summary": summary,
        "duplicates": {
            "policy": (
                "Remove a training sample only when its exact encoded input also appears in "
                "development, holdout or the public validation cases. Exact duplicate inputs "
                "inside the official training split are retained and counted; they are "
                "distinct layout samples of the same input."
            ),
            "dropped_total": dropped_total,
            "dropped_protected_split_or_public_input": dropped_by_reason.get(
                "protected_split_or_public_input", 0
            ),
            "by_reason": dict(dropped_by_reason),
            "retained_duplicate_training_inputs": retained_duplicates,
            "unique_training_inputs": total_train - retained_duplicates,
            "training_source_files_without_remaining_samples": sum(
                1 for entry in ledger.values() if entry.get("cache") is None
            ),
        },
        "inventory": {
            "raw_source_files_total": 9000,
            "raw_layouts_total": 9000 * 112,
            "development_holdout_source_files": len(excluded),
            "training_source_files_used": len(train_entries),
            "training_source_files_inventoried": len(sources),
        },
        "isolation": {
            "unique_training_inputs": total_train - retained_duplicates,
            "training_samples": total_train,
            "development_holdout_inputs": 2240,
            "cross_split_input_duplicates": 0,
            "public_validation_input_duplicates": 0,
            "development_holdout_still_covers_block_counts": blocks_by_split,
        },
        "hashes": {
            "prepare_script_sha256": sha256_file(Path(__file__)),
            "gnn_pilot/data.py": sha256_file(ROOT / "gnn_pilot/data.py"),
            "gnn_pilot/model.py": sha256_file(ROOT / "gnn_pilot/model.py"),
            "scripts/floorset_io.py": sha256_file(ROOT / "scripts/floorset_io.py"),
            "scripts/train_gnn_pilot.py": sha256_file(ROOT / "scripts/train_gnn_pilot.py"),
            "scripts/train_gnn_pilot_v3.py": sha256_file(ROOT / "scripts/train_gnn_pilot_v3.py"),
        },
        "n_nodes_max": max((entry.get("n_nodes_max", 0) for entry in ledger.values()), default=0),
        "resume_verified_by_byte_hashing": not args.reuse_ledger_without_hashing,
        "prepare_seconds": time.time() - started,
    }
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(manifest["duplicates"], indent=2), flush=True)
    print(
        f"cache ready in {manifest['prepare_seconds'] / 60:.1f} min; max nodes {manifest['n_nodes_max']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
