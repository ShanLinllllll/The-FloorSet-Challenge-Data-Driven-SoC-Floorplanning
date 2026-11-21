"""Immutable source evidence and exclusive output directories for comparisons."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_path(digest, root=ROOT):
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid source SHA-256")
    return Path(root) / "outputs/source_snapshots" / (digest + ".py")


def snapshot_source(path, root=ROOT):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    destination = snapshot_path(digest, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if sha256(destination) != digest:
            raise ValueError(f"Corrupt source snapshot: {destination}")
    return digest


def verify_recorded_sources(hashes, root=ROOT):
    """Verify the exact source bytes recorded by an existing experiment."""
    root = Path(root)
    evidence = {}
    for name, digest in hashes.items():
        path = root / name
        current = sha256(path) if path.is_file() else None
        mode = "current"
        if current != digest:
            path = snapshot_path(digest, root)
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"Recorded source unavailable or changed: {name} ({digest})")
            mode = "snapshot"
        evidence[name] = {
            "sha256": digest,
            "verified_from": mode,
            "path": str(path.relative_to(root)),
            "current_sha256": current,
        }
    archived = [name for name, item in evidence.items() if item["verified_from"] == "snapshot"]
    if archived:
        print(
            "Verified original source snapshots (current files differ): " + ", ".join(archived),
            flush=True,
        )
    return evidence


def verify_comparison_sources(comparison, root=ROOT):
    """Verify saved-layout provenance without asserting the live solver is unchanged.

    Callers separately verify the official evaluator and saved scores.
    """
    hashes = dict(comparison["source_hashes"])
    if "benchmark_script_sha256" in comparison:
        hashes["scripts/compare_gnn_pilot.py"] = comparison["benchmark_script_sha256"]
    return verify_recorded_sources(hashes, root)


def compatible_training_sources(hashes, root=ROOT):
    """Allow only formatting/comment changes when extending an old training run.

    AST comparison retains imports, docstrings, constants and every operation.
    Unknown or corrupted historical bytes, or any code change, are rejected.
    The new run records current hashes; the original checkpoint stays untouched.
    """
    root = Path(root)
    evidence = verify_recorded_sources(hashes, root)
    current_hashes = {}
    for name, item in evidence.items():
        current = root / name
        original = root / item["path"]
        if ast.dump(ast.parse(current.read_bytes())) != ast.dump(ast.parse(original.read_bytes())):
            raise ValueError(f"Training code changed beyond formatting: {name}")
        current_hashes[name] = snapshot_source(current, root)
    return current_hashes


@contextmanager
def comparison_output(requested=None, root=ROOT):
    """Reserve an empty directory before loading models or producing any plots."""
    root = Path(root)
    if requested is None:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        output = (
            root / "outputs/comparisons" / f"{stamp}_{uuid.uuid4().hex[:8]}" / "validation.json"
        )
    else:
        output = root / requested
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ".comparison.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise FileExistsError(
            f"Comparison directory is already reserved: {output.parent}"
        ) from None
    try:
        with stream:
            stream.write(json.dumps({"pid": os.getpid(), "output": output.name}))
        if any(path != lock for path in output.parent.iterdir()):
            raise FileExistsError(
                f"Comparison directory is not empty: {output.parent}. Use a new --output directory."
            )
        yield output
    finally:
        lock.unlink()


def write_new_json(path, value):
    # Exclusive creation also protects against an external writer during a run.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
