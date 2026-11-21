"""Resolve the default model without embedding a mutable selection in solver code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_default_checkpoint(root=ROOT):
    root = Path(root).resolve()
    spec = json.loads((root / "config/gnn_default.json").read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported default-checkpoint configuration version")
    relative = Path(spec["checkpoint"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("The default checkpoint must be relative to the project directory")
    completion = json.loads((path.parent / "result.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["checkpoint_sha256"] or digest != completion["checkpoint_sha256"]:
        raise ValueError(
            "Default checkpoint does not match its configuration and completed training result"
        )
    return path, spec
