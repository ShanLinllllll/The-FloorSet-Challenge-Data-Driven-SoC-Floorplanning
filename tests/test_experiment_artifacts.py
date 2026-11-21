"""Regression coverage for default selection, historic evidence and output safety."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gnn_pilot.checkpoints import load_default_checkpoint
from experiment_artifacts import (
    compatible_training_sources,
    comparison_output,
    snapshot_path,
    snapshot_source,
    verify_comparison_sources,
    write_new_json,
)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / ".cache/tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to(parent.resolve())
        self.addCleanup(self.temporary.cleanup)

    def select(self, name, data):
        path = self.root / name / "best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        (path.parent / "result.json").write_text(json.dumps({"checkpoint_sha256": digest}))
        spec = {
            "schema_version": 1,
            "checkpoint": str(path.relative_to(self.root)),
            "checkpoint_sha256": digest,
        }
        (self.root / "config").mkdir(exist_ok=True)
        (self.root / "config/gnn_default.json").write_text(json.dumps(spec))
        return path

    def test_switching_default_only_changes_selection(self):
        source = (ROOT / "gnn_optimizer.py").read_bytes()
        first = self.select("run_30", b"checkpoint A")
        self.assertEqual(load_default_checkpoint(self.root)[0], first)
        second = self.select("run_48", b"checkpoint B")
        self.assertEqual(load_default_checkpoint(self.root)[0], second)
        self.assertEqual((ROOT / "gnn_optimizer.py").read_bytes(), source)

    def test_default_rejects_checkpoint_or_completion_mismatch(self):
        path = self.select("model", b"valid checkpoint")
        path.write_bytes(b"replaced checkpoint")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_default_checkpoint(self.root)
        path = self.select("model", b"valid checkpoint")
        (path.parent / "result.json").write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_default_checkpoint(self.root)

    def test_historic_source_requires_exact_snapshot(self):
        source = self.root / "solver.py"
        source.write_text("original implementation\n")
        digest = snapshot_source(source, self.root)
        run = {"source_hashes": {"solver.py": digest}}
        self.assertEqual(
            verify_comparison_sources(run, self.root)["solver.py"]["verified_from"], "current"
        )
        source.write_text("changed implementation\n")
        proof = verify_comparison_sources(run, self.root)["solver.py"]
        self.assertEqual(proof["verified_from"], "snapshot")
        snapshot_path(digest, self.root).write_bytes(b"corrupted snapshot")
        with self.assertRaisesRegex(ValueError, "unavailable or changed"):
            verify_comparison_sources(run, self.root)

    def test_unknown_source_hash_is_not_silently_accepted(self):
        (self.root / "solver.py").write_text("changed source")
        with self.assertRaisesRegex(ValueError, "unavailable or changed"):
            verify_comparison_sources({"source_hashes": {"solver.py": "f" * 64}}, self.root)

    def test_training_continuation_accepts_only_formatting_changes(self):
        source = self.root / "train.py"
        source.write_text("def loss(x):\n return x+1\n", encoding="utf-8")
        original = snapshot_source(source, self.root)
        source.write_text("# Reformatted\ndef loss(x):\n    return x + 1\n", encoding="utf-8")
        current = compatible_training_sources({"train.py": original}, self.root)
        self.assertNotEqual(current["train.py"], original)
        self.assertEqual(
            snapshot_path(current["train.py"], self.root).read_bytes(), source.read_bytes()
        )
        source.write_text("def loss(x):\n    return x + 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "beyond formatting"):
            compatible_training_sources({"train.py": original}, self.root)
        snapshot_path(original, self.root).write_bytes(b"corrupt historical source")
        with self.assertRaisesRegex(ValueError, "unavailable or changed"):
            compatible_training_sources({"train.py": original}, self.root)

    def test_automatic_runs_get_distinct_directories(self):
        with comparison_output(root=self.root) as first:
            write_new_json(first, {"run": 1})
        with comparison_output(root=self.root) as second:
            write_new_json(second, {"run": 2})
        self.assertNotEqual(first.parent, second.parent)
        self.assertEqual(json.loads(first.read_text()), {"run": 1})

    def test_existing_result_and_plot_are_preserved(self):
        for filename in ("validation.json", "comparison_021.png"):
            folder = self.root / filename.replace(".", "_")
            folder.mkdir()
            original = folder / filename
            original.write_bytes(b"existing artifact")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                with comparison_output(folder / "validation.json", self.root):
                    self.fail("Occupied directory was accepted")
            self.assertEqual(original.read_bytes(), b"existing artifact")
            self.assertFalse((folder / ".comparison.lock").exists())

    def test_active_reservation_blocks_another_writer(self):
        target = self.root / "run/validation.json"
        with comparison_output(target, self.root):
            with self.assertRaisesRegex(FileExistsError, "already reserved"):
                with comparison_output(target, self.root):
                    self.fail("Second writer was accepted")
            self.assertTrue((target.parent / ".comparison.lock").exists())
        self.assertFalse((target.parent / ".comparison.lock").exists())

    def test_result_commit_cannot_replace_an_external_file(self):
        target = self.root / "result.json"
        target.write_text("existing result")
        with self.assertRaises(FileExistsError):
            write_new_json(target, {"new": "result"})
        self.assertEqual(target.read_text(), "existing result")


if __name__ == "__main__":
    unittest.main(verbosity=2)
