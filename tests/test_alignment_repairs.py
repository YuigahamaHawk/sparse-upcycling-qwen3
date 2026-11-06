"""CPU regression checks for the actual SFT metadata and reference-cache failures."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.trainers import dpo as train_dpo
# Historical regressions deliberately load definitions from the frozen recipe.
# Its main() is never called; current execution does not depend on this archive.
_recipe = ROOT / "docs/archive/scripts-before-20261029/run_alignment_budget.py"
_spec = importlib.util.spec_from_file_location("historical_alignment_budget", _recipe)
run_alignment_budget = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_alignment_budget)
run_alignment_budget.ROOT = ROOT


class AlignmentRepairs(unittest.TestCase):
    def test_merge_gate_is_shift_invariant_and_rejects_different_probabilities(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.npz", root / "b.npz"
            np.savez(first, logits=np.asarray([[1., 2., 3.]]), losses=np.asarray([2.]))
            np.savez(second, logits=np.asarray([[101., 102., 103.]]), losses=np.asarray([2.]))
            with patch.object(run_alignment_budget, "RUN", root):
                run_alignment_budget.compare("shift", first, second)
                np.savez(second, logits=np.asarray([[3., 2., 1.]]), losses=np.asarray([2.]))
                with self.assertRaises(RuntimeError):
                    run_alignment_budget.compare("bad", first, second)

    def test_resume_skips_completed_groups_across_epochs(self):
        orders = [list(range(10)), list(reversed(range(10)))]
        full = [item for epoch, order in enumerate(orders)
                for item in train_dpo.resumable_groups(order, 4, epoch, 3, 0)]
        resumed = [item for epoch, order in enumerate(orders)
                   for item in train_dpo.resumable_groups(order, 4, epoch, 3, 4)]
        self.assertEqual(resumed, full[4:])
        self.assertEqual([step for step, _ in resumed], [5, 6])
        self.assertEqual(resumed[-1][1], [1, 0])

    @unittest.skipUnless(os.name == "posix", "POSIX process-group deadline guard")
    def test_deadline_guard_terminates_worker_process(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(run_alignment_budget, "RUN", directory), \
                 patch.object(run_alignment_budget, "LOGS", directory), \
                 patch.object(run_alignment_budget.time, "time", side_effect=[100, 200]):
                with self.assertRaises(TimeoutError):
                    run_alignment_budget.command("deadline-test", [sys.executable, "-c", "import time; time.sleep(600)"], "", 170)
            self.assertEqual(json.loads((directory / "pipeline_status.json").read_text())["phase"], "deadline-test")

    def test_sft_and_dpo_merge_without_cpt_steps_and_block(self):
        for stage in ("sft", "dpo"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                base, output = root / "base", root / "output"
                base.mkdir()
                weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
                a, b = torch.ones(8, 3), torch.ones(2, 8) * 0.01
                save_file({"expert.weight": weight, "gate.weight": torch.zeros(2, 3)}, str(base / "weights.safetensors"))
                (base / "config.json").write_text("{}")
                (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                    "expert.weight": "weights.safetensors", "gate.weight": "weights.safetensors"}}))
                adapter = root / "adapter.pt"
                torch.save({"state": {"expert.A": a, "expert.B": b, "gate.weight": torch.ones(2, 3)},
                            "args": {"rank": 8, "epochs": 2}, "step": 500}, adapter)
                subprocess.run([sys.executable, str(ROOT / "scripts/conversion/merge_adapters.py"), "--base", str(base),
                                "--adapter", str(adapter), "--output", str(output), "--stage", stage], check=True)
                # Detach mmap-backed tensors before Windows removes the temp dir.
                merged = {key: value.clone() for key, value in
                          load_file(str(output / "weights.safetensors")).items()}
                torch.testing.assert_close(merged["expert.weight"], weight + b @ a)
                torch.testing.assert_close(merged["gate.weight"], torch.ones(2, 3))
                manifest = json.loads((output / "merge_manifest.json").read_text())
                self.assertEqual(manifest["training_steps"], 500)
                self.assertEqual(manifest["merged_adapter_tensors"], 3)
                self.assertIsNone(manifest["trained_tokens"])
                from models.evaluation.merged_weights import MergedWeightAuditor
                from types import SimpleNamespace
                audit = MergedWeightAuditor(SimpleNamespace(
                    base=base, adapter=adapter, merged=output, output=root / 'audit.json'))
                self.assertTrue(audit.run()['passed'])
                # An independent auditor must detect a wrong production result.
                merged['expert.weight'][0, 0] += 1
                save_file(merged, str(output / 'weights.safetensors'))
                with self.assertRaises(SystemExit):
                    audit.run()
                report = json.loads((root / 'audit.json').read_text())
                self.assertFalse(report['passed'])
                self.assertEqual(report['mismatches'], ['expert.weight'])

    def test_reference_cache_atomic_write_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference_logps.npz"
            train_dpo.save_reference(path, [-1.0, -2.0], [-3.0, -4.0], "provenance")
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".npz.tmp.npz").exists())
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["chosen"], [-1, -2])
                self.assertEqual(data["fingerprint"].item(), "provenance")

    def test_partial_reference_cache_resumes_at_next_pair(self):
        class Model:
            def eval(self):
                return self
            def train(self):
                return self
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "refs.npz"
            train_dpo.save_reference(path, [-10.0], [-11.0], "same")
            pairs = [([1], [2], [3]), ([4], [5], [6])]
            with patch.object(train_dpo, "sequence_logprob", side_effect=[torch.tensor(-20.), torch.tensor(-21.)]) as score:
                chosen, rejected = train_dpo.precompute_reference(Model(), pairs, "cpu", path, "same")
            self.assertEqual(score.call_count, 2)
            np.testing.assert_array_equal(chosen, [-10., -20.])
            np.testing.assert_array_equal(rejected, [-11., -21.])


if __name__ == "__main__":
    unittest.main()
