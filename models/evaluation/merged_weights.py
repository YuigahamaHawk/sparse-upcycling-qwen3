"""Independent CPU tensor audit of a merged adapter checkpoint.

This deliberately computes expected tensors separately from AdapterMerger:
reusing the production merger would hide a shared arithmetic bug.
"""
import hashlib
import json
import torch
from safetensors import safe_open


class MergedWeightAuditor:
    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        torch.set_num_threads(8)
        bundle = torch.load(args.adapter, map_location="cpu", weights_only=False)
        state = bundle["state"]
        scale = 8 / bundle["args"]["rank"]
        index = json.loads((args.base / "model.safetensors.index.json").read_text())
        merged_index = json.loads((args.merged / "model.safetensors.index.json").read_text())
        if index["weight_map"] != merged_index["weight_map"]:
            raise ValueError("Base and merged indexes differ")
        checked, changed, consumed, mismatches = 0, 0, set(), []
        for filename in sorted(set(index["weight_map"].values())):
            with safe_open(str(args.base / filename), framework="pt", device="cpu") as source, \
                 safe_open(str(args.merged / filename), framework="pt", device="cpu") as target:
                if set(source.keys()) != set(target.keys()):
                    raise ValueError("Shard keys differ: " + filename)
                for key in source.keys():
                    original = source.get_tensor(key)
                    expected = original
                    if key in state:
                        expected = state[key].to(dtype=original.dtype)
                        consumed.add(key)
                        changed += 1
                    elif key.endswith(".weight"):
                        first, second = key[:-7] + ".A", key[:-7] + ".B"
                        if first in state and second in state:
                            expected = (original.float() + (state[second] @ state[first]) * scale).to(original.dtype)
                            consumed.update((first, second))
                            changed += 1
                    actual = target.get_tensor(key)
                    if not torch.equal(expected, actual):
                        mismatches.append(key)
                    checked += 1
            print(json.dumps({"shard": filename, "checked": checked, "mismatches": len(mismatches)}), flush=True)
        unused = sorted(set(state) - consumed)
        report = {"passed": not mismatches and not unused, "checked_tensors": checked,
                  "updated_weight_tensors": changed, "consumed_adapter_tensors": len(consumed),
                  "mismatches": mismatches, "unused_adapter_keys": unused,
                  "adapter_sha256": hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
                  "note": "Exact CPU merge arithmetic check. Does not imply exact BF16 inference agreement with the unmerged adapter."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        if not report["passed"]:
            raise SystemExit(1)
        return report
