"""Merge a trained LoRA/router adapter into a standalone Qwen3-MoE checkpoint."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


class AdapterMerger:
    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        if args.output.exists() and any(args.output.iterdir()):
            raise SystemExit(f"Output must be empty: {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)
        index = json.loads((args.base / "model.safetensors.index.json").read_text())
        adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
        state = adapter["state"]
        rank = adapter["args"]["rank"]
        scale = 8 / rank  # LoRALinear default alpha=8
        merged = set()
        for item in args.base.iterdir():
            if item.is_file() and item.name not in {"model.safetensors.index.json", "merge_manifest.json"} and not item.name.endswith(".safetensors"):
                shutil.copy2(item, args.output / item.name)
        cfg_path = args.output / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["output_router_logits"] = False  # HF 4.53 auxiliary loss cannot handle dense layers
        cfg_path.write_text(json.dumps(cfg, indent=2))

        for filename in sorted(set(index["weight_map"].values())):
            result = {}
            with safe_open(str(args.base / filename), framework="pt", device="cpu") as source:
                for key in source.keys():
                    weight = source.get_tensor(key)
                    if key in state:
                        weight = state[key].to(dtype=weight.dtype)
                        merged.add(key)
                    elif key.endswith(".weight"):
                        stem = key[:-7]
                        a_key, b_key = stem + ".A", stem + ".B"
                        if a_key in state and b_key in state:
                            delta = torch.matmul(state[b_key], state[a_key]) * scale
                            weight = (weight.float() + delta).to(weight.dtype)
                            merged.update((a_key, b_key))
                    result[key] = weight.contiguous()
            save_file(result, str(args.output / filename), metadata={"format": "pt"})
            print(f"merged {filename}", flush=True)

        unused = set(state) - merged
        if unused:
            raise RuntimeError(f"Adapter tensors not merged: {sorted(unused)[:10]}")
        shutil.copy2(args.base / "model.safetensors.index.json",
                     args.output / "model.safetensors.index.json")
        training_args = adapter["args"]
        tokens = adapter.get("trained_tokens")
        if tokens is None and args.stage == "cpt" and "steps" in training_args and "block" in training_args:
            tokens = training_args["steps"] * training_args["block"]
        manifest = {
            "status": "complete", "stage": args.stage,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base": str(args.base), "adapter": str(args.adapter),
            "adapter_sha256": hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
            "training_steps": adapter.get("step", training_args.get("steps")),
            "trained_tokens": tokens,
            "trained_pairs": adapter.get("trained_pairs"),
            "adapter_rank": rank, "merged_adapter_tensors": len(merged),
            "lora_alpha": 8, "shards": len(set(index["weight_map"].values())),
        }
        for name in (args.stage + "_manifest.json", "merge_manifest.json"):
            path = args.output / name
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(path)
        print(f"Standalone {args.stage.upper()} checkpoint: {args.output}", flush=True)
