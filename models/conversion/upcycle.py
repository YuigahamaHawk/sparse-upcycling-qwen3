"""Stream a Qwen3 dense checkpoint into a Qwen3-MoE checkpoint.

Only the FFNs selected by --layers change. All other tensors retain their
original names and values. Each selected expert is an exact FFN copy.
"""
import json
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from models.backbones.qwen3 import Qwen3Backbone, Qwen3Shape


def layer_set(n, spec):
    if spec == "all":
        return set(range(n))
    if spec == "alternating":
        return set(range(1, n, 2))
    if spec == "last-half":
        return set(range(n // 2, n))
    return {int(x) for x in spec.split(",")}


def read_tensor(root, mapping, name):
    with safe_open(str(root / mapping[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


class Qwen3Upcycler:
    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        if args.output.exists() and any(args.output.iterdir()):
            raise SystemExit(f"Output directory must be empty: {args.output}")
        if not (1 <= args.top_k <= args.experts):
            raise ValueError("Require 1 <= top-k <= experts")
        source_config = json.loads((args.source / "config.json").read_text())
        shape = Qwen3Shape.from_config(source_config, require_dense=True)
        n = shape.num_hidden_layers
        sparse = layer_set(n, args.layers)
        if not sparse or min(sparse) < 0 or max(sparse) >= n:
            raise ValueError("Invalid sparse layer selection")
        index_path = args.source / "model.safetensors.index.json"
        if index_path.exists():
            mapping = json.loads(index_path.read_text())["weight_map"]
        else:
            mapping = {}
            for file in args.source.glob("*.safetensors"):
                with safe_open(str(file), framework="pt", device="cpu") as f:
                    mapping.update({key: file.name for key in f.keys()})
        cfg = Qwen3Backbone.upcycled_config(source_config, sparse=sparse,
            experts=args.experts, top_k=args.top_k, normalize_topk=args.normalize_topk)
        args.output.mkdir(parents=True)
        cfg.save_pretrained(args.output)
        for item in args.source.iterdir():
            if item.is_file() and item.name not in {"config.json", "model.safetensors.index.json"} and not item.name.endswith(".safetensors"):
                shutil.copy2(item, args.output / item.name)

        torch.manual_seed(args.seed)
        new_map = {}
        total_bytes = 0
        groups = {i: [] for i in range(n)}
        other = []
        for name in mapping:
            if name.startswith("model.layers."):
                groups[int(name.split(".")[2])].append(name)
            else:
                other.append(name)

        def write_group(names, filename, layer=None):
            nonlocal total_bytes
            result = {}
            for name in names:
                tensor = read_tensor(args.source, mapping, name)
                if layer in sparse and name.startswith(f"model.layers.{layer}.mlp."):
                    stem = f"model.layers.{layer}.mlp."
                    for expert in range(args.experts):
                        result[stem + f"experts.{expert}." + name[len(stem):]] = tensor.clone()
                else:
                    result[name] = tensor
            if layer in sparse:
                result[f"model.layers.{layer}.mlp.gate.weight"] = (
                    torch.randn(args.experts, source_config["hidden_size"]) * args.router_std
                ).to(dtype=next(iter(result.values())).dtype)
            out = args.output / filename
            save_file(result, str(out), metadata={"format": "pt"})
            for key, tensor in result.items():
                new_map[key] = filename
                total_bytes += tensor.numel() * tensor.element_size()
            print(f"wrote {filename} ({len(result)} tensors)", flush=True)

        write_group(other, "model-nonlayer.safetensors")
        for i in range(n):
            write_group(groups[i], f"model-layer-{i:02d}.safetensors", i)
        (args.output / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": new_map}, indent=2))
        (args.output / "upcycling.json").write_text(json.dumps({
            "source": str(args.source), "sparse_layers": sorted(sparse), "experts": args.experts,
            "top_k": args.top_k, "router_std": args.router_std,
            "normalize_topk": args.normalize_topk, "seed": args.seed,
            "parameters": total_bytes // 2, "weight_bytes": total_bytes,
        }, indent=2))
        print(f"complete: {total_bytes / 1e9:.2f} GB weights", flush=True)
