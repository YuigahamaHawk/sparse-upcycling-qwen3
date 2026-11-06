"""Held-out causal-language-model loss for dense, initial MoE, or MoE adapter."""
import json
import math
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3MoeForCausalLM
from models.components.lora import inject_expert_lora
from models.components.moe import install_qwen3_patches


class LanguageModelEvaluator:
    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        install_qwen3_patches()
        cls = Qwen3ForCausalLM if args.dense else Qwen3MoeForCausalLM
        model = cls.from_pretrained(args.model, torch_dtype=torch.bfloat16,
            device_map="auto", max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
            attn_implementation="eager").eval()
        if args.adapter:
            adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
            rank = int(adapter["args"]["rank"])
            inject_expert_lora(model, rank)
            missing, unexpected = model.load_state_dict(adapter["state"], strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
            missing_adapter = [key for key in missing if key.endswith((".A", ".B"))]
            if missing_adapter:
                raise RuntimeError(f"Missing adapter keys: {missing_adapter}")
            print(f"loaded {len(adapter['state'])} adapter tensors", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        with args.data.open(encoding="utf-8") as handle:
            rows = [json.loads(line)["text"] for line in handle if line.strip()]
        tokens = []
        for row in rows:
            tokens.extend(tokenizer.encode(row, add_special_tokens=False))
            tokens.append(tokenizer.eos_token_id)
        device = model.model.embed_tokens.weight.device
        losses = []
        probes = []
        with torch.no_grad():
            for i in range(args.blocks):
                sample = tokens[i * args.block:(i + 1) * args.block]
                if len(sample) < args.block:
                    break
                ids = torch.tensor([sample], dtype=torch.long, device=device)
                outputs = model(ids, labels=ids, use_cache=False, output_router_logits=False)
                value = float(outputs.loss)
                if not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite validation loss at block {i}")
                losses.append(value)
                if args.save_probe:
                    probes.append(outputs.logits[0, -1].float().cpu().numpy())
                print(f"block {i + 1}/{args.blocks}: {value:.4f}", flush=True)
        if not losses:
            raise ValueError("No complete validation blocks")
        if args.save_probe:
            args.save_probe.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_probe, logits=np.stack(probes), losses=np.asarray(losses))
        result = {"model": str(args.model), "adapter": str(args.adapter) if args.adapter else None,
                  "blocks": len(losses), "tokens": len(losses) * args.block,
                  "loss": sum(losses) / len(losses),
                  "perplexity": math.exp(sum(losses) / len(losses))}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
