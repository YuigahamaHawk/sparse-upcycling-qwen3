"""Small JSONL CPT validation with expert LoRA and trainable routers."""
import json
import random
from pathlib import Path
import torch
from transformers import AutoTokenizer
from models.components.lora import LoRALinear, adapter_state
from models.components.moe import sparse_balance_loss, stable_sparse_moe_forward
from models.trainers.basic_trainer import BasicTrainer


def batches(path, tokenizer, block, seed):
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line)["text"] for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No text rows in {path}")
    random.Random(seed).shuffle(rows)
    tokens = []
    for row in rows:
        tokens.extend(tokenizer.encode(row, add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
    if len(tokens) < block + 1:
        raise ValueError(f"Need at least {block + 1} tokens; got {len(tokens)}")
    chunks = [tokens[i:i + block] for i in range(0, len(tokens) - block + 1, block)]
    while True:
        random.Random(seed).shuffle(chunks)
        for chunk in chunks:
            yield torch.tensor([chunk], dtype=torch.long)


class SmokeCPTTrainer(BasicTrainer):
    def run(self):
        args = self.args
        self.seed()
        devices = self.require_gpus()
        model = self.load_model(devices, reject_offload=False)
        print(f"device_map={model.hf_device_map}", flush=True)
        params, optimizer = self.prepare_optimizer(model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        stream = batches(args.data, tokenizer, args.block, args.seed)
        input_device = model.model.embed_tokens.weight.device
        args.output.mkdir(parents=True, exist_ok=True)
        log = args.output / "metrics.jsonl"
        model.train()
        print(f"trainable_parameters={sum(p.numel() for p in params):,} devices={devices}", flush=True)
        for step in range(1, args.steps + 1):
            ids = next(stream).to(input_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=ids, labels=ids, use_cache=False,
                            output_router_logits=True)
            if not torch.isfinite(outputs.loss):
                print(f"nonfinite: loss={outputs.loss} aux={outputs.aux_loss} "
                      f"logits_finite={bool(torch.isfinite(outputs.logits).all())} "
                      f"logit_min={outputs.logits.nan_to_num().min()} "
                      f"logit_max={outputs.logits.nan_to_num().max()}", flush=True)
                raise RuntimeError(f"Nonfinite loss at step {step}")
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            record = {"step": step, "loss": float(outputs.loss.detach()),
                      "aux_loss": float(outputs.aux_loss.detach()),
                      "tokens": step * args.block}
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        torch.save({"state": adapter_state(model), "args": vars(args)}, args.output / "adapter.pt")
        print(f"Saved adapter to {args.output / 'adapter.pt'}", flush=True)
