"""Supervised fine-tuning on an upcycled Qwen3-MoE checkpoint.

Trains expert LoRA adapters and routers with the loss masked to response
tokens; base weights stay frozen. Input JSONL rows need "prompt" and
"completion" fields. The Qwen3 backbone installs the stable MoE forward
and the aux-loss patches required on Corex.
"""
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from models.components.lora import adapter_state
from models.trainers.checkpoint import atomic_save, atomic_json
from models.trainers.basic_trainer import BasicTrainer
from models.trainers.common import eval_loss as eval_wiki, schedule


def build_examples(path, tokenizer, max_len):
    examples = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        completion_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
        completion_ids = completion_ids + [tokenizer.eos_token_id]
        if not prompt_ids or len(prompt_ids) + len(completion_ids) > max_len:
            skipped += 1
            continue
        examples.append((prompt_ids, completion_ids))
    print(f"examples={len(examples)} skipped_too_long={skipped}", flush=True)
    if not examples:
        raise ValueError(f"No usable rows in {path}")
    return examples


def micro_batch(example, device):
    prompt_ids, completion_ids = example
    ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    return (torch.tensor([ids], dtype=torch.long, device=device),
            torch.tensor([labels], dtype=torch.long, device=device))


@torch.no_grad()
def eval_examples(model, examples, device):
    model.eval()
    values = []
    for example in examples:
        ids, labels = micro_batch(example, device)
        out = model(input_ids=ids, labels=labels, use_cache=False,
                    output_router_logits=False)
        values.append(float(out.loss))
    model.train()
    return sum(values) / len(values)


class SFTTrainer(BasicTrainer):
    def run(self):
        args = self.args
        if args.epochs <= 0 or args.grad_accum <= 0:
            raise ValueError("Invalid training length")
        args.output.mkdir(parents=True, exist_ok=True)
        self.seed()

        n_gpus = self.require_gpus()
        model = self.load_model(n_gpus, reject_offload=True)
        params, optimizer = self.prepare_optimizer(model)

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        examples = build_examples(args.data, tokenizer, args.max_len)
        val_examples = build_examples(args.eval_data, tokenizer, args.max_len)
        wiki_tokens = []
        with args.wiki_eval.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                wiki_tokens.extend(tokenizer.encode(json.loads(line)["text"],
                                                    add_special_tokens=False))
                wiki_tokens.append(tokenizer.eos_token_id)
                if len(wiki_tokens) >= 8 * 64:
                    break
        wiki_blocks = [torch.tensor([wiki_tokens[i * 64:(i + 1) * 64]], dtype=torch.long)
                       for i in range(8)]
        if any(b.shape[1] != 64 for b in wiki_blocks):
            raise ValueError("WikiText evaluation corpus too short")

        steps_per_epoch = math.ceil(len(examples) / args.grad_accum)
        total_steps = steps_per_epoch * args.epochs
        input_device = model.model.embed_tokens.weight.device
        run_info = {"model": str(args.model), "data": str(args.data),
                    "examples": len(examples), "val_examples": len(val_examples),
                    "epochs": args.epochs, "target_steps": total_steps,
                    "grad_accum": args.grad_accum, "max_len": args.max_len,
                    "rank": args.rank,
                    "trainable_parameters": sum(v.numel() for v in params),
                    "gpu_count": n_gpus, "device_map": model.hf_device_map}
        atomic_json(run_info, args.output / "run.json")
        print(json.dumps(run_info), flush=True)

        metrics_path = args.output / "metrics.jsonl"
        start_step = 0
        resume_path = args.output / "resume.pt"
        if args.resume:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            if checkpoint["args"]["model"] != str(args.model):
                raise ValueError("Resume checkpoint model differs from current run")
            missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint["step"]
            print(f"resumed at optimizer step {start_step}", flush=True)
            if start_step >= total_steps:
                raise ValueError("Checkpoint already reached target steps")
            if metrics_path.exists():
                records = [json.loads(line) for line in metrics_path.read_text().splitlines()
                           if line.strip()]
                if any(record["step"] > start_step for record in records):
                    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                    backup = args.output / f"metrics-before-resume-{stamp}.jsonl"
                    shutil.copy2(metrics_path, backup)
                    tmp = metrics_path.with_suffix(".jsonl.tmp")
                    tmp.write_text("".join(json.dumps(record) + "\n" for record in records
                                           if record["step"] <= start_step))
                    os.replace(tmp, metrics_path)
                    print(f"archived pre-resume metrics to {backup}", flush=True)

        run_start = time.monotonic()
        opt_step = start_step
        for epoch in range(args.epochs):
            if opt_step >= total_steps:
                break
            order = list(range(len(examples)))
            random.Random(args.seed + epoch).shuffle(order)
            for offset in range(0, len(order), args.grad_accum):
                group = order[offset:offset + args.grad_accum]
                if opt_step + 1 <= start_step:
                    continue
                optimizer.zero_grad(set_to_none=True)
                losses, auxes = [], []
                for idx in group:
                    ids, labels = micro_batch(examples[idx], input_device)
                    out = model(input_ids=ids, labels=labels, use_cache=False,
                                output_router_logits=True)
                    if not torch.isfinite(out.loss):
                        raise RuntimeError(f"Nonfinite loss at opt step {opt_step + 1}")
                    (out.loss / len(group)).backward()
                    losses.append(float(out.loss.detach()))
                    auxes.append(float(out.aux_loss.detach()))
                lr = schedule(opt_step + 1, args, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"Nonfinite gradient at opt step {opt_step + 1}")
                optimizer.step()
                opt_step += 1
                record = {"step": opt_step, "epoch": epoch + 1,
                          "tokens": opt_step * args.grad_accum,
                          "loss": sum(losses) / len(losses),
                          "aux_loss": sum(auxes) / len(auxes), "grad_norm": grad_norm,
                          "lr": lr,
                          "elapsed_seconds": round(time.monotonic() - run_start, 1)}
                if opt_step % args.eval_every == 0 or opt_step == total_steps:
                    record["val_completion_loss"] = eval_examples(model, val_examples,
                                                                  input_device)
                    record["wiki_eval_loss"] = eval_wiki(model, wiki_blocks, input_device)
                    record["wiki_eval_perplexity"] = math.exp(record["wiki_eval_loss"])
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if opt_step % args.save_every == 0 or opt_step == total_steps:
                    bundle = {"state": adapter_state(model), "args": vars(args),
                              "step": opt_step, "epoch": epoch + 1}
                    atomic_save(bundle, args.output / "adapter-latest.pt")
                    atomic_save({**bundle, "optimizer": optimizer.state_dict()},
                                resume_path)
                    atomic_json({"step": opt_step, "last_loss": record["loss"],
                                 "last_val_completion_loss": record.get("val_completion_loss"),
                                 "last_wiki_perplexity": record.get("wiki_eval_perplexity")},
                                args.output / "status.json")
                    print(f"saved resumable checkpoint at step {opt_step}", flush=True)
                if opt_step >= total_steps:
                    break
        final = {"state": adapter_state(model), "args": vars(args),
                 "step": opt_step, "trained_examples": len(examples) * args.epochs}
        atomic_save(final, args.output / "adapter.pt")
        atomic_json({"status": "complete", "steps": opt_step}, args.output / "complete.json")
        print("SFT adapter complete", flush=True)
