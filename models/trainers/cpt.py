"""Resumable, monitored adapter/router CPT on a pretokenized corpus."""
import json
import math
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from models.components.lora import adapter_state
from models.trainers.checkpoint import atomic_save, atomic_json
from models.trainers.basic_trainer import BasicTrainer
from models.trainers.common import eval_loss


class CPTTrainer(BasicTrainer):
    def run(self):
        args = self.args
        if args.steps <= 0 or args.block < 2 or args.grad_accum <= 0:
            raise ValueError("Invalid training length")
        args.output.mkdir(parents=True, exist_ok=True)
        tokens = np.memmap(args.tokens, mode="r", dtype="<u4")
        if len(tokens) < args.block * 10:
            raise ValueError("Training token stream is too short")
        self.seed()
        n_gpus = self.require_gpus()
        model = self.load_model(n_gpus, reject_offload=True)
        params, optimizer = self.prepare_optimizer(model)
        start_step = 0
        resume_path = args.output / "resume.pt"
        if args.resume:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            if checkpoint["args"]["model"] != args.model or checkpoint["args"]["tokens"] != args.tokens:
                raise ValueError("Resume checkpoint model/data differs from current run")
            missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint["step"]
            print(f"resumed at step {start_step}", flush=True)
            if start_step >= args.steps:
                raise ValueError("Checkpoint already reached target steps")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        with args.eval_data.open(encoding="utf-8") as handle:
            texts = [json.loads(line)["text"] for line in handle if line.strip()]
        eval_tokens = []
        for text in texts:
            eval_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
            eval_tokens.append(tokenizer.eos_token_id)
            if len(eval_tokens) >= 8 * 64:
                break
        eval_blocks = [torch.tensor([eval_tokens[i * 64:(i + 1) * 64]], dtype=torch.long)
                       for i in range(8)]
        if any(b.shape[1] != 64 for b in eval_blocks):
            raise ValueError("Evaluation corpus too short")
        input_device = model.model.embed_tokens.weight.device
        run_info = {"model": str(args.model), "tokens_file": str(args.tokens),
            "corpus_tokens": len(tokens), "target_steps": args.steps,
            "tokens_per_step": args.block * args.grad_accum,
            "trainable_parameters": sum(v.numel() for v in params), "gpu_count": n_gpus,
            "device_map": model.hf_device_map}
        atomic_json(run_info, args.output / "run.json")
        print(json.dumps(run_info), flush=True)
        metrics_path = args.output / "metrics.jsonl"
        if args.resume and metrics_path.exists():
            records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            if any(record["step"] > start_step for record in records):
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                backup = args.output / f"metrics-before-resume-{stamp}.jsonl"
                shutil.copy2(metrics_path, backup)
                tmp = metrics_path.with_suffix(".jsonl.tmp")
                tmp.write_text("".join(json.dumps(record) + "\n" for record in records
                                       if record["step"] <= start_step))
                os.replace(tmp, metrics_path)
                print(f"archived pre-resume metrics to {backup}", flush=True)
        model.train()
        run_start = time.monotonic()
        for step in range(start_step + 1, args.steps + 1):
            if step <= args.warmup:
                lr = args.lr * step / max(1, args.warmup)
            else:
                progress = (step - args.warmup) / max(1, args.steps - args.warmup)
                lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            losses, auxes = [], []
            for micro in range(args.grad_accum):
                sample_id = (step - 1) * args.grad_accum + micro
                rng = np.random.default_rng(args.seed + sample_id)
                offset = int(rng.integers(0, len(tokens) - args.block))
                chunk = np.asarray(tokens[offset:offset + args.block], dtype=np.int64)
                ids = torch.from_numpy(chunk.copy()).unsqueeze(0).to(input_device)
                out = model(input_ids=ids, labels=ids, use_cache=False,
                            output_router_logits=True)
                if not torch.isfinite(out.loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}, micro {micro}")
                (out.loss / args.grad_accum).backward()
                losses.append(float(out.loss.detach()))
                auxes.append(float(out.aux_loss.detach()))
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Nonfinite gradient at step {step}")
            optimizer.step()
            record = {"step": step, "tokens": step * args.block * args.grad_accum,
                "loss": sum(losses) / len(losses), "aux_loss": sum(auxes) / len(auxes),
                "grad_norm": grad_norm, "lr": lr,
                "elapsed_seconds": round(time.monotonic() - run_start, 1)}
            if step % args.eval_every == 0 or step == args.steps:
                record["eval_loss"] = eval_loss(model, eval_blocks, input_device)
                record["eval_perplexity"] = math.exp(record["eval_loss"])
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                state = adapter_state(model)
                bundle = {"state": state, "args": vars(args), "step": step,
                          "trained_tokens": step * args.block * args.grad_accum}
                atomic_save(bundle, args.output / "adapter-latest.pt")
                atomic_save({**bundle, "optimizer": optimizer.state_dict()}, resume_path)
                atomic_json({"step": step, "tokens": bundle["trained_tokens"],
                             "last_loss": record["loss"], "last_eval_perplexity": record.get("eval_perplexity")},
                            args.output / "status.json")
                print(f"saved resumable checkpoint at step {step}", flush=True)
        atomic_save(bundle, args.output / "adapter.pt")
        atomic_json({"status": "complete", "steps": args.steps,
                     "trained_tokens": args.steps * args.block * args.grad_accum},
                    args.output / "complete.json")
