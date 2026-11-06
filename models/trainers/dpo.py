"""DPO preference alignment on top of a merged SFT Qwen3-MoE checkpoint.

The reference model is the frozen SFT checkpoint itself: its sequence
log-probabilities are precomputed once before any optimizer update and
cached, so only one model instance ever resides in memory. Fresh expert
LoRA adapters have B=0, so the reference pass still equals the frozen SFT
function. Only these adapters and routers receive DPO updates afterward.

Input JSONL rows need "prompt", "chosen" and "rejected" fields. The Qwen3 backbone installs the stable MoE forward and aux-loss patches required
on Corex. Router auxiliary loss is intentionally off here: routers learn
through the policy gradient path only, as in standard DPO.
"""
import hashlib
import json
import math
import os
import random
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from models.components.lora import adapter_state
from models.trainers.checkpoint import atomic_save, atomic_json
from models.trainers.basic_trainer import BasicTrainer
from models.trainers.common import schedule


STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def should_stop(deadline=None):
    return STOP_REQUESTED or (deadline is not None and time.time() >= deadline)


def reference_key(model_path, pairs):
    digest = hashlib.sha256()
    digest.update(str(model_path.resolve()).encode())
    for name in ("config.json", "merge_manifest.json"):
        file = model_path / name
        if file.is_file():
            digest.update(file.read_bytes())
    digest.update(json.dumps(pairs, separators=(",", ":")).encode())
    return digest.hexdigest()


def save_reference(path, chosen, rejected, key):
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, chosen=chosen, rejected=rejected, fingerprint=np.asarray(key))
    os.replace(tmp, path)


def build_pairs(path, tokenizer, max_len):
    pairs = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        chosen_ids = tokenizer.encode(row["chosen"], add_special_tokens=False)
        rejected_ids = tokenizer.encode(row["rejected"], add_special_tokens=False)
        chosen_ids = chosen_ids + [tokenizer.eos_token_id]
        rejected_ids = rejected_ids + [tokenizer.eos_token_id]
        if (not prompt_ids or not row["chosen"].strip() or not row["rejected"].strip()
                or len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)) > max_len):
            skipped += 1
            continue
        pairs.append((prompt_ids, chosen_ids, rejected_ids))
    print(f"pairs={len(pairs)} skipped={skipped}", flush=True)
    if not pairs:
        raise ValueError(f"No usable rows in {path}")
    return pairs


def sequence_logprob(model, prompt_ids, completion_ids, device):
    """Sum log-probability of the completion tokens under the current model."""
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    hidden = model.model(input_ids=ids, use_cache=False,
                         output_router_logits=False).last_hidden_state
    start = len(prompt_ids)
    # Row t of the slice predicts token t+1; completion tokens start at index start.
    selected = hidden[0, start - 1:-1]
    logits = model.lm_head(selected)
    targets = ids[0, start:].to(logits.device)
    token_logp = -F.cross_entropy(logits.float(), targets, reduction="none")
    return token_logp.sum()


@torch.no_grad()
def precompute_reference(model, pairs, device, path, key, deadline=None):
    chosen_ref, rejected_ref = [], []
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            if "fingerprint" in data and str(data["fingerprint"].item()) == key:
                chosen_ref, rejected_ref = data["chosen"].tolist(), data["rejected"].tolist()
                if len(chosen_ref) != len(rejected_ref) or len(chosen_ref) > len(pairs):
                    raise ValueError("Invalid reference cache lengths")
                if not np.isfinite(chosen_ref).all() or not np.isfinite(rejected_ref).all():
                    raise ValueError("Nonfinite reference cache")
                print(f"reference cache {len(chosen_ref)}/{len(pairs)} from {path}", flush=True)
            else:
                print("Reference provenance changed; recomputing", flush=True)
    model.eval()
    for i, (prompt_ids, chosen_ids, rejected_ids) in enumerate(pairs):
        if i < len(chosen_ref):
            continue
        if should_stop(deadline):
            save_reference(path, chosen_ref, rejected_ref, key)
            raise SystemExit(75)
        chosen_ref.append(float(sequence_logprob(model, prompt_ids, chosen_ids, device)))
        rejected_ref.append(float(sequence_logprob(model, prompt_ids, rejected_ids, device)))
        if not math.isfinite(chosen_ref[-1]) or not math.isfinite(rejected_ref[-1]):
            raise RuntimeError(f"Nonfinite reference score at pair {i}")
        if (i + 1) % 25 == 0:
            save_reference(path, chosen_ref, rejected_ref, key)
            atomic_json({"phase": "reference", "completed": i + 1, "total": len(pairs),
                         "updated_utc": datetime.now().astimezone().isoformat()},
                        path.with_suffix(".status.json"))
            print(f"reference pass {i + 1}/{len(pairs)}", flush=True)
    model.train()
    chosen_ref = np.asarray(chosen_ref, dtype=np.float64)
    rejected_ref = np.asarray(rejected_ref, dtype=np.float64)
    save_reference(path, chosen_ref, rejected_ref, key)
    print(f"cached reference log-probabilities to {path}", flush=True)
    return chosen_ref, rejected_ref


def resumable_groups(order, grad_accum, epoch, steps_per_epoch, start_step):
    for offset in range(0, len(order), grad_accum):
        group_step = epoch * steps_per_epoch + offset // grad_accum + 1
        if group_step > start_step:
            yield group_step, order[offset:offset + grad_accum]


class DPOTrainer(BasicTrainer):
    def run(self):
        args = self.args
        args.output.mkdir(parents=True, exist_ok=True)
        deadline = datetime.fromisoformat(args.stop_at).timestamp() if args.stop_at else None
        if args.stop_at and datetime.fromisoformat(args.stop_at).tzinfo is None:
            raise ValueError("--stop-at must include timezone")
        if should_stop(deadline):
            raise SystemExit(75)
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        if not args.resume and (args.output / "metrics.jsonl").exists():
            raise ValueError("Existing metrics require --resume or a new output directory")
        self.seed()

        n_gpus = self.require_gpus()
        model = self.load_model(n_gpus, reject_offload=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        pairs = build_pairs(args.data, tokenizer, args.max_len)
        val_pairs = build_pairs(args.eval_data, tokenizer, args.max_len)
        input_device = model.model.embed_tokens.weight.device

        ref_path = args.output / "reference_logps.npz"
        fingerprint = reference_key(args.model, pairs)

        steps_per_epoch = math.ceil(len(pairs) / args.grad_accum)
        total_steps = steps_per_epoch * args.epochs

        params, optimizer = self.prepare_optimizer(model)

        if args.smoke_test and not args.resume:
            idx = max(range(len(pairs)), key=lambda i: 2 * len(pairs[i][0]) + len(pairs[i][1]) + len(pairs[i][2]))
            prompt_ids, chosen_ids, rejected_ids = pairs[idx]
            for device_idx in range(n_gpus):
                torch.cuda.reset_peak_memory_stats(device_idx)
            pc = sequence_logprob(model, prompt_ids, chosen_ids, input_device)
            pr = sequence_logprob(model, prompt_ids, rejected_ids, input_device)
            # The policy is still exactly the SFT reference. Detached scores give
            # the initial DPO gradient without first scoring the entire corpus.
            smoke_loss = -F.logsigmoid(args.beta * ((pc - pc.detach()) - (pr - pr.detach())))
            smoke_loss.backward()
            smoke_grad = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if not math.isfinite(float(smoke_loss.detach())) or not math.isfinite(smoke_grad):
                raise RuntimeError("Longest-pair smoke test is nonfinite")
            atomic_json({"pair_index": idx, "chosen_length": len(prompt_ids) + len(chosen_ids),
                         "rejected_length": len(prompt_ids) + len(rejected_ids),
                         "loss": float(smoke_loss.detach()), "grad_norm": smoke_grad,
                         "optimizer_updates": 0,
                         "peak_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(n_gpus)]},
                        args.output / "smoke_test.json")
            optimizer.zero_grad(set_to_none=True)
            del pc, pr, smoke_loss
            print("Longest-pair backward passed; no optimizer update performed", flush=True)

        ref_chosen, ref_rejected = precompute_reference(model, pairs, input_device, ref_path,
                                                       fingerprint, deadline)
        val_ref_chosen, val_ref_rejected = precompute_reference(
            model, val_pairs, input_device, args.output / "validation_reference_logps.npz",
            reference_key(args.model, val_pairs), deadline)

        run_info = {"model": str(args.model), "data": str(args.data),
                    "pairs": len(pairs), "val_pairs": len(val_pairs),
                    "epochs": args.epochs, "target_steps": total_steps, "beta": args.beta,
                    "grad_accum": args.grad_accum, "max_len": args.max_len,
                    "rank": args.rank,
                    "trainable_parameters": sum(v.numel() for v in params),
                    "gpu_count": n_gpus, "device_map": model.hf_device_map,
                    "reference_fingerprint": fingerprint, "stop_at": args.stop_at,
                    "physical_gpus": os.environ.get("APPROVED_PHYSICAL_GPUS")}
        atomic_json(run_info, args.output / "run.json")
        print(json.dumps(run_info), flush=True)

        metrics_path = args.output / "metrics.jsonl"
        start_step = 0
        trained_pairs = 0
        trained_tokens = 0
        resume_path = args.output / "resume.pt"
        if args.resume:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            if checkpoint["args"]["model"] != str(args.model):
                raise ValueError("Resume checkpoint model differs from current run")
            for name in ("data", "eval_data", "seed", "rank", "max_len", "grad_accum", "beta", "epochs"):
                if str(checkpoint["args"][name]) != str(getattr(args, name)):
                    raise ValueError(f"Resume configuration differs: {name}")
            if checkpoint.get("reference_fingerprint") != fingerprint:
                raise ValueError("Resume reference fingerprint differs")
            missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint["step"]
            trained_pairs = checkpoint.get("trained_pairs", 0)
            trained_tokens = checkpoint.get("trained_tokens", 0)
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

        @torch.no_grad()
        def eval_policy():
            model.eval()
            margins, wins, raw_wins = [], 0.0, 0
            for idx, (prompt_ids, chosen_ids, rejected_ids) in enumerate(val_pairs):
                pc = float(sequence_logprob(model, prompt_ids, chosen_ids, input_device))
                pr = float(sequence_logprob(model, prompt_ids, rejected_ids, input_device))
                margin = args.beta * ((pc - val_ref_chosen[idx]) - (pr - val_ref_rejected[idx]))
                margins.append(margin)
                wins += float(margin > 0) + 0.5 * float(margin == 0)
                raw_wins += int(pc > pr)
            model.train()
            return {"val_dpo_accuracy": wins / len(val_pairs),
                    "val_mean_margin": sum(margins) / len(margins),
                    "val_raw_preference_accuracy": raw_wins / len(val_pairs)}

        run_start = time.monotonic()
        opt_step = start_step
        def save_checkpoint(record, epoch):
            bundle = {"state": adapter_state(model), "args": vars(args), "step": opt_step,
                      "epoch": epoch, "trained_pairs": trained_pairs, "trained_tokens": trained_tokens,
                      "reference_fingerprint": fingerprint, "torch_rng_state": torch.get_rng_state()}
            atomic_save(bundle, args.output / "adapter-latest.pt")
            atomic_save({**bundle, "optimizer": optimizer.state_dict()}, resume_path)
            atomic_json({"step": opt_step, "last_loss": record.get("loss"),
                         "last_reward_accuracy": record.get("reward_accuracy"),
                         "last_mean_margin": record.get("mean_margin"),
                         "trained_pairs": trained_pairs, "trained_tokens": trained_tokens,
                         "updated_utc": datetime.now().astimezone().isoformat()}, args.output / "status.json")
            print(f"saved resumable checkpoint at step {opt_step}", flush=True)

        atomic_json(eval_policy(), args.output / "initial_eval.json")
        record = {}
        for epoch in range(args.epochs):
            if opt_step >= total_steps:
                break
            order = list(range(len(pairs)))
            random.Random(args.seed + epoch).shuffle(order)
            for group_step, group in resumable_groups(order, args.grad_accum, epoch, steps_per_epoch, start_step):
                if should_stop(deadline):
                    save_checkpoint(record, epoch + 1)
                    atomic_json({"status": "stopped_at_deadline", "steps": opt_step},
                                args.output / "stopped.json")
                    return
                if group_step != opt_step + 1:
                    raise RuntimeError("Non-contiguous optimizer step")
                optimizer.zero_grad(set_to_none=True)
                losses, margins, wins = [], [], 0
                for idx in group:
                    prompt_ids, chosen_ids, rejected_ids = pairs[idx]
                    pc = sequence_logprob(model, prompt_ids, chosen_ids, input_device)
                    pr = sequence_logprob(model, prompt_ids, rejected_ids, input_device)
                    rc = float(ref_chosen[idx])
                    rr = float(ref_rejected[idx])
                    logits_pair = args.beta * ((pc - rc) - (pr - rr))
                    loss = -F.logsigmoid(logits_pair)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(f"Nonfinite DPO loss at step {opt_step + 1}")
                    (loss / len(group)).backward()
                    losses.append(float(loss.detach()))
                    margins.append(float(logits_pair.detach()))
                    wins += int(float(logits_pair.detach()) > 0)
                lr = schedule(opt_step + 1, args, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"Nonfinite gradient at opt step {opt_step + 1}")
                optimizer.step()
                trained_pairs += len(group)
                trained_tokens += sum(2 * len(pairs[i][0]) + len(pairs[i][1]) + len(pairs[i][2]) for i in group)
                opt_step += 1
                record = {"step": opt_step, "epoch": epoch + 1,
                          "loss": sum(losses) / len(losses),
                          "reward_accuracy": wins / len(margins),
                          "mean_margin": sum(margins) / len(margins),
                          "grad_norm": grad_norm, "lr": lr,
                          "trained_pairs": trained_pairs, "trained_tokens": trained_tokens,
                          "elapsed_seconds": round(time.monotonic() - run_start, 1)}
                if opt_step % args.eval_every == 0 or opt_step == total_steps:
                    record.update(eval_policy())
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if opt_step % args.save_every == 0 or opt_step == total_steps:
                    save_checkpoint(record, epoch + 1)
                if opt_step >= total_steps:
                    break
        final = {"state": adapter_state(model), "args": vars(args),
                 "step": opt_step, "trained_pairs": trained_pairs, "trained_tokens": trained_tokens,
                 "reference_fingerprint": fingerprint}
        atomic_save(final, args.output / "adapter.pt")
        atomic_json({"status": "complete", "steps": opt_step}, args.output / "complete.json")
        print("DPO adapter complete", flush=True)
