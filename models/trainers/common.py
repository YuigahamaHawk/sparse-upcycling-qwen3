"""Shared evaluation and learning-rate helpers for stage trainers."""
import math
import torch


def schedule(step, args, total_steps):
    if step <= args.warmup:
        return args.lr * step / max(1, args.warmup)
    progress = (step - args.warmup) / max(1, total_steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def eval_loss(model, blocks, device):
    model.eval()
    values = []
    with torch.no_grad():
        for ids in blocks:
            ids = ids.to(device)
            out = model(input_ids=ids, labels=ids, use_cache=False,
                        output_router_logits=False)
            values.append(float(out.loss))
    model.train()
    return sum(values) / len(values)
