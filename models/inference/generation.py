"""Greedy generation sanity check for a merged Qwen3-MoE checkpoint.

Runs a fixed set of instructions through the model with the Alpaca prompt
template and stores the generations for qualitative before/after comparison.
"""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3MoeForCausalLM
from models.components.lora import inject_expert_lora
from models.components.moe import install_qwen3_patches
from models.inference.common import TEMPLATE


PROMPTS = [
    {"instruction": "用一句话解释什么是机器学习。"},
    {"instruction": "Write a Python function that reverses a string.",
     "input": "def reverse(s: str) -> str:"},
    {"instruction": "Summarize the plot of Romeo and Juliet in three sentences."},
    {"instruction": "List three tips for writing clear technical documentation."},
    {"instruction": "把“今天天气很好，我们去公园散步吧”翻译成英文。"},
    {"instruction": "Explain the difference between TCP and UDP to a beginner."},
    {"instruction": "给一个初中生解释为什么天空是蓝色的。"},
    {"instruction": "Draft a polite email declining a meeting invitation."},
]

class SampleGenerator:
    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        install_qwen3_patches()
        model = Qwen3MoeForCausalLM.from_pretrained(args.model,
            torch_dtype=torch.bfloat16, device_map="auto",
            max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
            attn_implementation="eager").eval()
        if args.adapter:
            adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
            rank = int(adapter["args"]["rank"])
            inject_expert_lora(model, rank)
            missing, unexpected = model.load_state_dict(adapter["state"], strict=False)
            if unexpected or any(key.endswith((".A", ".B")) for key in missing):
                raise RuntimeError("Adapter parameters were not loaded completely")
            model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        device = model.model.embed_tokens.weight.device
        rows = []
        for item in PROMPTS:
            input_part = f"### Input:\n{item['input']}\n\n" if item.get("input") else ""
            prompt = TEMPLATE.format(instruction=item["instruction"], input_part=input_part)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(ids, do_sample=False, max_new_tokens=args.max_new_tokens,
                                     use_cache=True, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            rows.append({"instruction": item["instruction"], "prompt": prompt,
                         "model": str(args.model), "adapter": str(args.adapter) if args.adapter else None,
                         "output": text})
            print(f"--- {item['instruction']}\n{text}\n", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {args.output}", flush=True)
