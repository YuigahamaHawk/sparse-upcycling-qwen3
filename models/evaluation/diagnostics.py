"""Locate the first nonfinite layer in dense or upcycled Qwen3 inference."""
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3MoeForCausalLM
from models.components.moe import install_qwen3_patches


class ForwardDiagnostics:
    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        install_qwen3_patches()
        cls = Qwen3ForCausalLM if args.dense else Qwen3MoeForCausalLM
        model = cls.from_pretrained(args.model, torch_dtype=torch.bfloat16,
            device_map="auto", max_memory={i: "28GiB" for i in range(torch.cuda.device_count())},
            attn_implementation="eager").eval()
        print(model.hf_device_map, flush=True)
        if args.lora:
            from models.components.lora import inject_expert_lora
            inject_expert_lora(model, 4)
        for i, layer in enumerate(model.model.layers):
            def hook(module, inputs, output, i=i):
                h = output[0]
                finite = bool(torch.isfinite(h).all())
                print(i, finite, float(h.float().nan_to_num().abs().max()), flush=True)
                if not finite:
                    raise RuntimeError(f"first nonfinite layer {i}")
            layer.register_forward_hook(hook)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.data:
            from models.trainers.smoke_cpt import batches
            from pathlib import Path
            ids = next(batches(Path(args.data), tokenizer, 64, 42))
        else:
            ids = tokenizer("Sparse upcycling starts from a dense model checkpoint.",
                            return_tensors="pt").input_ids
        ids = ids.to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(ids, output_router_logits=False, use_cache=False)
        print("logits finite", bool(torch.isfinite(out.logits).all()), flush=True)
