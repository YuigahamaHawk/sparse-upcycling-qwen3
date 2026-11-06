"""Check structural and numerical correctness of the converter on a tiny model."""
import subprocess
import sys
import tempfile
from pathlib import Path

import gc
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeForCausalLM



def main():
    assert torch.version.cuda is None, "Use a CPU-only PyTorch build for this check"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dense_dir, moe_dir = root / "dense", root / "moe"
        cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128)
        torch.manual_seed(11)
        dense = Qwen3ForCausalLM(cfg).eval()
        dense.save_pretrained(dense_dir, safe_serialization=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/conversion/upcycle.py"),
            "--source", str(dense_dir), "--output", str(moe_dir),
            "--experts", "4", "--top-k", "2", "--layers", "alternating",
            "--normalize-topk"], check=True)
        moe = Qwen3MoeForCausalLM.from_pretrained(moe_dir).eval()
        # Release file-backed safetensors storage before Windows temp cleanup.
        for parameter in moe.parameters():
            parameter.data = parameter.detach().clone()
        for i in (1, 3):
            original = dense.model.layers[i].mlp
            copied = moe.model.layers[i].mlp
            for expert in copied.experts:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    assert torch.equal(getattr(original, name).weight,
                                       getattr(expert, name).weight)
        ids = torch.randint(0, 256, (2, 12))
        with torch.no_grad():
            a = dense(ids).logits
            b = moe(ids, output_router_logits=False).logits
        delta = (a - b).abs().max().item()
        print(f"maximum dense/MoE logit difference: {delta:.8g}")
        assert delta < 1e-4, delta
        del moe, dense
        gc.collect()


if __name__ == "__main__":
    main()
