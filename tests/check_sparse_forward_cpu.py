"""Verify fixed expert iteration preserves the upstream MoE calculation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe import modeling_qwen3_moe


def main():
    assert torch.version.cuda is None, "Use a CPU-only PyTorch build for this check"
    original = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward
    from models.components.moe import install_qwen3_patches, stable_sparse_moe_forward
    install_qwen3_patches()

    torch.manual_seed(17)
    config = Qwen3MoeConfig(hidden_size=64, intermediate_size=48,
                            moe_intermediate_size=32, num_experts=8,
                            num_experts_per_tok=2)
    block = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock(config)
    inputs = torch.randn(2, 11, 64)

    def run(fn):
        block.zero_grad(set_to_none=True)
        x = inputs.clone().requires_grad_()
        y, logits = fn(block, x)
        (y.square().mean() + logits.square().mean()).backward()
        return y.detach(), logits.detach(), x.grad.detach(), block.gate.weight.grad.detach().clone()

    reference = run(original)
    patched = run(stable_sparse_moe_forward)
    for name, a, b in zip(("output", "logits", "input_grad", "router_grad"), reference, patched):
        maximum = (a - b).abs().max().item()
        print(name, maximum)
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6), name


if __name__ == "__main__":
    main()
