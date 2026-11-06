"""Expert LoRA modules; legacy A/B/base parameter names are preserved."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.backbones.qwen3 import sparse_layers


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        result = self.base(x)
        update = F.linear(F.linear(x.float(), self.A), self.B)
        return result + update.to(result.dtype) * self.scale


def adapter_state(model):
    return {k: v.detach().cpu() for k, v in model.named_parameters()
            if v.requires_grad}

def inject_expert_lora(model, rank, alpha=8):
    """Wrap expert projections once, preserving historical state_dict keys."""
    for _, layer in sparse_layers(model):
        for expert in layer.mlp.experts:
            for name in ("gate_proj", "up_proj", "down_proj"):
                base = getattr(expert, name)
                if isinstance(base, LoRALinear):
                    raise ValueError("expert LoRA is already installed")
                setattr(expert, name, LoRALinear(base, rank, alpha))


def prepare_adapter_training(model, rank, alpha=8):
    for param in model.parameters():
        param.requires_grad_(False)
    for _, layer in sparse_layers(model):
        layer.mlp.gate.weight.requires_grad_(True)
    inject_expert_lora(model, rank, alpha)
    return [param for param in model.parameters() if param.requires_grad]
