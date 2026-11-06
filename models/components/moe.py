"""Copied-expert routing forward and idempotent Corex/HF workarounds.

Importing this module does not monkeypatch Transformers. Install explicitly
immediately before loading a Qwen3-MoE model.
"""
import torch
from torch.nn import functional as F


def _original_balance_loss():
    from transformers.models.qwen3_moe import modeling_qwen3_moe as hf
    return getattr(hf, "_upcycling_original_balance_loss", hf.load_balancing_loss_func)


def sparse_balance_loss(logits, *args, **kwargs):
    """Transformers 4.53 includes None for dense layers in the router tuple."""
    if isinstance(logits, tuple):
        logits = tuple(item for item in logits if item is not None)
    return _original_balance_loss()(logits, *args, **kwargs)


def stable_sparse_moe_forward(self, hidden_states):
    """Avoid a Corex nonzero() index corruption seen after long training.

    This preserves Hugging Face's routing and expert arithmetic. Only the
    eight-element list of active experts is replaced by a fixed Python loop.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    for expert_idx in range(self.num_experts):
        rank_idx, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        current_state = hidden_states[token_idx]
        current_hidden_states = self.experts[expert_idx](current_state)
        current_hidden_states = current_hidden_states * routing_weights[token_idx, rank_idx, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

def install_qwen3_patches():
    from transformers.models.qwen3_moe import modeling_qwen3_moe as hf
    if not hasattr(hf, "_upcycling_original_balance_loss"):
        hf._upcycling_original_balance_loss = hf.load_balancing_loss_func
        hf._upcycling_original_sparse_forward = hf.Qwen3MoeSparseMoeBlock.forward
    hf.load_balancing_loss_func = sparse_balance_loss
    hf.Qwen3MoeSparseMoeBlock.forward = stable_sparse_moe_forward
