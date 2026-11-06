"""Qwen3 architecture contract: tensor shapes, sparse layers and HF loaders.

Qwen3.5/3.8's hybrid attention is a different backbone and is not supported by
this adapter. Do not bypass that distinction by changing a model_type string.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3Shape:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int = 0
    num_experts_per_tok: int = 0

    @classmethod
    def from_config(cls, config, *, require_dense=False):
        data = config if isinstance(config, dict) else config.to_dict()
        kind = data.get("model_type")
        if kind not in ("qwen3", "qwen3_moe") or require_dense and kind != "qwen3":
            raise ValueError(f"Unsupported backbone {kind!r}: expected Qwen3 {'dense' if require_dense else 'dense/MoE'}")
        return cls(model_type=kind, hidden_size=data["hidden_size"],
                   intermediate_size=data["intermediate_size"],
                   num_hidden_layers=data["num_hidden_layers"],
                   num_attention_heads=data["num_attention_heads"],
                   num_key_value_heads=data["num_key_value_heads"],
                   head_dim=data.get("head_dim", data["hidden_size"] // data["num_attention_heads"]),
                   num_experts=data.get("num_experts", 0),
                   num_experts_per_tok=data.get("num_experts_per_tok", 0))


def sparse_layers(model):
    """Yield Qwen3 sparse decoder layers; indices remain checkpoint-compatible."""
    for index, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "experts"):
            yield index, layer


class Qwen3Backbone:
    """One architecture adapter shared by training and conversion."""

    @staticmethod
    def load_pretrained(path, **kwargs):
        from transformers import Qwen3MoeForCausalLM
        from models.components.moe import install_qwen3_patches
        install_qwen3_patches()
        return Qwen3MoeForCausalLM.from_pretrained(path, **kwargs)

    @staticmethod
    def prepare_adapters(model, rank):
        from models.components.lora import prepare_adapter_training
        return prepare_adapter_training(model, rank)

    @staticmethod
    def upcycled_config(source_config, *, sparse, experts, top_k, normalize_topk):
        from transformers import Qwen3MoeConfig
        shape = Qwen3Shape.from_config(source_config, require_dense=True)
        args = {k: v for k, v in source_config.items()
                if k not in {"architectures", "model_type", "transformers_version"}}
        config = Qwen3MoeConfig(**args, decoder_sparse_step=1,
            moe_intermediate_size=shape.intermediate_size,
            num_experts=experts, num_experts_per_tok=top_k,
            norm_topk_prob=normalize_topk, output_router_logits=False,
            router_aux_loss_coef=0.01,
            mlp_only_layers=sorted(set(range(shape.num_hidden_layers)) - sparse))
        config.architectures = ["Qwen3MoeForCausalLM"]
        return config
