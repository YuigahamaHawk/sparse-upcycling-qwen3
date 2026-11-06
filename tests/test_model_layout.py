"""CPU component and dependency regressions for the models/ refactor."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from models.backbones.qwen3 import Qwen3Shape
from models.components.lora import LoRALinear, adapter_state, inject_expert_lora
from models.components.moe import stable_sparse_moe_forward
from models.paths import PROJECT_ROOT
from models.trainers.basic_trainer import BasicTrainer


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        self.experts = nn.ModuleList([Expert(), Expert()])
        self.num_experts, self.top_k, self.norm_topk_prob = 2, 2, True

    forward = stable_sparse_moe_forward


class TinyComponents(nn.Module):
    """Only small component tensors, not a pretrained/HF language model."""
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(7, 4)
        layer = nn.Module()
        layer.mlp = SparseBlock()
        self.model.layers = nn.ModuleList([layer])


class ComponentTrainer(BasicTrainer):
    def run(self):
        raise AssertionError("GPU training must not be launched by these tests")


class ModelLayoutTests(unittest.TestCase):
    def test_model_library_does_not_depend_on_cli_scripts(self):
        for path in (PROJECT_ROOT / 'models').rglob('*.py'):
            if path.name == 'upcycling_inference.py':
                continue  # explicit import compatibility alias, not implementation
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    name = node.module or ''
                    self.assertFalse(name.startswith(('scripts', 'train_adapters', 'train_long', 'train_sft', 'train_dpo', 'serving_common', 'gpu_lease')), str(path))
                if isinstance(node, ast.Import):
                    self.assertFalse(any(n.name.startswith('scripts') for n in node.names), str(path))
            self.assertNotIn('sys.path.insert', path.read_text(encoding='utf-8'), str(path))

    def test_qwen3_shapes_and_new_architecture_boundary(self):
        shape = Qwen3Shape.from_config({'model_type': 'qwen3', 'hidden_size': 64,
                   'intermediate_size': 128, 'num_hidden_layers': 4,
                   'num_attention_heads': 4, 'num_key_value_heads': 2})
        self.assertEqual((shape.head_dim, shape.num_hidden_layers), (16, 4))
        with self.assertRaisesRegex(ValueError, 'Unsupported backbone'):
            Qwen3Shape.from_config({'model_type': 'qwen3_5'})

    def test_lora_forward_scaling_and_legacy_state_names(self):
        torch.manual_seed(12)
        base = nn.Linear(4, 3, bias=False)
        layer = LoRALinear(base, rank=2, alpha=8)
        with torch.no_grad():
            layer.B.fill_(.25)
        inputs = torch.randn(5, 4)
        expected = base(inputs) + (inputs @ layer.A.T @ layer.B.T) * 4
        torch.testing.assert_close(layer(inputs), expected)
        self.assertEqual(set(layer.state_dict()), {'A', 'B', 'base.weight'})

    def test_converter_preserves_dense_tensors_and_copies_selected_experts(self):
        from safetensors.torch import load_file, save_file
        from models.conversion.upcycle import Qwen3Upcycler
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'dense', root / 'moe'
            source.mkdir()
            config = {'model_type': 'qwen3', 'hidden_size': 4, 'intermediate_size': 6,
                      'num_hidden_layers': 2, 'num_attention_heads': 1,
                      'num_key_value_heads': 1, 'head_dim': 4, 'vocab_size': 7,
                      'max_position_embeddings': 32, 'tie_word_embeddings': False}
            (source / 'config.json').write_text(json.dumps(config))
            weights = {'model.embed_tokens.weight': torch.randn(7, 4)}
            for layer in range(2):
                for name, shape in [('gate_proj', (6, 4)), ('up_proj', (6, 4)), ('down_proj', (4, 6))]:
                    weights[f'model.layers.{layer}.mlp.{name}.weight'] = torch.randn(*shape)
            save_file(weights, str(source / 'weights.safetensors'))
            args = SimpleNamespace(source=source, output=output, experts=2, top_k=2,
                   layers='alternating', router_std=.02, normalize_topk=True, seed=42)
            Qwen3Upcycler(args).run()
            merged = {}
            for shard in output.glob('*.safetensors'):
                merged.update({k: v.clone() for k, v in load_file(str(shard)).items()})
            torch.testing.assert_close(merged['model.embed_tokens.weight'], weights['model.embed_tokens.weight'])
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                torch.testing.assert_close(merged[f'model.layers.0.mlp.{name}.weight'], weights[f'model.layers.0.mlp.{name}.weight'])
                for expert in range(2):
                    torch.testing.assert_close(merged[f'model.layers.1.mlp.experts.{expert}.{name}.weight'], weights[f'model.layers.1.mlp.{name}.weight'])
            self.assertEqual(tuple(merged['model.layers.1.mlp.gate.weight'].shape), (2, 4))
            result = json.loads((output / 'config.json').read_text())
            self.assertEqual(result['model_type'], 'qwen3_moe')
            self.assertEqual(result['mlp_only_layers'], [0])
            self.assertTrue(result['norm_topk_prob'])

    def test_basic_trainer_freezes_base_updates_adapters_and_round_trips(self):
        torch.manual_seed(9)
        components = TinyComponents()
        args = SimpleNamespace(rank=2, lr=.01, model=Path('unused'), seed=9, max_memory='1GiB')
        trainer = ComponentTrainer(args)
        inputs = torch.randn(2, 3, 4)
        before = components.model.layers[0].mlp(inputs)[0].detach()
        with patch.object(torch.cuda, 'device_count', side_effect=AssertionError('GPU probe forbidden')):
            params, optimizer = trainer.prepare_optimizer(components)
        torch.testing.assert_close(components.model.layers[0].mlp(inputs)[0], before)
        trainable = {n for n, p in components.named_parameters() if p.requires_grad}
        expected = {'model.layers.0.mlp.gate.weight'}
        for expert in range(2):
            for projection in ('gate_proj', 'up_proj', 'down_proj'):
                expected.update(f'model.layers.0.mlp.experts.{expert}.{projection}.{part}' for part in ['A', 'B'])
        self.assertEqual(trainable, expected)
        frozen = {n: p.detach().clone() for n, p in components.named_parameters() if not p.requires_grad}
        output, router_logits = components.model.layers[0].mlp(inputs)
        loss = output.square().mean() + router_logits.square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in params))
        optimizer.step()
        for name, original in frozen.items():
            torch.testing.assert_close(dict(components.named_parameters())[name], original)
        self.assertEqual(set(adapter_state(components)), expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'resume.pt'
            trainer.atomic_save({'state': adapter_state(components), 'args': vars(args),
                                 'optimizer': optimizer.state_dict(), 'step': 1}, path)
            restored = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(restored['step'], 1)
            self.assertEqual(set(restored['state']), expected)
        with self.assertRaisesRegex(ValueError, 'already installed'):
            inject_expert_lora(components, 2)
        self.assertFalse(torch.cuda.is_initialized())

    def test_all_stage_trainers_share_basic_trainer_without_loading_in_constructor(self):
        from models.trainers import CPTTrainer, DPOTrainer, SFTTrainer, SmokeCPTTrainer
        args = SimpleNamespace()
        with patch.object(torch.cuda, 'device_count', side_effect=AssertionError('GPU probe forbidden')):
            for cls in [CPTTrainer, DPOTrainer, SFTTrainer, SmokeCPTTrainer]:
                trainer = cls(args)
                self.assertIsInstance(trainer, BasicTrainer)
                self.assertIsNone(trainer.model)
                self.assertIsNone(trainer.optimizer)

    def test_legacy_component_imports_are_the_same_implementation(self):
        from scripts.training.smoke_cpt import LoRALinear as legacy_lora
        from models.upcycling_inference import InferenceEngine as legacy_engine
        from models.inference.engine import InferenceEngine
        self.assertIs(legacy_lora, LoRALinear)
        self.assertIs(legacy_engine, InferenceEngine)


if __name__ == '__main__':
    unittest.main()
