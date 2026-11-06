"""CPU regressions for serialized prompts, stage helpers and adapter loading."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.components.lora import adapter_state, prepare_adapter_training
from models.inference.engine import InferenceEngine
from models.trainers.common import eval_loss, schedule
from scripts.data.prepare_domain_data import alpaca_rows
from test_model_layout import TinyComponents


class SharedHelpersTests(unittest.TestCase):
    def test_alpaca_data_preserves_historical_prompt_bytes(self):
        head = ("Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n")
        rows = list(alpaca_rows([
            {"instruction": "  Explain  ", "input": "  context  ", "output": "  answer  "},
            {"instruction": "  ", "input": "", "output": "answer"},
            {"instruction": "skip", "output": "  "},
        ]))
        self.assertEqual(rows, [
            {"prompt": head + "Explain\n\n### Input:\ncontext\n\n### Response:\n", "completion": "answer"},
            {"prompt": head + "\n\n### Response:\n", "completion": "answer"},
        ])

    def test_warmup_and_cosine_learning_rate_boundaries(self):
        args = SimpleNamespace(warmup=2, lr=.01, min_lr=.001)
        for step, expected in [(0, 0), (1, .005), (2, .01), (4, .0055), (6, .001)]:
            self.assertAlmostEqual(schedule(step, args, 6), expected)
        args.warmup = 0
        self.assertAlmostEqual(schedule(0, args, 6), 0)
        self.assertAlmostEqual(schedule(6, args, 6), .001)

    def test_shared_validation_disables_gradients_and_restores_training(self):
        class Model(torch.nn.Module):
            def forward(self, input_ids, labels, use_cache, output_router_logits):
                self_test.assertFalse(self.training)
                self_test.assertFalse(torch.is_grad_enabled())
                self_test.assertIs(input_ids, labels)
                self_test.assertFalse(use_cache)
                self_test.assertFalse(output_router_logits)
                return SimpleNamespace(loss=input_ids.float().mean())
        self_test = self
        model = Model()
        loss = eval_loss(model, [torch.tensor([[1, 3]]), torch.tensor([[5, 7]])], 'cpu')
        self.assertEqual(loss, 4.)
        self.assertTrue(model.training)

    def test_serving_loads_expert_adapters_and_routers_using_shared_injection(self):
        torch.manual_seed(23)
        source = TinyComponents()
        target = copy.deepcopy(source)
        prepare_adapter_training(source, rank=2)
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if parameter.requires_grad:
                    parameter.fill_(.07 if name.endswith('.B') else .11)
        state = adapter_state(source)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'base'
            bundle = Path(tmp) / 'adapter.pt'
            torch.save({'state': state, 'args': {'rank': 2, 'model': str(base)}}, bundle)
            engine = InferenceEngine.__new__(InferenceEngine)  # no lease/model/GPU startup
            engine.model = target
            with patch.object(torch.cuda, 'device_count', side_effect=AssertionError('GPU probe forbidden')):
                info = engine._load_adapter({'adapter': str(bundle), 'path': str(base)}, torch)
            self.assertEqual(info['tensors'], len(state))
            inputs = torch.randn(2, 3, 4)
            source_output, source_logits = source.model.layers[0].mlp(inputs)
            target_output, target_logits = target.model.layers[0].mlp(inputs)
            torch.testing.assert_close(target_output, source_output)
            torch.testing.assert_close(target_logits, source_logits)
            with self.assertRaisesRegex(RuntimeError, 'base differs'):
                engine._load_adapter({'adapter': str(bundle), 'path': str(base / 'wrong')}, torch)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
