"""CPU-only regression tests for serving contracts and GPU lease enforcement."""
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from models.runtime import gpu_lease
from models.inference.api import create_app
from models.inference.common import (BusyError, RequestError, chat_prompt, check_parameters,
                            instruction_prompt, registry, trim_stop, without_allocator_warmup)


class FakeEngine:
    def __init__(self):
        self.specs = registry()
        self.model_id, self.phase = "moe-sft", "ready"
        self.lock = threading.Lock()
        self.last_prompt = None

    def check_lease(self):
        pass

    def status(self):
        return {"model": self.model_id, "phase": self.phase}

    def load(self, name):
        if name not in self.specs:
            raise RequestError("unknown model")
        if self.lock.locked():
            raise BusyError("busy")
        self.model_id, self.phase = name, "ready"
        return self.status()

    def generate(self, prompt, **kw):
        check_parameters(kw["max_new_tokens"], kw["temperature"], kw["top_p"], kw["seed"], kw["stop"])
        if self.lock.locked():
            raise BusyError("busy")
        if kw.get("model") and kw["model"] != self.model_id:
            raise RequestError("model mismatch")
        self.last_prompt = prompt
        return {"model": self.model_id, "text": "test response", "finish_reason": "stop",
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
                "latency_seconds": .01}

    def _unload(self):
        self.model_id, self.phase = None, "unloaded"

    def close(self):
        self._unload()

    def record(self, *a, **kw):
        pass


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.client = TestClient(create_app(self.engine, api_key="test-key"))
        self.headers = {"Authorization": "Bearer test-key"}

    def post(self, path, value):
        return self.client.post(path, json=value, headers=self.headers)

    def test_authentication(self):
        self.assertEqual(self.client.get('/healthz').status_code, 200)
        self.assertEqual(self.client.get('/v1/models').status_code, 401)
        self.assertEqual(self.client.get('/readyz', headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get('/readyz', headers={'Authorization': 'Bearer wrong-key'}).status_code, 401)
        self.assertEqual(self.client.get('/readyz', headers={'Authorization': 'Basic test-key'}).status_code, 401)

    def test_swagger_uses_bearer_security_instead_of_reserved_header_parameter(self):
        schema = self.client.get('/openapi.json').json()
        scheme = schema['components']['securitySchemes']['ModelAPIKey']
        self.assertEqual(scheme['type'], 'http')
        self.assertEqual(scheme['scheme'], 'bearer')
        operation = schema['paths']['/generate']['post']
        self.assertEqual(operation['security'], [{'ModelAPIKey': []}])
        self.assertFalse(any(p['name'].lower() == 'authorization' for p in operation.get('parameters', [])))

    def test_instruction_and_raw_prompt_are_distinct(self):
        response = self.post('/generate', {"instruction": "summarize", "input": "text"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.engine.last_prompt, instruction_prompt("summarize", "text"))
        self.post('/generate', {"prompt": "A raw continuation prefix"})
        self.assertEqual(self.engine.last_prompt, "A raw continuation prefix")
        self.assertEqual(self.post('/generate', {"prompt": "x", "instruction": "y"}).status_code, 422)
        self.assertEqual(self.post('/generate', {"prompt": "x", "input": "y"}).status_code, 422)

    def test_openai_nonstreaming_shapes(self):
        r = self.post('/v1/chat/completions', {"model": "moe-sft", "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['object'], 'chat.completion')
        self.assertEqual(r.json()['choices'][0]['message']['role'], 'assistant')
        r = self.post('/v1/completions', {"model": "moe-sft", "prompt": "hello"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['usage']['total_tokens'], 10)

    def test_reject_unsupported_streaming_and_tools(self):
        base = {"model": "moe-sft", "messages": [{"role": "user", "content": "hello"}]}
        self.assertEqual(self.post('/v1/chat/completions', {**base, "stream": True}).status_code, 422)
        self.assertEqual(self.post('/v1/chat/completions', {**base, "tools": []}).status_code, 422)

    def test_validation_and_context_limits(self):
        self.assertEqual(self.post('/generate', {"prompt": "x", "max_new_tokens": 1025}).status_code, 422)
        self.assertEqual(self.post('/generate', {"prompt": "x", "top_p": 0}).status_code, 422)
        self.assertEqual(self.post('/generate', {"prompt": "x", "stop": [""]}).status_code, 400)
        self.assertEqual(self.post('/generate', {"prompt": "x", "model": "dense-base"}).status_code, 400)

    def test_busy_load_and_unload(self):
        with self.engine.lock:
            self.assertEqual(self.post('/generate', {"prompt": "x"}).status_code, 429)
            self.assertEqual(self.post('/admin/load', {"model": "moe-dpo"}).status_code, 429)
            self.assertEqual(self.post('/admin/unload', {}).status_code, 429)
        self.assertEqual(self.post('/admin/load', {"model": "moe-dpo"}).json()['model'], 'moe-dpo')
        self.assertEqual(self.post('/admin/load', {"model": "arbitrary-path"}).status_code, 400)
        self.assertEqual(self.post('/admin/unload', {}).status_code, 200)
        self.assertEqual(self.client.get('/readyz', headers=self.headers).status_code, 503)

    def test_chat_history_preserves_alpaca(self):
        messages = [{"role": "system", "content": "brief"}, {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"}, {"role": "user", "content": "three"}]
        text = chat_prompt(messages)
        self.assertIn('### Instruction:\nbrief\n\nthree', text)
        self.assertIn('USER: one\nASSISTANT: two', text)
        with self.assertRaises(RequestError):
            chat_prompt([{"role": "assistant", "content": "a"}])
        with self.assertRaises(RequestError):
            chat_prompt([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])

    def test_registry_dpo_is_original_adapter_path(self):
        spec = registry()['moe-dpo']
        self.assertTrue(spec['path'].endswith('merged-v2'))
        self.assertEqual(Path(spec['adapter']).parts[-2:], ('dpo-hh-20261020', 'adapter.pt'))
        self.assertEqual(trim_stop('abENDcd', ['END']), ('ab', True))

    def test_allocator_warmup_patch_is_scoped_and_restored_on_failure(self):
        def original(*args, **kwargs):
            raise AssertionError('large preallocation should be disabled')
        module = SimpleNamespace(caching_allocator_warmup=original)
        with self.assertRaises(RuntimeError):
            with without_allocator_warmup(module):
                self.assertIsNone(module.caching_allocator_warmup(None, {}, None))
                raise RuntimeError('simulated model-loading failure')
        self.assertIs(module.caching_allocator_warmup, original)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        # Self-contained CPU fixture; never read or reuse a real GPU reservation.
        self.data = {
            'container': 'cpu-unit-test-only',
            'starts_at': '2026-10-28T16:00:00+08:00',
            'expires_at': '2026-10-28T21:00:00+08:00',
            'shutdown_margin_seconds': 300,
            'authorized_physical_gpus': [2, 3, 4, 5],
            'active_physical_gpus': [2, 3],
            'container_device_order': [2, 3, 4, 5],
            'cuda_visible_devices': '0,1',
        }
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'lease.json'
        self.now = datetime.fromisoformat('2026-10-28T17:00:00+08:00').timestamp()

    def tearDown(self):
        self.temp.cleanup()

    def check(self, data=None, now=None):
        self.path.write_text(json.dumps(self.data if data is None else data))
        return gpu_lease.lease_details(self.path, self.now if now is None else now, verify_devices=False)

    def test_active_mapping_and_deadline(self):
        data, deadline = self.check()
        self.assertEqual(data['active_physical_gpus'], [2, 3])
        self.assertEqual(deadline, datetime.fromisoformat('2026-10-28T20:55:00+08:00').timestamp())

    def test_stale_and_early_leases_refused(self):
        for timestamp in ['2026-10-28T15:59:00+08:00', '2026-10-28T20:55:00+08:00', '2026-10-29T17:00:00+08:00']:
            with self.assertRaises(ValueError):
                self.check(now=datetime.fromisoformat(timestamp).timestamp())

    def test_wrong_gpu_mask_and_unauthorized_devices_refused(self):
        for changes in [{"cuda_visible_devices": "2,3"}, {"active_physical_gpus": [8, 9]},
                        {"cuda_visible_devices": "-1,0"}, {"container_device_order": [2, 3, 4, 8]}]:
            with self.assertRaises(ValueError):
                self.check({**self.data, **changes})

    @unittest.skipUnless(os.name == 'posix', 'flock requires Linux')
    def test_separate_entry_points_cannot_overlap_on_the_same_gpu(self):
        directory = Path(self.temp.name) / 'locks'
        with gpu_lease.lease_locks(self.data, directory):
            with self.assertRaises(RuntimeError):
                with gpu_lease.lease_locks(self.data, directory):
                    self.fail('overlapping inference worker was permitted')
        with gpu_lease.lease_locks(self.data, directory):
            pass  # locks are released after the previous worker exits

    @unittest.skipUnless(os.name == 'posix', 'process-group watchdog requires Linux')
    def test_watchdog_kills_worker_even_during_startup(self):
        script = Path(self.temp.name) / 'worker.py'
        script.write_text('import time\ntime.sleep(90)\n')
        old_handlers = {s: signal.getsignal(s) for s in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]}
        began = time.monotonic()
        try:
            with patch.object(gpu_lease, 'lease_details', return_value=(self.data, time.time() + .2)):
                result = gpu_lease.supervise(script, self.path, [], Path(self.temp.name) / 'locks')
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        self.assertNotEqual(result, 0)
        self.assertLess(time.monotonic() - began, 5)


if __name__ == '__main__':
    unittest.main()
