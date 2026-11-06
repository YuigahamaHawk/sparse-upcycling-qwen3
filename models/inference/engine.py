"""Verified HF/Corex inference path, including custom expert LoRA AND routers.

Torch is imported only after the GPU worker lease and mask have been checked.
"""
import gc
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from models.paths import PROJECT_ROOT as ROOT
from models.runtime.gpu_lease import require_worker_lease
from models.inference.common import (BusyError, LeaseExpired, RequestError, check_parameters,
                                     registry, trim_stop, without_allocator_warmup)


class InferenceEngine:
    def __init__(self, registry_path, *, context_limit=2048, output_cap=512, memory="28GiB", status_path=None):
        self.lease, self.deadline = require_worker_lease()
        self.specs = registry(registry_path)
        self.context_limit, self.output_cap, self.memory = context_limit, output_cap, memory
        self.status_path = Path(status_path) if status_path else None
        self.lock = threading.Lock()
        self.model = self.tokenizer = self.model_id = None
        self.phase, self.last_error = "unloaded", None

    def check_lease(self):
        if time.time() >= self.deadline:
            raise LeaseExpired("GPU lease reached shutdown margin")

    def status(self):
        return {"phase": self.phase, "model": self.model_id, "pid": os.getpid(),
                "physical_gpus": self.lease["active_physical_gpus"],
                "stop_at": datetime.fromtimestamp(self.deadline, tz=datetime.fromisoformat(self.lease["expires_at"]).tzinfo).isoformat(),
                "stop_at_epoch": self.deadline, "context_limit": self.context_limit,
                "output_cap": self.output_cap, "last_error": self.last_error,
                "updated_at_epoch": time.time()}

    def record(self, event, **fields):
        value = {"event": event, **self.status(), **fields}
        print(json.dumps(value, ensure_ascii=False), flush=True)
        if self.status_path:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.status_path.with_suffix(".tmp")
            temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.status_path)

    def _unload(self):
        import torch
        self.model = self.tokenizer = self.model_id = None
        gc.collect()
        torch.cuda.empty_cache()
        self.phase = "unloaded"

    def close(self):
        with self.lock:
            self._unload()
            self.record("unloaded")

    def load(self, model_id):
        self.check_lease()
        if model_id not in self.specs:
            raise RequestError(f"unknown model: {model_id}")
        if not self.lock.acquire(blocking=False):
            raise BusyError("another load or generation is running")
        try:
            if self.model_id == model_id and self.phase == "ready":
                return self.status()
            spec = self.specs[model_id]
            model_path = Path(spec["path"])
            if not (model_path / "config.json").is_file():
                raise RequestError(f"missing checkpoint config: {model_path}")
            if spec.get("adapter") and not Path(spec["adapter"]).is_file():
                raise RequestError("adapter file is missing")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, modeling_utils
            from models.components.moe import install_qwen3_patches
            install_qwen3_patches()
            count = torch.cuda.device_count()
            if count != len(self.lease["active_physical_gpus"]):
                raise RuntimeError(f"GPU visibility mismatch: {count} devices")
            needed = int(spec["min_gpus"])
            if count < needed:
                raise RequestError(f"{model_id} needs at least {needed} GPUs at BF16")
            self._unload()
            self.phase, self.last_error = "loading", None
            self.record("loading", target=model_id)
            started = time.monotonic()
            with without_allocator_warmup(modeling_utils):
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path), torch_dtype=torch.bfloat16, local_files_only=True,
                    device_map="auto", max_memory={i: self.memory for i in range(needed)},
                    low_cpu_mem_usage=True, attn_implementation="eager")
            mapping = self.model.hf_device_map
            if any(str(v) in ("cpu", "disk") for v in mapping.values()):
                raise RuntimeError("checkpoint spilled to CPU/disk; refusing silently degraded deployment")
            adapter_info = self._load_adapter(spec, torch) if spec.get("adapter") else None
            self.model.eval().requires_grad_(False)
            self.model.config.use_cache = True
            if self.model.config.model_type == "qwen3_moe":
                # Init checkpoints enable router aux loss for training. With KV
                # cache its full attention mask does not match new-token gates.
                # Generation does not need aux loss or returned router logits.
                self.model.config.output_router_logits = False
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True)
            self.model_id, self.phase = model_id, "ready"
            self.check_lease()
            self.record("ready", load_seconds=round(time.monotonic()-started, 3),
                        device_map=mapping, adapter=adapter_info,
                        gpu_names=[torch.cuda.get_device_name(i) for i in range(needed)])
            return self.status()
        except Exception as exc:
            if self.phase == "loading":
                self._unload()
                self.phase, self.last_error = "failed", str(exc)
                self.record("load_failed")
            raise
        finally:
            self.lock.release()

    def _load_adapter(self, spec, torch):
        bundle = torch.load(spec["adapter"], map_location="cpu", weights_only=False)
        state, rank = bundle["state"], int(bundle["args"]["rank"])
        if rank < 1:
            raise RuntimeError("invalid adapter rank")
        base = Path(bundle["args"]["model"])
        if not base.is_absolute():
            base = ROOT / base
        if base.resolve() != Path(spec["path"]).resolve():
            raise RuntimeError("DPO adapter base differs from registered checkpoint")
        expected = set()
        for i, layer in enumerate(self.model.model.layers):
            if not hasattr(layer.mlp, "experts"):
                continue
            prefix = f"model.layers.{i}.mlp"
            expected.add(f"{prefix}.gate.weight")
            for j, expert in enumerate(layer.mlp.experts):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    for suffix in ("A", "B"):
                        expected.add(f"{prefix}.experts.{j}.{name}.{suffix}")
        if set(state) != expected:
            raise RuntimeError(f"adapter key mismatch: missing={sorted(expected-set(state))[:5]}, "
                               f"unexpected={sorted(set(state)-expected)[:5]}")
        if any(not torch.isfinite(t).all().item() for t in state.values()):
            raise RuntimeError("adapter contains nonfinite weights")
        from models.components.lora import inject_expert_lora
        inject_expert_lora(self.model, rank)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected or expected.intersection(missing):
            raise RuntimeError("adapter or router tensors were not loaded completely")
        digest = hashlib.sha256(Path(spec["adapter"]).read_bytes()).hexdigest()
        return {"path": spec["adapter"], "rank": rank, "tensors": len(state), "sha256": digest}

    def generate(self, prompt, *, model=None, max_new_tokens=256, temperature=0.0, top_p=1.0, seed=42, stop=None):
        self.check_lease()
        stops = stop or []
        check_parameters(max_new_tokens, temperature, top_p, seed, stops, self.output_cap)
        if not isinstance(prompt, str) or not prompt or len(prompt) > 100000:
            raise RequestError("prompt must contain 1..100000 characters")
        if not self.lock.acquire(blocking=False):
            raise BusyError("model is busy; retry after the current request finishes")
        try:
            if self.phase != "ready":
                raise RequestError("no model is ready; load a registered model first")
            if model and model != self.model_id:
                raise RequestError(f"loaded model is {self.model_id}; use /admin/load to switch sequentially")
            import torch
            from transformers import StoppingCriteria, StoppingCriteriaList
            encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_tokens = encoded.input_ids.shape[1]
            context = min(self.context_limit, int(self.model.config.max_position_embeddings))
            if input_tokens + max_new_tokens > context:
                raise RequestError(f"input ({input_tokens}) + requested output ({max_new_tokens}) exceeds context limit {context}; input was not truncated")
            tokenizer, deadline = self.tokenizer, self.deadline

            class RequestStop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    if time.time() >= deadline:
                        return True
                    return bool(stops and any(s in tokenizer.decode(input_ids[0, input_tokens:], skip_special_tokens=True) for s in stops))

            device = self.model.model.embed_tokens.weight.device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            torch.manual_seed(seed)
            kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                          use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
                          stopping_criteria=StoppingCriteriaList([RequestStop()]))
            if self.model.config.model_type == "qwen3_moe":
                kwargs["output_router_logits"] = False
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=top_p)
            started = time.monotonic()
            with torch.inference_mode():
                output = self.model.generate(**encoded, **kwargs)
            new_ids = output[0, input_tokens:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True)
            text, stopped = trim_stop(raw, stops)
            output_tokens = len(new_ids)
            seconds = time.monotonic() - started
            self.check_lease()
            eos_ids = self.model.generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            eos = bool(output_tokens and new_ids[-1].item() in eos_ids)
            result = {"model": self.model_id, "text": text,
                      "finish_reason": "stop" if stopped or eos else "length",
                      "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens},
                      "latency_seconds": round(seconds, 4),
                      "completion_tokens_per_second": round(output_tokens / max(seconds, 1e-6), 3)}
            self.record("generation", usage=result["usage"], latency_seconds=result["latency_seconds"])
            return result
        except Exception as exc:
            import torch
            if isinstance(exc, torch.OutOfMemoryError):
                torch.cuda.empty_cache()
                self.last_error = "GPU out of memory; reduce input/output length"
                self.record("oom")
            raise
        finally:
            self.lock.release()
