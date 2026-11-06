"""CPU-only registry, prompting and request limits shared by HTTP and CLI."""
import json
from contextlib import contextmanager
from pathlib import Path
from models.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
DEFAULT_REGISTRY = ROOT / "configs/serving/models.json"
TEMPLATE = ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n{input_part}### Response:\n")


class RequestError(ValueError):
    pass


class BusyError(RuntimeError):
    pass


class LeaseExpired(RuntimeError):
    pass


@contextmanager
def without_allocator_warmup(modeling_utils):
    """Avoid Corex's large contiguous preallocation on repeated model switches.

    Only the HF loading optimization is disabled; tensors, dtype and inference
    math are unchanged. Caller holds the exclusive model-loading lock.
    """
    original = modeling_utils.caching_allocator_warmup
    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    try:
        yield
    finally:
        modeling_utils.caching_allocator_warmup = original


def registry(path=DEFAULT_REGISTRY):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    root = Path(data["project_root"])
    result = {}
    for name, spec in data["models"].items():
        spec = dict(spec)
        for key in ("path", "adapter"):
            if key in spec:
                value = Path(spec[key])
                spec[key] = str(value if value.is_absolute() else root / value)
        result[name] = spec
    return result


def instruction_prompt(instruction, input_text=""):
    if not isinstance(instruction, str) or not instruction.strip():
        raise RequestError("instruction must be a nonempty string")
    part = f"### Input:\n{input_text}\n\n" if input_text else ""
    return TEMPLATE.format(instruction=instruction, input_part=part)


def chat_prompt(messages):
    """Preserve Alpaca serialization; do not invent a native Qwen chat template."""
    if not messages or messages[-1]["role"] != "user":
        raise RequestError("messages must end with a user message")
    systems, history = [], []
    expected = "user"
    for i, msg in enumerate(messages[:-1]):
        role, content = msg["role"], msg["content"]
        if role == "system":
            if i != 0:
                raise RequestError("system is allowed only at the beginning")
            systems.append(content)
        elif role == expected:
            history.append(f"{role.upper()}: {content}")
            expected = "assistant" if role == "user" else "user"
        else:
            raise RequestError("user and assistant messages must alternate")
    if expected != "user":
        raise RequestError("user and assistant messages must alternate")
    instruction = "\n\n".join(systems + [messages[-1]["content"]])
    context = "Conversation history:\n" + "\n".join(history) if history else ""
    return instruction_prompt(instruction, context)


def check_parameters(max_new_tokens, temperature, top_p, seed, stops, cap=512):
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= cap:
        raise RequestError(f"max_new_tokens must be between 1 and {cap}")
    if not 0 <= temperature <= 2 or not 0 < top_p <= 1:
        raise RequestError("temperature must be 0..2 and top_p must be >0..1")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise RequestError("seed must be a uint32 integer")
    if len(stops) > 8 or any(not isinstance(s, str) or not s or len(s) > 256 for s in stops):
        raise RequestError("stop must contain at most 8 nonempty strings of at most 256 characters")


def trim_stop(text, stops):
    positions = [text.find(s) for s in stops if s in text]
    return (text[:min(positions)], True) if positions else (text, False)
