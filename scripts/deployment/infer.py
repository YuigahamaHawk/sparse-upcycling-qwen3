"""Standalone local-GPU inference, using the same loader and lease as HTTP.

JSONL rows: {"id": optional, "prompt": "..."} OR
{ "instruction": "...", "input": optional }. Output is flushed after each row.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.runtime.gpu_lease import require_worker_lease, supervise
from models.inference.common import DEFAULT_REGISTRY, check_parameters, instruction_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--lease", type=Path, required=True)
    p.add_argument("--model", default="moe-sft")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--instruction")
    source.add_argument("--input-jsonl", type=Path)
    p.add_argument("--input", default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context-limit", type=int, default=2048)
    p.add_argument("--max-memory", default="28GiB")
    p.add_argument("--lease-worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    try:
        check_parameters(args.max_new_tokens, args.temperature, args.top_p, args.seed, [])
    except ValueError as exc:
        p.error(str(exc))
    if args.context_limit <= args.max_new_tokens:
        p.error("context-limit must exceed max-new-tokens")
    if args.input and args.instruction is None:
        p.error("--input is allowed only with --instruction")
    if args.input_jsonl and not args.input_jsonl.is_file():
        p.error("input JSONL does not exist")
    if args.output.exists():
        p.error("output already exists; choose a new path to preserve previous results")
    if not args.lease_worker:
        raise SystemExit(supervise(Path(__file__), args.lease, sys.argv[1:]))
    require_worker_lease()
    from models.inference.engine import InferenceEngine
    engine = InferenceEngine(args.registry, context_limit=args.context_limit, memory=args.max_memory)
    if args.input_jsonl:
        rows = [json.loads(s) for s in args.input_jsonl.read_text(encoding="utf-8").splitlines() if s.strip()]
    else:
        rows = [{"prompt": args.prompt}] if args.prompt is not None else [{"instruction": args.instruction, "input": args.input}]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        engine.load(args.model)
        with args.output.open("x", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                if ("prompt" in row) == ("instruction" in row):
                    raise ValueError(f"row {index}: provide exactly one of prompt/instruction")
                prompt = row["prompt"] if "prompt" in row else instruction_prompt(row["instruction"], row.get("input", ""))
                result = engine.generate(prompt, max_new_tokens=args.max_new_tokens,
                                         temperature=args.temperature, top_p=args.top_p, seed=args.seed)
                result.update(id=row.get("id", index), prompt=prompt)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
