"""Supervised inference service CLI; HTTP API lives in models.inference.api."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.runtime.gpu_lease import require_worker_lease, supervise
from models.inference.common import DEFAULT_REGISTRY


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--lease", type=Path, required=True)
    p.add_argument("--model", default="moe-sft")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18082)
    p.add_argument("--context-limit", type=int, default=2048)
    p.add_argument("--output-cap", type=int, default=512)
    p.add_argument("--max-memory", default="28GiB")
    p.add_argument("--status", type=Path, default=ROOT / "runs/serving/status.json")
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--lease-worker", action="store_true", help=argparse.SUPPRESS)
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.output_cap <= 512 or args.context_limit < args.output_cap + 1:
        parser.error("output-cap must be 1..512 and context-limit must exceed it")
    if args.host not in ("127.0.0.1", "::1", "localhost") and not args.api_key_file:
        parser.error("binding beyond localhost requires --api-key-file")
    if not args.lease_worker:
        raise SystemExit(supervise(Path(__file__), args.lease, sys.argv[1:]))
    require_worker_lease()
    from models.inference.engine import InferenceEngine
    from models.inference.api import create_app
    import uvicorn
    key = args.api_key_file.read_text().strip() if args.api_key_file else None
    if args.api_key_file and not key:
        parser.error("API key file is empty")
    engine = InferenceEngine(args.registry, context_limit=args.context_limit,
                             output_cap=args.output_cap, memory=args.max_memory, status_path=args.status)
    # Exactly one process owns the model. Never use reload or multiple workers.
    uvicorn.run(create_app(engine, args.model, key), host=args.host, port=args.port,
                workers=1, timeout_graceful_shutdown=10)

def __getattr__(name):
    from importlib import import_module
    return getattr(import_module("models.inference.api"), name)


if __name__ == "__main__":
    main()
