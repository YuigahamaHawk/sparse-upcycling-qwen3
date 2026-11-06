"""CLI entry point; implementation: models.conversion.upcycle.Qwen3Upcycler."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("/home/wurendi/models/Qwen-3-8B"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--layers", default="alternating")
    p.add_argument("--router-std", type=float, default=0.02)
    p.add_argument("--normalize-topk", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    from models.conversion.upcycle import Qwen3Upcycler
    return Qwen3Upcycler(args).run()


def __getattr__(name):
    # Historical helper imports resolve to the implementation, without making
    # any model implementation depend on a CLI script.
    from importlib import import_module
    return getattr(import_module("models.conversion.upcycle"), name)


if __name__ == "__main__":
    main()
