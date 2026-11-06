"""CLI entry point; implementation: models.trainers.sft.SFTTrainer."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--eval-data", type=Path, required=True)
    p.add_argument("--wiki-eval", type=Path, required=True,
                   help="JSONL text corpus for the CLM regression check")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-len", type=int, default=640)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--max-memory", default="28GiB")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    from models.trainers.sft import SFTTrainer
    return SFTTrainer(args).run()


def __getattr__(name):
    # Historical helper imports resolve to the implementation, without making
    # any model implementation depend on a CLI script.
    from importlib import import_module
    return getattr(import_module("models.trainers.sft"), name)


if __name__ == "__main__":
    main()
