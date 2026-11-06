"""CLI entry point; implementation: models.trainers.smoke_cpt.SmokeCPTTrainer."""
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-memory", default="28GiB")
    p.add_argument("--seed", type=int, default=42)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    from models.trainers.smoke_cpt import SmokeCPTTrainer
    return SmokeCPTTrainer(args).run()


def __getattr__(name):
    # Historical helper imports resolve to the implementation, without making
    # any model implementation depend on a CLI script.
    from importlib import import_module
    return getattr(import_module("models.trainers.smoke_cpt"), name)


if __name__ == "__main__":
    main()
