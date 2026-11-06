"""CLI entry point; implementation: models.evaluation.language_model.LanguageModelEvaluator."""
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
    p.add_argument("--adapter", type=Path)
    p.add_argument("--dense", action="store_true")
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-memory", default="28GiB")
    p.add_argument("--save-probe", type=Path)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    from models.evaluation.language_model import LanguageModelEvaluator
    return LanguageModelEvaluator(args).run()


def __getattr__(name):
    # Historical helper imports resolve to the implementation, without making
    # any model implementation depend on a CLI script.
    from importlib import import_module
    return getattr(import_module("models.evaluation.language_model"), name)


if __name__ == "__main__":
    main()
