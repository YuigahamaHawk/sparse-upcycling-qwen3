"""CLI entry point; implementation: models.evaluation.diagnostics.ForwardDiagnostics."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dense", action="store_true")
    p.add_argument("--data")
    p.add_argument("--lora", action="store_true")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    from models.evaluation.diagnostics import ForwardDiagnostics
    return ForwardDiagnostics(args).run()


def __getattr__(name):
    # Historical helper imports resolve to the implementation, without making
    # any model implementation depend on a CLI script.
    from importlib import import_module
    return getattr(import_module("models.evaluation.diagnostics"), name)


if __name__ == "__main__":
    main()
