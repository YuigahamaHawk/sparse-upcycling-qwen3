"""CLI entry point; implementation: models.conversion.merge.AdapterMerger."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("cpt", "sft", "dpo"), default="cpt")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    from models.conversion.merge import AdapterMerger
    return AdapterMerger(args).run()


def __getattr__(name):
    # Historical helper imports resolve to the implementation, without making
    # any model implementation depend on a CLI script.
    from importlib import import_module
    return getattr(import_module("models.conversion.merge"), name)


if __name__ == "__main__":
    main()
