"""CLI entry point; implementation: models.evaluation.merged_weights.MergedWeightAuditor."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--merged", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from models.evaluation.merged_weights import MergedWeightAuditor
    return MergedWeightAuditor(args).run()


if __name__ == "__main__":
    main()
