"""Compatibility import; canonical implementation: models.inference.engine."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from importlib import import_module
sys.modules[__name__] = import_module("models.inference.engine")
