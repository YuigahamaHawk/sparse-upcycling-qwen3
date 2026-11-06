"""Compatibility import; canonical implementation: models.runtime.gpu_lease."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from importlib import import_module
sys.modules[__name__] = import_module("models.runtime.gpu_lease")
