"""Shared source-tree locations; importing this module never initializes Torch."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
