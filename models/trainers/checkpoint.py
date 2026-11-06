"""Atomic writes shared by all trainers; legacy dictionary formats are kept."""
import json
import os


def atomic_save(obj, path):
    import torch
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def atomic_json(obj, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2))
    os.replace(temp, path)
