"""Shared trainer and stage implementations, without eager model imports."""
from importlib import import_module


_EXPORTS = {
    "BasicTrainer": (".basic_trainer", "BasicTrainer"),
    "SmokeCPTTrainer": (".smoke_cpt", "SmokeCPTTrainer"),
    "CPTTrainer": (".cpt", "CPTTrainer"),
    "SFTTrainer": (".sft", "SFTTrainer"),
    "DPOTrainer": (".dpo", "DPOTrainer"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    return getattr(import_module(module, __name__), symbol)
