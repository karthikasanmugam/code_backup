"""
Model registry. This is the ONE place in the codebase allowed to switch on
model type by name — the runner, metrics, and reporting code always talk to
a `BaseVLA` instance and never ask "is this ACT or Diffusion".

Adapters register themselves with the @register_model("name") decorator on
import, so adding a new model family means adding one file, not editing this
one.
"""

from __future__ import annotations

from typing import Dict, Type

from vla_eval.core.base_vla import BaseVLA

_REGISTRY: Dict[str, Type[BaseVLA]] = {}


def register_model(name: str):
    """Class decorator: @register_model("act") class ACTAdapter(BaseVLA): ..."""

    def _decorator(cls: Type[BaseVLA]) -> Type[BaseVLA]:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(
                f"Model type '{key}' is already registered to {_REGISTRY[key].__name__}; "
                f"refusing to overwrite with {cls.__name__}"
            )
        _REGISTRY[key] = cls
        return cls

    return _decorator


def create_model(model_type: str, **kwargs) -> BaseVLA:
    """Instantiate (but do not load()) the adapter registered for model_type.

    kwargs are passed through to the adapter's __init__ for anything it
    needs before load() — e.g. action_dim for a mock/test adapter.
    """
    key = model_type.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered — did you import the adapter module?)"
        raise KeyError(f"Unknown model_type '{model_type}'. Available: {available}")
    return _REGISTRY[key](**kwargs)


def available_models() -> list:
    return sorted(_REGISTRY)