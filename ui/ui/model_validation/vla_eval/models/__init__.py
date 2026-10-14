"""Model adapters."""

# Import adapters here so registry decorators execute on package import.
from vla_eval.models.act import ACTAdapter  # noqa: F401
from vla_eval.models.diffusion import DiffusionAdapter  # noqa: F401
from vla_eval.models.smolvla import SmolVLAAdapter  # noqa: F401

__all__ = [
    "ACTAdapter",
    "DiffusionAdapter",
    "SmolVLAAdapter",
]
    