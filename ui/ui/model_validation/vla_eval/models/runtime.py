"""Inference-only settings; never rewrite a saved checkpoint or training config."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch

PROFILES = ("optimized", "checkpoint", "fast")


@dataclass(frozen=True)
class InferenceRuntime:
    profile: str
    device: torch.device
    amp_dtype: torch.dtype | None

    @classmethod
    def resolve(cls, profile: str, device: torch.device) -> "InferenceRuntime":
        if profile not in PROFILES:
            raise ValueError(f"inference_profile must be one of {PROFILES}")
        dtype = None
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Select CPU or check the GPU environment.")
            # Query the selected device, including cuda:N, rather than GPU 0.
            major, _ = torch.cuda.get_device_capability(device)
            if profile != "checkpoint":
                if major >= 8:
                    dtype = torch.bfloat16
                elif major >= 7:
                    dtype = torch.float16
        return cls(profile, device, dtype)

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def metadata(self) -> dict:
        return {
            "profile": self.profile,
            "device": str(self.device),
            "device_name": (
                torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            ),
            "precision": str(self.amp_dtype or torch.float32).removeprefix("torch."),
        }


@contextmanager
def prediction_seed(seed: int, device: str):
    """Repeat stochastic predictions without advancing the caller's RNG state."""
    selected = torch.device(device)
    devices = []
    if selected.type == "cuda":
        devices = [selected.index if selected.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        yield


def load_evaluation_model(model_type: str, checkpoint: str, device: str, profile: str = "checkpoint"):
    """Load once per job. ACT retains its existing inference behavior."""
    from vla_eval.core.registry import create_model

    if profile not in PROFILES:
        raise ValueError(f"inference_profile must be one of {PROFILES}")
    kwargs = {"device": device}
    if model_type in {"diffusion", "smolvla"}:
        kwargs["inference_profile"] = profile
    model = create_model(model_type, **kwargs)
    model.load(checkpoint)
    return model
