"""Checkpoint-specific configuration for the LeRobot Diffusion Policy adapter."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiffusionCheckpointConfig:
    """Normalized inference-relevant view of a Diffusion checkpoint config."""

    checkpoint_path: Path
    type: str
    device: str
    use_amp: bool
    n_obs_steps: int
    horizon: int
    n_action_steps: int
    input_features: dict[str, dict[str, Any]]
    output_features: dict[str, dict[str, Any]]
    normalization_mapping: dict[str, str]
    vision_backbone: str | None
    resize_shape: tuple[int, ...] | None
    crop_ratio: float | None
    crop_shape: tuple[int, ...] | None
    crop_is_random: bool
    pretrained_backbone_weights: str | None
    use_separate_rgb_encoder_per_camera: bool
    down_dims: tuple[int, ...]
    kernel_size: int
    n_groups: int
    diffusion_step_embed_dim: int
    use_film_scale_modulation: bool
    gradient_checkpointing: bool
    noise_scheduler_type: str
    num_train_timesteps: int
    beta_schedule: str
    beta_start: float
    beta_end: float
    prediction_type: str
    clip_sample: bool
    clip_sample_range: float
    num_inference_steps: int | None
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device_override: str | None = None,
    ) -> "DiffusionCheckpointConfig":
        root = Path(checkpoint_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Diffusion checkpoint not found: {root}")
        if not root.is_dir():
            raise ValueError(f"Diffusion checkpoint must be a directory: {root}")

        config_path = root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing Diffusion checkpoint config: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if str(raw.get("type", "")).lower() != "diffusion":
            raise ValueError(
                f"Expected Diffusion checkpoint config type='diffusion', got {raw.get('type')!r}"
            )

        inputs = dict(raw.get("input_features", {}))
        outputs = dict(raw.get("output_features", {}))

        def shape_of(spec: dict[str, Any]) -> tuple[int, ...]:
            shape = spec.get("shape")
            if not shape:
                raise ValueError(f"Feature has no shape: {spec}")
            return tuple(int(v) for v in shape)

        state_shape = shape_of(inputs["observation.state"])
        action_shape = shape_of(outputs["action"])

        if len(state_shape) != 1:
            raise ValueError(f"Expected 1D observation.state, got {state_shape}")
        if len(action_shape) != 1:
            raise ValueError(f"Expected 1D action, got {action_shape}")

        return cls(
            checkpoint_path=root,
            type=str(raw["type"]),
            device=str(device_override or raw.get("device", "cuda")),
            use_amp=bool(raw.get("use_amp", False)),
            n_obs_steps=int(raw["n_obs_steps"]),
            horizon=int(raw["horizon"]),
            n_action_steps=int(raw["n_action_steps"]),
            input_features=inputs,
            output_features=outputs,
            normalization_mapping={
                str(k): str(v).upper()
                for k, v in dict(raw.get("normalization_mapping", {})).items()
            },
            vision_backbone=raw.get("vision_backbone"),
            resize_shape=(
                tuple(int(v) for v in raw["resize_shape"])
                if raw.get("resize_shape") is not None
                else None
            ),
            crop_ratio=(
                float(raw["crop_ratio"])
                if raw.get("crop_ratio") is not None
                else None
            ),
            crop_shape=(
                tuple(int(v) for v in raw["crop_shape"])
                if raw.get("crop_shape") is not None
                else None
            ),
            crop_is_random=bool(raw.get("crop_is_random", True)),
            pretrained_backbone_weights=raw.get("pretrained_backbone_weights"),
            use_separate_rgb_encoder_per_camera=bool(
                raw.get("use_separate_rgb_encoder_per_camera", True)
            ),
            down_dims=tuple(int(v) for v in raw.get("down_dims", (512, 1024, 2048))),
            kernel_size=int(raw.get("kernel_size", 5)),
            n_groups=int(raw.get("n_groups", 8)),
            diffusion_step_embed_dim=int(raw.get("diffusion_step_embed_dim", 128)),
            use_film_scale_modulation=bool(raw.get("use_film_scale_modulation", True)),
            gradient_checkpointing=bool(raw.get("gradient_checkpointing", False)),
            noise_scheduler_type=str(raw.get("noise_scheduler_type", "DDPM")),
            num_train_timesteps=int(raw.get("num_train_timesteps", 100)),
            beta_schedule=str(raw.get("beta_schedule", "squaredcos_cap_v2")),
            beta_start=float(raw.get("beta_start", 1e-4)),
            beta_end=float(raw.get("beta_end", 0.02)),
            prediction_type=str(raw.get("prediction_type", "epsilon")),
            clip_sample=bool(raw.get("clip_sample", True)),
            clip_sample_range=float(raw.get("clip_sample_range", 1.0)),
            num_inference_steps=(
                int(raw["num_inference_steps"])
                if raw.get("num_inference_steps") is not None
                else None
            ),
            raw=raw,
        )

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(
            key
            for key, spec in self.input_features.items()
            if str(spec.get("type", "")).upper() == "VISUAL"
        )

    @property
    def state_keys(self) -> tuple[str, ...]:
        return tuple(
            key
            for key, spec in self.input_features.items()
            if str(spec.get("type", "")).upper() == "STATE"
        )

    @property
    def state_dim(self) -> int:
        return int(self.input_features["observation.state"]["shape"][0])

    @property
    def leader_state_dim(self) -> int | None:
        feature = self.input_features.get("observation.leader_state")
        return int(feature["shape"][0]) if feature is not None else None

    @property
    def action_dim(self) -> int:
        return int(self.output_features["action"]["shape"][0])

    def validate(self) -> None:
        """Validate the exact interface expected by this checkpoint adapter."""
        expected_inputs = {
            "observation.state": ("STATE", (16,)),
            "observation.leader_state": ("STATE", (16,)),
            "observation.images.world": ("VISUAL", (3, 240, 320)),
            "observation.images.wrist_right": ("VISUAL", (3, 240, 320)),
            "observation.images.wrist_left": ("VISUAL", (3, 240, 320)),
        }
        expected_output = {"action": ("ACTION", (16,))}

        if self.n_obs_steps < 1 or self.horizon < 1 or self.n_action_steps < 1:
            raise ValueError("Observation/action horizons must be positive")
        if self.n_obs_steps - 1 + self.n_action_steps > self.horizon:
            raise ValueError("Diffusion horizon must cover observation history and returned action steps")

        actual_input_keys = set(self.input_features)
        if actual_input_keys != set(expected_inputs):
            raise ValueError(
                "Diffusion input feature mismatch. "
                f"Missing={sorted(set(expected_inputs) - actual_input_keys)}; "
                f"extra={sorted(actual_input_keys - set(expected_inputs))}."
            )

        for key, (expected_type, expected_shape) in expected_inputs.items():
            spec = self.input_features[key]
            actual_type = str(spec.get("type", "")).upper()
            actual_shape = tuple(int(v) for v in spec.get("shape", []))
            if (actual_type, actual_shape) != (expected_type, expected_shape):
                raise ValueError(
                    f"Feature '{key}' mismatch: expected {expected_type}/{expected_shape}; "
                    f"got {actual_type}/{actual_shape}."
                )

        if set(self.output_features) != set(expected_output):
            raise ValueError(
                f"Diffusion outputs must be exactly ['action']; got {sorted(self.output_features)}"
            )
        out = self.output_features["action"]
        out_pair = (str(out.get("type", "")).upper(), tuple(int(v) for v in out.get("shape", [])))
        if out_pair != expected_output["action"]:
            raise ValueError(
                "Diffusion action output mismatch: expected ACTION/(16,), "
                f"got {out_pair[0]}/{out_pair[1]}"
            )

        expected_norm = {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
        for key, value in expected_norm.items():
            if self.normalization_mapping.get(key) != value:
                raise ValueError(
                    f"Normalization mismatch for {key}: expected {value}, "
                    f"got {self.normalization_mapping.get(key)!r}"
                )

        # The native LeRobot configuration validates architecture and scheduler
        # settings; they are not fixed to one training run in this adapter.
        if self.num_train_timesteps < 1:
            raise ValueError("num_train_timesteps must be positive")

        # null is intentional in the supplied checkpoint config: the policy/runtime
        # chooses the inference-step count. Do not replace it with the training value.
        if self.num_inference_steps is not None and self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive when set")
