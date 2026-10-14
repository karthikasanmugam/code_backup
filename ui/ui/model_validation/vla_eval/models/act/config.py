"""ACT checkpoint configuration and compatibility validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EXPECTED_INPUTS = {
    "observation.state": ("STATE", (16,)),
    "observation.leader_state": ("STATE", (16,)),
    "observation.images.world": ("VISUAL", (3, 240, 320)),
    "observation.images.wrist_right": ("VISUAL", (3, 240, 320)),
    "observation.images.wrist_left": ("VISUAL", (3, 240, 320)),
}

EXPECTED_OUTPUT = {
    "action": ("ACTION", (16,)),
}


@dataclass
class ACTCheckpointConfig:
    """Normalized view of the ACT config stored with a checkpoint."""

    type: str
    n_obs_steps: int
    input_features: dict[str, dict[str, Any]]
    output_features: dict[str, dict[str, Any]]
    device: str = "cuda"
    use_amp: bool = False
    chunk_size: int = 50
    n_action_steps: int = 50
    normalization_mapping: dict[str, str] = field(default_factory=dict)
    vision_backbone: str | None = None
    pretrained_backbone_weights: str | None = None
    dim_model: int | None = None
    n_heads: int | None = None
    dim_feedforward: int | None = None
    n_encoder_layers: int | None = None
    n_decoder_layers: int | None = None
    use_vae: bool | None = None
    latent_dim: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ACTCheckpointConfig":
        if str(data.get("type", "")).lower() != "act":
            raise ValueError(
                f"Checkpoint config type must be 'act', got {data.get('type')!r}"
            )
        return cls(
            type=str(data["type"]),
            n_obs_steps=int(data["n_obs_steps"]),
            input_features=dict(data["input_features"]),
            output_features=dict(data["output_features"]),
            device=str(data.get("device", "cuda")),
            use_amp=bool(data.get("use_amp", False)),
            chunk_size=int(data["chunk_size"]),
            n_action_steps=int(data.get("n_action_steps", data["chunk_size"])),
            normalization_mapping=dict(data.get("normalization_mapping", {})),
            vision_backbone=data.get("vision_backbone"),
            pretrained_backbone_weights=data.get("pretrained_backbone_weights"),
            dim_model=data.get("dim_model"),
            n_heads=data.get("n_heads"),
            dim_feedforward=data.get("dim_feedforward"),
            n_encoder_layers=data.get("n_encoder_layers"),
            n_decoder_layers=data.get("n_decoder_layers"),
            use_vae=data.get("use_vae"),
            latent_dim=data.get("latent_dim"),
            raw=dict(data),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "ACTCheckpointConfig":
        root = Path(checkpoint_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(
                "ACT checkpoint must be a directory containing its config/model files. "
                f"Received: {root}"
            )

        candidates = [
            root / "config.json",
            root / "policy_config.json",
        ]
        config_path = next((p for p in candidates if p.exists()), None)
        if config_path is None:
            raise FileNotFoundError(
                "Could not find ACT checkpoint configuration. Expected one of: "
                + ", ".join(str(p) for p in candidates)
            )

        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self) -> None:
        """Validate the fields that matter for inference compatibility."""
        if self.n_obs_steps != 1:
            raise ValueError(
                f"This ACT adapter currently expects n_obs_steps=1; checkpoint has {self.n_obs_steps}."
            )

        if self.chunk_size < 1:
            raise ValueError("ACT chunk_size must be >= 1")

        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError(
                "ACT n_action_steps must satisfy 1 <= n_action_steps <= chunk_size; "
                f"got {self.n_action_steps} and {self.chunk_size}."
            )

        missing = sorted(set(EXPECTED_INPUTS) - set(self.input_features))
        extra = sorted(set(self.input_features) - set(EXPECTED_INPUTS))
        if missing or extra:
            raise ValueError(
                "ACT input feature mismatch. "
                f"Missing={missing}; extra={extra}."
            )

        for key, (expected_type, expected_shape) in EXPECTED_INPUTS.items():
            feature = self.input_features[key]
            actual_type = str(feature.get("type", "")).upper()
            actual_shape = tuple(int(x) for x in feature.get("shape", []))
            if actual_type != expected_type or actual_shape != expected_shape:
                raise ValueError(
                    f"ACT input feature '{key}' mismatch: "
                    f"expected type={expected_type}, shape={expected_shape}; "
                    f"got type={actual_type}, shape={actual_shape}."
                )

        if set(self.output_features) != {"action"}:
            raise ValueError(
                f"ACT output features must be exactly ['action']; got {sorted(self.output_features)}"
            )

        action_feature = self.output_features["action"]
        actual_type = str(action_feature.get("type", "")).upper()
        actual_shape = tuple(int(x) for x in action_feature.get("shape", []))
        if (actual_type, actual_shape) != EXPECTED_OUTPUT["action"]:
            raise ValueError(
                "ACT action output mismatch: "
                f"expected type=ACTION, shape=(16,); "
                f"got type={actual_type}, shape={actual_shape}."
            )

        expected_norm = {
            "VISUAL": "MEAN_STD",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
        for key, value in expected_norm.items():
            actual = str(self.normalization_mapping.get(key, "")).upper()
            if actual != value:
                raise ValueError(
                    f"ACT normalization mismatch for {key}: expected {value}, got {actual!r}"
                )

        if self.vision_backbone != "resnet18":
            raise ValueError(
                f"Unexpected ACT vision_backbone: {self.vision_backbone!r}"
            )

    @property
    def action_dim(self) -> int:
        return int(self.output_features["action"]["shape"][0])

    @property
    def camera_keys(self) -> list[str]:
        return sorted(
            key for key, spec in self.input_features.items()
            if str(spec.get("type", "")).upper() == "VISUAL"
        )

    @property
    def state_keys(self) -> list[str]:
        return sorted(
            key for key, spec in self.input_features.items()
            if str(spec.get("type", "")).upper() == "STATE"
        )
