"""SmolVLA checkpoint configuration and compatibility validation."""

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

EXPECTED_OUTPUT = {"action": ("ACTION", (16,))}


@dataclass
class SmolVLACheckpointConfig:
    """Normalized view of the SmolVLA config stored with a checkpoint."""

    type: str
    n_obs_steps: int
    input_features: dict[str, dict[str, Any]]
    output_features: dict[str, dict[str, Any]]
    device: str = "cuda"
    use_amp: bool = False
    use_peft: bool = False
    pretrained_path: str | None = None
    pretrained_revision: str | None = None
    chunk_size: int = 50
    n_action_steps: int = 50
    normalization_mapping: dict[str, str] = field(default_factory=dict)
    max_state_dim: int = 32
    max_action_dim: int = 32
    resize_imgs_with_padding: tuple[int, int] | None = (512, 512)
    empty_cameras: int = 0
    adapt_to_pi_aloha: bool = False
    use_delta_joint_actions_aloha: bool = False
    tokenizer_max_length: int = 48
    num_steps: int = 10
    use_cache: bool = True
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True
    vlm_model_name: str | None = None
    load_vlm_weights: bool = False
    add_image_special_tokens: bool = False
    attention_mode: str | None = None
    prefix_length: int | None = None
    pad_language_to: str = "longest"
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    min_period: float = 0.004
    max_period: float = 4.0
    rtc_config: dict[str, Any] | None = None
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SmolVLACheckpointConfig":
        if str(data.get("type", "")).lower() != "smolvla":
            raise ValueError(
                f"Checkpoint config type must be 'smolvla', got {data.get('type')!r}"
            )

        resize_value = data.get("resize_imgs_with_padding")
        resize = None if resize_value is None else tuple(int(x) for x in resize_value)

        return cls(
            type=str(data["type"]),
            n_obs_steps=int(data["n_obs_steps"]),
            input_features=dict(data["input_features"]),
            output_features=dict(data["output_features"]),
            device=str(data.get("device", "cuda")),
            use_amp=bool(data.get("use_amp", False)),
            use_peft=bool(data.get("use_peft", False)),
            pretrained_path=data.get("pretrained_path"),
            pretrained_revision=data.get("pretrained_revision"),
            chunk_size=int(data["chunk_size"]),
            n_action_steps=int(data.get("n_action_steps", data["chunk_size"])),
            normalization_mapping=dict(data.get("normalization_mapping", {})),
            max_state_dim=int(data.get("max_state_dim", 32)),
            max_action_dim=int(data.get("max_action_dim", 32)),
            resize_imgs_with_padding=resize,
            empty_cameras=int(data.get("empty_cameras", 0)),
            adapt_to_pi_aloha=bool(data.get("adapt_to_pi_aloha", False)),
            use_delta_joint_actions_aloha=bool(data.get("use_delta_joint_actions_aloha", False)),
            tokenizer_max_length=int(data.get("tokenizer_max_length", 48)),
            num_steps=int(data.get("num_steps", 10)),
            use_cache=bool(data.get("use_cache", True)),
            freeze_vision_encoder=bool(data.get("freeze_vision_encoder", False)),
            train_expert_only=bool(data.get("train_expert_only", False)),
            train_state_proj=bool(data.get("train_state_proj", True)),
            vlm_model_name=data.get("vlm_model_name"),
            load_vlm_weights=bool(data.get("load_vlm_weights", False)),
            add_image_special_tokens=bool(data.get("add_image_special_tokens", False)),
            attention_mode=data.get("attention_mode"),
            prefix_length=data.get("prefix_length"),
            pad_language_to=str(data.get("pad_language_to", "longest")),
            num_expert_layers=int(data.get("num_expert_layers", -1)),
            num_vlm_layers=int(data.get("num_vlm_layers", 16)),
            self_attn_every_n_layers=int(data.get("self_attn_every_n_layers", 2)),
            expert_width_multiplier=float(data.get("expert_width_multiplier", 0.75)),
            min_period=float(data.get("min_period", 0.004)),
            max_period=float(data.get("max_period", 4.0)),
            rtc_config=data.get("rtc_config"),
            compile_model=bool(data.get("compile_model", False)),
            compile_mode=str(data.get("compile_mode", "max-autotune")),
            raw=dict(data),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "SmolVLACheckpointConfig":
        root = Path(checkpoint_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(
                "SmolVLA checkpoint must be a directory containing its config/model files. "
                f"Received: {root}"
            )

        candidates = [root / "config.json", root / "policy_config.json"]
        config_path = next((p for p in candidates if p.exists()), None)
        if config_path is None:
            raise FileNotFoundError(
                "Could not find SmolVLA checkpoint configuration. Expected one of: "
                + ", ".join(str(p) for p in candidates)
            )

        with config_path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def validate(self) -> None:
        if self.n_obs_steps != 1:
            raise ValueError(
                f"This SmolVLA adapter expects n_obs_steps=1; checkpoint has {self.n_obs_steps}."
            )
        if self.chunk_size < 1:
            raise ValueError("SmolVLA chunk_size must be >= 1")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError(
                "SmolVLA n_action_steps must satisfy 1 <= n_action_steps <= chunk_size; "
                f"got {self.n_action_steps} and {self.chunk_size}."
            )

        missing = sorted(set(EXPECTED_INPUTS) - set(self.input_features))
        extra = sorted(set(self.input_features) - set(EXPECTED_INPUTS))
        if missing or extra:
            raise ValueError(
                f"SmolVLA input feature mismatch. Missing={missing}; extra={extra}."
            )

        for key, (expected_type, expected_shape) in EXPECTED_INPUTS.items():
            feature = self.input_features[key]
            actual_type = str(feature.get("type", "")).upper()
            actual_shape = tuple(int(x) for x in feature.get("shape", []))
            if actual_type != expected_type or actual_shape != expected_shape:
                raise ValueError(
                    f"SmolVLA input feature '{key}' mismatch: expected "
                    f"type={expected_type}, shape={expected_shape}; "
                    f"got type={actual_type}, shape={actual_shape}."
                )

        if set(self.output_features) != {"action"}:
            raise ValueError(
                f"SmolVLA output features must be exactly ['action']; got {sorted(self.output_features)}"
            )
        actual_output = self.output_features["action"]
        if (
            str(actual_output.get("type", "")).upper(),
            tuple(int(x) for x in actual_output.get("shape", [])),
        ) != EXPECTED_OUTPUT["action"]:
            raise ValueError(
                "SmolVLA action output mismatch: expected type=ACTION, shape=(16,); "
                f"got type={actual_output.get('type')!r}, shape={actual_output.get('shape')!r}."
            )

        expected_norm = {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
        for key, expected in expected_norm.items():
            actual = str(self.normalization_mapping.get(key, "")).upper()
            if actual != expected:
                raise ValueError(
                    f"SmolVLA normalization mismatch for {key}: expected {expected}, got {actual!r}"
                )

        if self.max_state_dim < self.state_dim or self.max_action_dim < self.action_dim:
            raise ValueError(
                "SmolVLA max_state_dim/max_action_dim must cover the checkpoint's original dimensions."
            )
        if self.resize_imgs_with_padding is not None and len(self.resize_imgs_with_padding) != 2:
            raise ValueError("resize_imgs_with_padding must be [width, height] or null")
        if self.vlm_model_name is None:
            raise ValueError("SmolVLA checkpoint must define vlm_model_name")

    @property
    def action_dim(self) -> int:
        return int(self.output_features["action"]["shape"][0])

    @property
    def state_dim(self) -> int:
        return int(self.input_features["observation.state"]["shape"][0])

    @property
    def camera_keys(self) -> list[str]:
        return sorted(
            key
            for key, spec in self.input_features.items()
            if str(spec.get("type", "")).upper() == "VISUAL"
        )

    @property
    def state_keys(self) -> list[str]:
        return sorted(
            key
            for key, spec in self.input_features.items()
            if str(spec.get("type", "")).upper() == "STATE"
        )
