"""Action/state-column labels in the dataset's original order."""

import json
from pathlib import Path
from typing import Any


def _feature_names(info: dict[str, Any], feature_key: str, dim: int, label: str) -> tuple[list[str], str]:
    feature = info.get("features", {}).get(feature_key, {})
    if feature.get("shape") is not None and list(feature["shape"]) != [dim]:
        raise ValueError(f"Dataset {feature_key} shape does not match the checkpoint {label} dimension")
    names = feature.get("names")
    if names is None:
        return [f"{label} {index + 1}" for index in range(dim)], "column_index"
    if (
        not isinstance(names, list)
        or len(names) != dim
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != dim
    ):
        raise ValueError(f"Dataset {feature_key}.names must contain {dim} unique {label} names in column order")
    return list(names), f"dataset.{feature_key}.names"


def action_joint_names(info: dict[str, Any], action_dim: int) -> tuple[list[str], str]:
    return _feature_names(info, "action", action_dim, "Action")


def _checkpoint_training_feature_names(checkpoint_path: str | Path, feature_key: str) -> list[str] | None:
    """The column order the checkpoint was actually trained on for `feature_key`,
    read from its own train_config.json -> training dataset's meta/info.json.

    Returns None when the training dataset is unavailable (moved/deleted) or its
    info.json has no explicit names for that feature -- callers must then skip
    cross-checking and trust the evaluation dataset's own order.
    """
    train_config_path = Path(checkpoint_path) / "train_config.json"
    if not train_config_path.is_file():
        return None
    try:
        train_config = json.loads(train_config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    dataset_root = (train_config.get("dataset") or {}).get("root")
    if not dataset_root:
        return None

    info_path = Path(dataset_root).expanduser() / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    names = (info.get("features", {}).get(feature_key, {})).get("names")
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
        return None
    return list(names)


def _resolve_feature_order(
    dataset_info: dict[str, Any],
    checkpoint_path: str | Path,
    feature_key: str,
    dim: int,
    label: str,
) -> tuple[list[str], str, list[int] | None]:
    """Column names/order to report for `feature_key`, plus the permutation
    (evaluation-dataset column -> checkpoint-training column) that must be applied
    to that feature's values before they reach/leave the model.

    The model was trained on a fixed column order baked into its weights (see
    ACTAdapter.predict for actions, observation_to_act_batch for state); it never
    sees the evaluation dataset's own order. If the evaluation dataset happens to
    store this feature in a different order (e.g. a re-recorded/re-converted
    dataset with the joints listed differently), feeding/reading columns
    positionally under a single `names[i]` silently mixes up two different joints
    -- the historical left/right arm swap class of bug. This permutes the
    evaluation-dataset values into the checkpoint's order instead of trusting a
    coincidental match, and raises if the two datasets don't even agree on which
    joints exist.
    """
    dataset_names, dataset_source = _feature_names(dataset_info, feature_key, dim, label)
    checkpoint_names = _checkpoint_training_feature_names(checkpoint_path, feature_key)

    if checkpoint_names is None:
        # Can't verify against the checkpoint's own training dataset (moved/deleted,
        # or it never recorded explicit names) -- fall back to trusting the
        # evaluation dataset's order, as before.
        return dataset_names, dataset_source, None

    if len(checkpoint_names) != dim:
        raise ValueError(
            f"Checkpoint's training dataset recorded {len(checkpoint_names)} {feature_key} names, "
            f"but the checkpoint's {label.lower()} dimension is {dim}"
        )

    if set(checkpoint_names) != set(dataset_names):
        raise ValueError(
            f"Evaluation dataset's {feature_key} joints do not match the checkpoint's training joints. "
            f"Training joints: {sorted(checkpoint_names)}; "
            f"evaluation dataset joints: {sorted(dataset_names)}"
        )

    if checkpoint_names == dataset_names:
        return checkpoint_names, dataset_source, None

    # Same joints, different column order: permutation[i] is the evaluation-dataset
    # column that holds the checkpoint-order joint i.
    permutation = [dataset_names.index(name) for name in checkpoint_names]
    return checkpoint_names, f"checkpoint.training_dataset.{feature_key}.names", permutation


def resolve_action_order(
    dataset_info: dict[str, Any],
    checkpoint_path: str | Path,
    action_dim: int,
) -> tuple[list[str], str, list[int] | None]:
    return _resolve_feature_order(dataset_info, checkpoint_path, "action", action_dim, "Action")


def resolve_state_order(
    dataset_info: dict[str, Any],
    checkpoint_path: str | Path,
    state_dim: int,
) -> tuple[list[str], str, list[int] | None]:
    return _resolve_feature_order(dataset_info, checkpoint_path, "observation.state", state_dim, "State")
