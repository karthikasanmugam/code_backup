#!/usr/bin/env python3
"""Resolve a UI-selected checkpoint/run path to an actual pretrained_model dir.

Mirrors resolve_pretrained_model_dir() in examples/daksha/ui/app.py: a path from
/api/checkpoints is a training run's output_dir
(OUTPUT_ROOT/<policy>/<dataset>/<job>_<timestamp>), and the real model weights live
under "<output_dir>/checkpoints/<step-or-last>/pretrained_model" (see
lerobot.common.train_utils.save_checkpoint). If the given path is already a
pretrained_model dir (has config.json), it's printed as-is. Kept as a standalone
script (rather than importing app.py) so run_inference.sh doesn't have to drag in
Flask and the rest of the UI backend just to resolve one path.

Usage: resolve_checkpoint.py <path>   -> prints the resolved pretrained_model dir
"""

from __future__ import annotations

import sys
from pathlib import Path


def resolve_pretrained_model_dir(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if (path / "config.json").exists():
        return path

    checkpoints_dir = path / "checkpoints"
    if not checkpoints_dir.is_dir():
        return path

    last_dir = checkpoints_dir / "last" / "pretrained_model"
    if last_dir.exists():
        return last_dir

    step_dirs = sorted(
        (d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if step_dirs:
        candidate = step_dirs[-1] / "pretrained_model"
        if candidate.exists():
            return candidate

    return path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: resolve_checkpoint.py <checkpoint-or-run-path>", file=sys.stderr)
        return 2
    resolved = resolve_pretrained_model_dir(sys.argv[1])
    if not (resolved / "config.json").exists():
        print(
            f"error: no pretrained_model found under {sys.argv[1]!r} (looked in {resolved})",
            file=sys.stderr,
        )
        return 1
    print(str(resolved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
