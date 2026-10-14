

#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _ensure_src_on_path(repo_root: Path) -> None:
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _preload_groot_config(repo_root: Path) -> None:
    groot_dir = repo_root / "src" / "lerobot" / "policies" / "groot"
    package_name = "lerobot.policies.groot"
    config_name = f"{package_name}.configuration_groot"

    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__file__ = str(groot_dir / "__init__.py")
        package.__path__ = [str(groot_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package

    if config_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(config_name, groot_dir / "configuration_groot.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load spec for {config_name}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[config_name] = module
        spec.loader.exec_module(module)

        package = sys.modules[package_name]
        setattr(package, "configuration_groot", module)
        setattr(package, "GrootConfig", module.GrootConfig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    _ensure_src_on_path(repo_root)
    _preload_groot_config(repo_root)

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
