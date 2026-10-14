from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .reporting import write_reports
from .validator import validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate LeRobot v2/v3 datasets, including chunked v3 storage."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory of the dataset.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yml"),
        help=(
            "Validator YAML configuration. "
            "Default: configs/default.yml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory from the configuration."
        ),
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    parser = build_parser()

    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    config_path = args.config.resolve()

    if not data_root.exists():
        parser.error(
            f"Dataset root does not exist: {data_root}"
        )

    if not data_root.is_dir():
        parser.error(
            f"Dataset root is not a directory: {data_root}"
        )

    try:
        config = load_config(
            config_path
        )
    except (
        FileNotFoundError,
        ConfigError,
        ValueError,
    ) as exc:
        print(
            f"[ERROR] Could not load configuration: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        validation = validate_dataset(
            data_root=data_root,
            config=config,
        )
    except Exception as exc:
        print(
            f"[ERROR] Dataset validation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    reporting_config = config.raw.get(
        "reporting",
        {},
    )

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = Path(
            reporting_config.get(
                "output_dir",
                "outputs",
            )
        )

    if not output_dir.is_absolute():
        # Treat configured output relative to the current project/run
        # directory, not the dataset.
        output_dir = Path.cwd() / output_dir

    scoring = config.raw.get(
        "scoring",
        {},
    )

    paths = write_reports(
        validation=validation,
        output_dir=output_dir,
        data_root=data_root,
        fps=None,
        stuck_thresh=None,
    )

    rejected = len(
        validation.rejected_episode_numbers
    )

    review = len(
        validation.review_episode_numbers
    )

    kept = len(
        validation.kept_episode_numbers
    )

    print()
    print("[+] Validation complete.")
    print(
        f"[*] Total Episodes : "
        f"{validation.total_episodes}"
    )
    print(
        f"[*] KEEP           : {kept}"
    )
    print(
        f"[*] REVIEW         : {review}"
    )
    print(
        f"[*] REJECT         : {rejected}"
    )
    print(
        f"[+] Flagged log    : "
        f"{paths['flagged_log']}"
    )
    print(
        f"[+] Flagged IDs    : "
        f"{paths['flagged_ids']}"
    )
    print(
        f"[+] Kept IDs       : "
        f"{paths['kept_ids']}"
    )
    print(
        f"[+] Scores CSV     : "
        f"{paths['scores_csv']}"
    )
    print(
        f"[+] Results JSON   : "
        f"{paths['results_json']}"
    )

    # Keep the parsed scoring config referenced so future CLI
    # extensions can expose score settings without restructuring.
    _ = scoring

    return 0



if __name__ == "__main__":
    main()