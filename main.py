"""Command-line entry point for the refactored Paper 3 analysis."""

import argparse
import sys
from pathlib import Path

if __package__:
    from .config import Paper3Config, VALID_RUN_MODES
    from .pipeline import Paper3Pipeline
else:
    # Support running from inside paper3_python with `python3 -m main`
    # or `python3 main.py` while preserving package-relative imports elsewhere.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paper3_python.config import Paper3Config, VALID_RUN_MODES
    from paper3_python.pipeline import Paper3Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the refactored Paper 3 analysis pipeline."
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=sorted(VALID_RUN_MODES),
        help="Analysis block to run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Paper3Config.data_dir,
        help="Directory containing the CHNS input data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Paper3Config.output_dir,
        help="Directory for generated figures and outputs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Paper3Config.log_dir,
        help="Directory for the run log.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Paper3Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        run_mode=args.mode,
    )
    pipeline = Paper3Pipeline(config)
    pipeline.run()
    return pipeline


if __name__ == "__main__":
    main()
