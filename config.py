"""Configuration for the refactored Paper 3 pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paper3Config:
    """Filesystem paths and run-mode settings used by the analysis pipeline."""

    data_dir: Path = Path("/Users/lenovo/PhD papers/Paper 3_new2/data/CHNS_data_analysis")
    output_dir: Path = Path("/Users/lenovo/PhD papers/Paper 3_new2/python_code/results")
    log_dir: Path = Path("/Users/lenovo/PhD papers/Paper 3_new2/python_code/logs")
    run_mode: str = "all"


VALID_RUN_MODES = {
    "prep",
    "baseline",
    "event",
    "bootstrap",
    "decomposition",
    "nonfarm",
    "placebo",
    "all",
}
