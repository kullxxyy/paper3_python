"""Self-contained Paper 3 workflow package."""

from .config import Paper3Config
from .pipeline import Paper3Pipeline

__all__ = ["Paper3Config", "Paper3Pipeline"]


if __name__ == "__main__":
    from .direct_run import run_pipeline_mode

    run_pipeline_mode("all")
