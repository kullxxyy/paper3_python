"""Helpers that make Paper 3 modules runnable from an IDE Run button."""

from pathlib import Path
from importlib import import_module
import sys


def ensure_project_root(file_path):
    """Add the python_code folder to sys.path for direct file execution."""
    path = Path(file_path).resolve()
    paper3_root = next(parent for parent in path.parents if parent.name == "paper3_python")
    project_root = paper3_root

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def run_pipeline_mode(mode="all"):
    """Run the refactored pipeline with default paths and a selected mode."""
    from workflow.config import Paper3Config
    from workflow.pipeline import Paper3Pipeline

    print("\n" + "=" * 90)
    print("PAPER 3")
    print(f"RUNNING MODE: {mode.upper()}")
    print("=" * 90)

    pipeline = Paper3Pipeline(Paper3Config(run_mode=mode))
    state = pipeline.run()

    print("\n" + "=" * 90)
    print(f"PAPER 3 MODE FINISHED: {mode.upper()}")
    print("=" * 90)

    return state


def run_preparation_step(step="prep"):
    """Run one preparation-oriented module directly from an IDE."""
    from workflow.config import Paper3Config

    prepare_data = import_module(
        "workflow.prep.01_data_prep"
    ).prepare_data
    build_nonfarm_outcomes = import_module(
        "workflow.prep.01_nonfarm_outcomes"
    ).build_nonfarm_outcomes
    build_friction_indices = import_module(
        "workflow.prep.05_friction_index"
    ).build_friction_indices
    build_household_panel = import_module(
        "workflow.prep.01_household_panel"
    ).build_household_panel

    config = Paper3Config(run_mode="prep")

    print("\n" + "=" * 90)
    print("PAPER 3")
    print(f"RUNNING PREPARATION STEP: {step.upper()}")
    print("=" * 90)

    state = prepare_data(
        datadir=config.data_dir,
        logdir=config.log_dir,
    )

    if step in {"nonfarm_outcomes", "friction", "panel"}:
        state["df"] = build_nonfarm_outcomes(state["df"])

    if step in {"friction", "panel"}:
        state["df"], state["friction_meta"] = build_friction_indices(
            state["df"],
            state["pre_waves"],
        )

    if step == "panel":
        state["hh"], state["run_fe"] = build_household_panel(state["df"])

    print("\n" + "=" * 90)
    print(f"PREPARATION STEP FINISHED: {step.upper()}")
    print("=" * 90)

    return state


if __name__ == "__main__":
    run_pipeline_mode("all")
