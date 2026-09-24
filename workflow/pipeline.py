"""Object-oriented entry point for the self-contained Paper 3 workflow."""

from importlib import import_module
from pathlib import Path
import sys


if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from workflow.config import Paper3Config, VALID_RUN_MODES


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
run_baseline_and_friction = import_module(
    "workflow.analyses.02_farm_main"
).run_baseline_and_friction
run_nonfarm_analysis = import_module(
    "workflow.analyses.03_nonfarm_main"
).run_nonfarm_analysis
run_farm_decomposition = import_module(
    "workflow.analyses.04_decomposition"
).run_farm_decomposition
run_friction_event_study = import_module(
    "workflow.analyses.06_event_study"
).run_friction_event_study
run_placebo_timing_tests = import_module(
    "workflow.analyses.07_placebo"
).run_placebo_timing_tests
run_cluster_inference_and_bootstrap = import_module(
    "workflow.analyses.08_bootstrap"
).run_cluster_inference_and_bootstrap


class Paper3Pipeline:
    """Coordinate the Paper 3 data preparation and analysis modules.

    The pipeline keeps intermediate outputs in ``self.state``. This mirrors the
    original script's ``STATE`` dictionary, while making the workflow accessible
    through one class.
    """

    def __init__(self, config=None):
        self.config = config or Paper3Config()
        self.state = {}

    def prepare(self):
        """Run common preparation needed by all analysis modes."""
        self.state.update(
            prepare_data(
                datadir=self.config.data_dir,
                logdir=self.config.log_dir,
            )
        )

        self.state["df"] = build_nonfarm_outcomes(
            self.state["df"]
        )

        self.state["df"], self.state["friction_meta"] = build_friction_indices(
            self.state["df"],
            self.state["pre_waves"],
        )

        self.state["hh"], self.state["run_fe"] = build_household_panel(
            self.state["df"]
        )

        return self.state

    def ensure_prepared(self):
        """Prepare once, then reuse the existing state."""
        if "hh" not in self.state:
            self.prepare()
        return self.state

    def run_baseline(self):
        self.ensure_prepared()
        self.state["hh"], self.state["baseline"] = run_baseline_and_friction(
            self.state["hh"],
            self.state["run_fe"],
            self.state["Xdemog"],
        )
        return self.state["baseline"]

    def run_event_study(self):
        self.ensure_prepared()
        self.state["hh"], self.state["event"] = run_friction_event_study(
            self.state["hh"],
            self.state["run_fe"],
            self.state["Xdemog"],
            outputdir=self.config.output_dir,
        )
        return self.state["event"]

    def run_bootstrap(self):
        if "baseline" not in self.state:
            self.run_baseline()
        if "event" not in self.state:
            self.run_event_study()

        self.state["bootstrap"] = run_cluster_inference_and_bootstrap(
            self.state["baseline"],
            self.state["event"],
        )
        return self.state["bootstrap"]

    def run_decomposition(self):
        self.ensure_prepared()
        self.state["decomposition"] = run_farm_decomposition(
            self.state["hh"],
            self.state["run_fe"],
            self.state["Xdemog"],
        )
        return self.state["decomposition"]

    def run_nonfarm(self):
        self.ensure_prepared()
        self.state["hh"], self.state["nonfarm"] = run_nonfarm_analysis(
            self.state["hh"],
            self.state["run_fe"],
            self.state["Xdemog"],
        )
        return self.state["nonfarm"]

    def run_placebo(self):
        self.ensure_prepared()
        self.state["placebo"] = run_placebo_timing_tests(
            self.state["hh"],
            self.state["run_fe"],
            self.state["Xdemog"],
        )
        return self.state["placebo"]

    def run(self, mode=None):
        """Run one of the supported modes and return the pipeline state."""
        run_mode = mode or self.config.run_mode
        if run_mode not in VALID_RUN_MODES:
            valid = ", ".join(sorted(VALID_RUN_MODES))
            raise ValueError(f"Unknown run mode '{run_mode}'. Choose one of: {valid}.")

        self.ensure_prepared()

        if run_mode in {"prep"}:
            return self.state

        if run_mode in {"baseline", "bootstrap", "all"}:
            self.run_baseline()

        if run_mode in {"event", "bootstrap", "all"}:
            self.run_event_study()

        if run_mode in {"bootstrap", "all"}:
            self.run_bootstrap()

        if run_mode in {"decomposition", "all"}:
            self.run_decomposition()

        if run_mode in {"nonfarm", "all"}:
            self.run_nonfarm()

        if run_mode in {"placebo", "all"}:
            self.run_placebo()

        return self.state


if __name__ == "__main__":
    from workflow.direct_run import run_pipeline_mode

    run_pipeline_mode("all")
