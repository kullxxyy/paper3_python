if __package__ in (None, ""):
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from workflow._deps import *
else:
    from .._deps import *

def run_placebo_timing_tests(hh, run_fe, Xdemog):
    """
    Pre-policy falsification tests for the MGPP DID design.

    Purpose
    -------
    Ask whether Hunan already showed a differential change relative to Guizhou
    BEFORE the true MGPP treatment period.

    Design
    ------
    Use only the three pre-policy CHNS waves: 1997, 2000, 2004.

    Two deliberately false treatment timings are tested:
      1) Fake post starts at wave 2000
      2) Fake post starts at wave 2004

    For each fake timing, estimate:
      A. Placebo baseline DID
      B. Placebo DID × pre-policy market friction

    Interpretation
    --------------
    These are falsification checks, not alternative treatment-effect estimates.
    Large/significant placebo coefficients would suggest that differential
    pre-policy movements may be contributing to the main DID result.

    Important
    ---------
    Treatment is assigned at the province level (Hunan vs Guizhou). Because the
    current design has only two provinces, we do NOT randomly permute treatment
    across communities: doing so would not respect the actual assignment level.
    """

    hh = hh.copy()

    pre_waves = [1997, 2000, 2004]
    fake_start_waves = [2000, 2004]

    # -------------------------------------------------------------------------
    # 1. PRE-POLICY SAMPLES ONLY
    # -------------------------------------------------------------------------

    placebo_base = hh[
        hh["wave"].isin(pre_waves)
        & hh["sample_hh_ctrl"].eq(1)
    ].copy()

    placebo_friction = hh[
        hh["wave"].isin(pre_waves)
        & hh["sample_friction_hh"].eq(1)
    ].copy()

    baseline_rows = []
    friction_rows = []
    baseline_models = {}
    friction_models = {}

    # -------------------------------------------------------------------------
    # 2. LOOP OVER FALSE POLICY DATES
    # -------------------------------------------------------------------------

    for fake_start in fake_start_waves:

        # =========================
        # A. PLACEBO BASELINE DID
        # =========================
        temp = placebo_base.copy()

        temp["placebo_post"] = (
            temp["wave"] >= fake_start
        ).astype(int)

        temp["placebo_did"] = (
            temp["rice_treated"]
            * temp["placebo_post"]
        )

        res_base, reg_base = run_fe(
            data=temp,
            yvar="asinh_hh_farm_hours",
            xvars=["placebo_did"] + Xdemog
        )

        baseline_models[fake_start] = {
            "result": res_base,
            "reg": reg_base,
        }

        baseline_rows.append({
            "Fake start wave": fake_start,
            "Coefficient": res_base.params["placebo_did"],
            "SE": res_base.std_errors["placebo_did"],
            "p-value": res_base.pvalues["placebo_did"],
            "N": res_base.nobs,
            "Households": (
                reg_base.index
                .get_level_values("hhid")
                .nunique()
            ),
            "Communities": reg_base["COMMID"].nunique(),
        })

        # =====================================
        # B. PLACEBO DID × MARKET FRICTION
        # =====================================
        temp_fr = placebo_friction.copy()

        temp_fr["placebo_post"] = (
            temp_fr["wave"] >= fake_start
        ).astype(int)

        temp_fr["placebo_did"] = (
            temp_fr["rice_treated"]
            * temp_fr["placebo_post"]
        )

        temp_fr["placebo_post_x_friction"] = (
            temp_fr["placebo_post"]
            * temp_fr["friction_core"]
        )

        temp_fr["placebo_did_x_friction"] = (
            temp_fr["placebo_did"]
            * temp_fr["friction_core"]
        )

        res_fr, reg_fr = run_fe(
            data=temp_fr,
            yvar="asinh_hh_farm_hours",
            xvars=[
                "placebo_did",
                "placebo_post_x_friction",
                "placebo_did_x_friction",
            ] + Xdemog
        )

        friction_models[fake_start] = {
            "result": res_fr,
            "reg": reg_fr,
        }

        friction_rows.append({
            "Fake start wave": fake_start,
            "Placebo DID": res_fr.params["placebo_did"],
            "DID SE": res_fr.std_errors["placebo_did"],
            "DID p-value": res_fr.pvalues["placebo_did"],
            "Placebo DID × Friction": (
                res_fr.params["placebo_did_x_friction"]
            ),
            "Interaction SE": (
                res_fr.std_errors["placebo_did_x_friction"]
            ),
            "Interaction p-value": (
                res_fr.pvalues["placebo_did_x_friction"]
            ),
            "N": res_fr.nobs,
            "Households": (
                reg_fr.index
                .get_level_values("hhid")
                .nunique()
            ),
            "Communities": reg_fr["COMMID"].nunique(),
        })

    # -------------------------------------------------------------------------
    # 3. TABLES
    # -------------------------------------------------------------------------

    baseline_placebo_table = pd.DataFrame(baseline_rows)
    friction_placebo_table = pd.DataFrame(friction_rows)

    print("\n" + "=" * 90)
    print("PRE-POLICY PLACEBO TIMING TESTS")
    print("Only waves 1997, 2000, 2004 are used")
    print("=" * 90)

    print("\nA. Placebo baseline DID")
    print(
        baseline_placebo_table
        .round(3)
        .to_string(index=False)
    )

    print("\nB. Placebo DID × market friction")
    print(
        friction_placebo_table
        .round(3)
        .to_string(index=False)
    )

    print("\nInterpretation:")
    print(
        "These coefficients should ideally be small and statistically weak, "
        "because every tested period is before the true MGPP post period."
    )

    return {
        "baseline_placebo_table": baseline_placebo_table,
        "friction_placebo_table": friction_placebo_table,
        "baseline_models": baseline_models,
        "friction_models": friction_models,
    }


if __name__ == "__main__":
    from workflow.direct_run import run_pipeline_mode

    run_pipeline_mode("placebo")
