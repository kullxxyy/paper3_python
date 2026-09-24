if __package__ in (None, ""):
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from workflow._deps import *
else:
    from .._deps import *

def run_nonfarm_analysis(
    hh,
    run_fe,
    Xdemog
):
    """
    CELL 8.

    Household primary-job nonfarm labor analysis.

    Includes:
      1. Household nonfarm panel sample
      2. Baseline DID
      3. Standard event study
      4. Static market-friction heterogeneity
      5. Friction event study
      6. Joint pre-trend / post tests
      7. Dynamic effects at -1 / 0 / +1 SD friction

    Outcome:
        asinh household primary-job nonfarm weekly hours

    Fixed effects:
        household FE + wave FE

    Cluster:
        COMMID

    Reference event period:
        -2
    """

    hh = hh.copy()


    # =============================================================================
    # 28. NONFARM HOUSEHOLD PANEL SAMPLE
    # =============================================================================

    hh["sample_nf_raw"] = (
        hh["sample_main"].eq(1)
        & hh[
            "asinh_hh_primary_nonfarm_hours"
        ].notna()
        & hh["COMMID"].notna()
    ).astype(int)


    # Number of usable nonfarm waves per household
    hh["n_nf_obs"] = (
        hh.groupby(
            "hhid"
        )["sample_nf_raw"]
        .transform("sum")
    )


    # At least two observations
    hh["sample_nf_hh"] = (
        hh["sample_nf_raw"].eq(1)
        & hh["n_nf_obs"].ge(2)
    ).astype(int)


    # Controls available
    hh["sample_nf_ctrl"] = (
        hh["sample_nf_hh"].eq(1)
        & hh["controls_complete"].eq(1)
    ).astype(int)


    # Friction available
    hh["sample_nf_friction"] = (
        hh["sample_nf_ctrl"].eq(1)
        & hh["friction_core"].notna()
    ).astype(int)


    print("\n" + "=" * 80)
    print("NONFARM HOUSEHOLD PANEL SAMPLE")
    print("=" * 80)


    print(
        "sample_nf_hh       =",
        hh["sample_nf_hh"].sum()
    )

    print(
        "sample_nf_ctrl     =",
        hh["sample_nf_ctrl"].sum()
    )

    print(
        "sample_nf_friction =",
        hh["sample_nf_friction"].sum()
    )


    print("\nSample by wave:")

    print(
        pd.crosstab(
            hh.loc[
                hh["sample_nf_hh"].eq(1),
                "wave"
            ],
            hh.loc[
                hh["sample_nf_hh"].eq(1),
                "rice_treated"
            ]
        )
    )


    # =============================================================================
    # 29. BASELINE NONFARM DID
    # =============================================================================

    # -----------------------------------------------------------------------------
    # 29.1 DID WITHOUT CONTROLS
    # -----------------------------------------------------------------------------

    nf_noctrl = hh[
        hh["sample_nf_hh"].eq(1)
    ].copy()


    res_nf_did1, reg_nf_did1 = (
        run_fe(
            data=nf_noctrl,
            yvar=
            "asinh_hh_primary_nonfarm_hours",
            xvars=[
                "rice_did"
            ]
        )
    )


    # -----------------------------------------------------------------------------
    # 29.2 DID WITH CONTROLS
    # -----------------------------------------------------------------------------

    nf_ctrl = hh[
        hh["sample_nf_ctrl"].eq(1)
    ].copy()


    res_nf_did2, reg_nf_did2 = (
        run_fe(
            data=nf_ctrl,
            yvar=
            "asinh_hh_primary_nonfarm_hours",
            xvars=[
                "rice_did"
            ] + Xdemog
        )
    )


    # -----------------------------------------------------------------------------
    # 29.3 BASELINE RESULTS TABLE
    # -----------------------------------------------------------------------------

    nf_did_table = pd.DataFrame({

        "Model": [
            "Nonfarm DID",
            "Nonfarm DID + controls"
        ],

        "DID": [
            res_nf_did1.params[
                "rice_did"
            ],
            res_nf_did2.params[
                "rice_did"
            ]
        ],

        "SE": [
            res_nf_did1.std_errors[
                "rice_did"
            ],
            res_nf_did2.std_errors[
                "rice_did"
            ]
        ],

        "p-value": [
            res_nf_did1.pvalues[
                "rice_did"
            ],
            res_nf_did2.pvalues[
                "rice_did"
            ]
        ],

        "N": [
            res_nf_did1.nobs,
            res_nf_did2.nobs
        ],

        "Households": [

            reg_nf_did1.index
            .get_level_values(
                "hhid"
            )
            .nunique(),

            reg_nf_did2.index
            .get_level_values(
                "hhid"
            )
            .nunique()
        ],

        "Communities": [

            reg_nf_did1[
                "COMMID"
            ].nunique(),

            reg_nf_did2[
                "COMMID"
            ].nunique()
        ]
    })


    print("\n" + "=" * 80)
    print("NONFARM BASELINE DID")
    print("=" * 80)


    print(
        nf_did_table
        .round(3)
        .to_string(
            index=False
        )
    )


    # =============================================================================
    # 30. STANDARD NONFARM EVENT STUDY
    # =============================================================================

    event_map = {
        "m9": -9,
        "m6": -6,
        "p0": 0,
        "p3": 3,
        "p5": 5
    }


    rice_event_vars = [
        "rice_evt_m9",
        "rice_evt_m6",
        "rice_evt_p0",
        "rice_evt_p3",
        "rice_evt_p5"
    ]


    res_nf_event, reg_nf_event = (
        run_fe(
            data=nf_ctrl,
            yvar=
            "asinh_hh_primary_nonfarm_hours",
            xvars=(
                rice_event_vars
                + Xdemog
            )
        )
    )


    # =============================================================================
    # 31. JOINT WALD TEST
    # =============================================================================

    def joint_wald_test(
        result,
        variables
    ):

        names = list(
            result.params.index
        )


        R = np.zeros(
            (
                len(variables),
                len(names)
            )
        )


        for i, var in enumerate(
            variables
        ):

            R[
                i,
                names.index(var)
            ] = 1


        test = result.wald_test(
            R
        )


        return {

            "Statistic":
                float(
                    np.asarray(
                        test.stat
                    ).squeeze()
                ),

            "p-value":
                float(
                    test.pval
                )
        }


    # -----------------------------------------------------------------------------
    # 31.1 PRETREND
    # -----------------------------------------------------------------------------

    nf_pretrend = joint_wald_test(
        res_nf_event,
        [
            "rice_evt_m9",
            "rice_evt_m6"
        ]
    )


    # -----------------------------------------------------------------------------
    # 31.2 POST-POLICY JOINT TEST
    # -----------------------------------------------------------------------------

    nf_post = joint_wald_test(
        res_nf_event,
        [
            "rice_evt_p0",
            "rice_evt_p3",
            "rice_evt_p5"
        ]
    )


    print("\n" + "=" * 80)
    print("NONFARM EVENT STUDY")
    print("Reference period: event = -2")
    print("=" * 80)


    print(
        res_nf_event.summary
    )


    print(
        "\nJoint pretrend p = "
        f"{nf_pretrend['p-value']:.4f}"
    )


    print(
        "Joint post p     = "
        f"{nf_post['p-value']:.4f}"
    )


    # =============================================================================
    # 32. STATIC NONFARM × CORE FRICTION
    # =============================================================================

    hh[
        "post_x_friction_core"
    ] = (
        hh["rice_post"]
        * hh["friction_core"]
    )


    hh[
        "did_x_friction_core"
    ] = (
        hh["rice_did"]
        * hh["friction_core"]
    )


    nf_friction_sample = hh[
        hh[
            "sample_nf_friction"
        ].eq(1)
    ].copy()


    res_nf_friction, reg_nf_friction = (
        run_fe(

            data=
            nf_friction_sample,

            yvar=
            "asinh_hh_primary_nonfarm_hours",

            xvars=[
                "rice_did",
                "post_x_friction_core",
                "did_x_friction_core"
            ] + Xdemog
        )
    )


    print("\n" + "=" * 80)
    print("NONFARM × CORE MARKET FRICTION")
    print("=" * 80)


    print(
        res_nf_friction.summary
    )


    print("\nMain coefficients:")


    print(
        "DID              = "
        f"{res_nf_friction.params['rice_did']:.3f}"
    )


    print(
        "Post × Friction  = "
        f"{res_nf_friction.params['post_x_friction_core']:.3f}"
    )


    print(
        "DID × Friction   = "
        f"{res_nf_friction.params['did_x_friction_core']:.3f}"
    )


    print(
        "Interaction SE    = "
        f"{res_nf_friction.std_errors['did_x_friction_core']:.3f}"
    )


    print(
        "Interaction p     = "
        f"{res_nf_friction.pvalues['did_x_friction_core']:.3f}"
    )


    # =============================================================================
    # 33. NONFARM FRICTION EVENT STUDY
    # =============================================================================

    for name, event_time in (
        event_map.items()
    ):

        # Event dummy for all households
        hh[f"evt_{name}"] = (
            hh["rice_event"]
            .eq(event_time)
        ).astype(int)


        # Common event × friction
        hh[
            f"evt_x_fr_{name}"
        ] = (
            hh[f"evt_{name}"]
            * hh["friction_core"]
        )


        # Treatment event × friction
        hh[
            f"rice_evt_x_fr_{name}"
        ] = (
            hh[
                f"rice_evt_{name}"
            ]
            * hh["friction_core"]
        )


    friction_time_vars = [
        "evt_x_fr_m9",
        "evt_x_fr_m6",
        "evt_x_fr_p0",
        "evt_x_fr_p3",
        "evt_x_fr_p5"
    ]


    triple_vars = [
        "rice_evt_x_fr_m9",
        "rice_evt_x_fr_m6",
        "rice_evt_x_fr_p0",
        "rice_evt_x_fr_p3",
        "rice_evt_x_fr_p5"
    ]


    nf_event_xvars = (
        rice_event_vars
        + friction_time_vars
        + triple_vars
        + Xdemog
    )


    nf_friction_sample = hh[
        hh[
            "sample_nf_friction"
        ].eq(1)
    ].copy()


    res_nf_fr_event, reg_nf_fr_event = (
        run_fe(

            data=
            nf_friction_sample,

            yvar=
            "asinh_hh_primary_nonfarm_hours",

            xvars=
            nf_event_xvars
        )
    )


    # =============================================================================
    # 34. EVENT-STUDY JOINT TESTS
    # =============================================================================

    # Mean-friction treatment pretrend
    nf_mean_pretrend = (
        joint_wald_test(
            res_nf_fr_event,
            [
                "rice_evt_m9",
                "rice_evt_m6"
            ]
        )
    )


    # Differential pretrend by friction
    nf_diff_pretrend = (
        joint_wald_test(
            res_nf_fr_event,
            [
                "rice_evt_x_fr_m9",
                "rice_evt_x_fr_m6"
            ]
        )
    )


    # Joint post-policy friction heterogeneity
    nf_post_heterogeneity = (
        joint_wald_test(
            res_nf_fr_event,
            [
                "rice_evt_x_fr_p0",
                "rice_evt_x_fr_p3",
                "rice_evt_x_fr_p5"
            ]
        )
    )


    print("\n" + "=" * 80)
    print("NONFARM FRICTION EVENT STUDY")
    print("=" * 80)


    print(
        res_nf_fr_event.summary
    )


    print(
        "\nMean-friction pretrend p = "
        f"{nf_mean_pretrend['p-value']:.4f}"
    )


    print(
        "Differential pretrend p  = "
        f"{nf_diff_pretrend['p-value']:.4f}"
    )


    print(
        "Joint post heterogeneity p = "
        f"{nf_post_heterogeneity['p-value']:.4f}"
    )


    # =============================================================================
    # 35. DYNAMIC NONFARM EFFECTS BY FRICTION LEVEL
    # =============================================================================

    dynamic_results = []


    for friction_level in [
        -1,
        0,
        1
    ]:

        for name, event_time in (
            event_map.items()
        ):

            base_var = (
                f"rice_evt_{name}"
            )


            interaction_var = (
                f"rice_evt_x_fr_{name}"
            )


            b = (
                res_nf_fr_event.params
            )


            V = (
                res_nf_fr_event.cov
            )


            effect = (
                b[base_var]
                + friction_level
                * b[interaction_var]
            )


            variance = (

                V.loc[
                    base_var,
                    base_var
                ]

                + friction_level ** 2
                * V.loc[
                    interaction_var,
                    interaction_var
                ]

                + 2
                * friction_level
                * V.loc[
                    base_var,
                    interaction_var
                ]
            )


            se = np.sqrt(
                variance
            )


            critical = (
                student_t.ppf(
                    0.975,
                    res_nf_fr_event.df_resid
                )
            )


            dynamic_results.append({

                "Friction":
                    friction_level,

                "Event":
                    event_time,

                "Effect":
                    effect,

                "SE":
                    se,

                "CI_low":
                    effect
                    - critical * se,

                "CI_high":
                    effect
                    + critical * se
            })


    nf_dynamic_table = (
        pd.DataFrame(
            dynamic_results
        )
    )


    print("\n" + "=" * 90)
    print("DYNAMIC NONFARM EFFECTS BY MARKET FRICTION")
    print("=" * 90)


    print(
        nf_dynamic_table
        .round(3)
        .to_string(
            index=False
        )
    )


    # =============================================================================
    # 36. NONFARM EVENT-STUDY SUMMARY TABLE
    # =============================================================================

    event_results = []


    for name, event_time in (
        event_map.items()
    ):

        event_results.append({

            "Event":
                event_time,

            "Treatment effect":
                res_nf_fr_event.params[
                    f"rice_evt_{name}"
                ],

            "Treatment SE":
                res_nf_fr_event.std_errors[
                    f"rice_evt_{name}"
                ],

            "Treatment p":
                res_nf_fr_event.pvalues[
                    f"rice_evt_{name}"
                ],

            "Friction heterogeneity":
                res_nf_fr_event.params[
                    f"rice_evt_x_fr_{name}"
                ],

            "Heterogeneity SE":
                res_nf_fr_event.std_errors[
                    f"rice_evt_x_fr_{name}"
                ],

            "Heterogeneity p":
                res_nf_fr_event.pvalues[
                    f"rice_evt_x_fr_{name}"
                ]
        })


    nf_event_table = (
        pd.DataFrame(
            event_results
        )
        .sort_values(
            "Event"
        )
    )


    print("\n" + "=" * 100)
    print("NONFARM EVENT-STUDY HETEROGENEITY")
    print("Reference period = -2")
    print("=" * 100)


    print(
        nf_event_table
        .round(3)
        .to_string(
            index=False
        )
    )


    # =============================================================================
    # RETURN RESULTS
    # =============================================================================

    results = {

        "nf_did_table":
            nf_did_table,

        "res_nf_did1":
            res_nf_did1,

        "reg_nf_did1":
            reg_nf_did1,

        "res_nf_did2":
            res_nf_did2,

        "reg_nf_did2":
            reg_nf_did2,

        "res_nf_event":
            res_nf_event,

        "reg_nf_event":
            reg_nf_event,

        "nf_pretrend":
            nf_pretrend,

        "nf_post":
            nf_post,

        "res_nf_friction":
            res_nf_friction,

        "reg_nf_friction":
            reg_nf_friction,

        "res_nf_fr_event":
            res_nf_fr_event,

        "reg_nf_fr_event":
            reg_nf_fr_event,

        "nf_mean_pretrend":
            nf_mean_pretrend,

        "nf_diff_pretrend":
            nf_diff_pretrend,

        "nf_post_heterogeneity":
            nf_post_heterogeneity,

        "nf_dynamic_table":
            nf_dynamic_table,

        "nf_event_table":
            nf_event_table
    }


    return hh, results


if __name__ == "__main__":
    from workflow.direct_run import run_pipeline_mode

    run_pipeline_mode("nonfarm")

# =============================================================================
# =============================================================================
# CELL 9 — PRE-POLICY PLACEBO TIMING TESTS
# =============================================================================
