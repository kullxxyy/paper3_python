from .._deps import *

def run_friction_event_study(
    hh,
    run_fe,
    Xdemog,
    outputdir=Path("/Users/lenovo/PhD papers/Paper 3_new2/python_code/results"),
):
    """
    CELL 5.
    Run the friction event study, joint pre-trend tests, dynamic effects by
    friction level, and save the event-study figure.

    Returns:
        hh_out, results
    """
    hh = hh.copy()

    # =============================================================================
    # 17. FRICTION EVENT STUDY
    # =============================================================================

    event_map = {
        "m9": -9,
        "m6": -6,
        "p0": 0,
        "p3": 3,
        "p5": 5
    }


    # -----------------------------------------------------------------------------
    # Create event × friction
    # and treated × event × friction
    # -----------------------------------------------------------------------------

    for name, event_time in event_map.items():

        # Event dummy for everyone
        hh[f"evt_{name}"] = (
            hh["rice_event"] == event_time
        ).astype(int)

        # Common event-specific friction trend
        hh[f"evt_x_fr_{name}"] = (
            hh[f"evt_{name}"]
            * hh["friction_core"]
        )

        # Treated × event × friction
        hh[f"rice_evt_x_fr_{name}"] = (
            hh[f"rice_evt_{name}"]
            * hh["friction_core"]
        )


    # =============================================================================
    # EVENT-STUDY SAMPLE
    # =============================================================================

    event_sample = hh[
        hh["sample_friction_hh"].eq(1)
    ].copy()


    # Standard treatment event-study variables
    rice_event_vars = [
        "rice_evt_m9",
        "rice_evt_m6",
        "rice_evt_p0",
        "rice_evt_p3",
        "rice_evt_p5"
    ]


    # Common event × friction controls
    friction_time_vars = [
        "evt_x_fr_m9",
        "evt_x_fr_m6",
        "evt_x_fr_p0",
        "evt_x_fr_p3",
        "evt_x_fr_p5"
    ]


    # Dynamic treatment heterogeneity
    triple_vars = [
        "rice_evt_x_fr_m9",
        "rice_evt_x_fr_m6",
        "rice_evt_x_fr_p0",
        "rice_evt_x_fr_p3",
        "rice_evt_x_fr_p5"
    ]


    event_xvars = (
        rice_event_vars
        + friction_time_vars
        + triple_vars
        + Xdemog
    )


    # =============================================================================
    # RUN EVENT STUDY
    # =============================================================================

    res_event, reg_event = run_fe(
        data=event_sample,
        yvar="asinh_hh_farm_hours",
        xvars=event_xvars
    )


    print("\n" + "=" * 80)
    print("FRICTION EVENT STUDY")
    print("=" * 80)

    print(res_event.summary)

    # =============================================================================
    # 18. EVENT-STUDY SUMMARY TABLE
    # =============================================================================

    event_results = []


    for name, event_time in event_map.items():

        event_results.append({

            "Event":
                event_time,

            "Treatment effect":
                res_event.params[
                    f"rice_evt_{name}"
                ],

            "Treatment SE":
                res_event.std_errors[
                    f"rice_evt_{name}"
                ],

            "Treatment p":
                res_event.pvalues[
                    f"rice_evt_{name}"
                ],

            "Friction heterogeneity":
                res_event.params[
                    f"rice_evt_x_fr_{name}"
                ],

            "Heterogeneity SE":
                res_event.std_errors[
                    f"rice_evt_x_fr_{name}"
                ],

            "Heterogeneity p":
                res_event.pvalues[
                    f"rice_evt_x_fr_{name}"
                ]
        })


    event_table = (
        pd.DataFrame(event_results)
          .sort_values("Event")
    )


    print("\n" + "=" * 100)
    print("EVENT-STUDY RESULTS")
    print("Reference period: event = -2")
    print("=" * 100)

    print(
        event_table
        .round(3)
        .to_string(index=False)
    )

    # =============================================================================
    # 19. JOINT PRE-TREND TESTS
    # =============================================================================


    def joint_wald_test(result, variables):

        param_names = list(
            result.params.index
        )

        R = np.zeros(
            (
                len(variables),
                len(param_names)
            )
        )

        for row, var in enumerate(variables):

            col = param_names.index(var)

            R[row, col] = 1


        test = result.wald_test(R)

        return {
            "Statistic": test.stat,
            "p-value": test.pval
        }


    # -----------------------------------------------------------------------------
    # Standard DID pre-trend
    # -----------------------------------------------------------------------------

    pretrend_treatment = joint_wald_test(
        res_event,
        [
            "rice_evt_m9",
            "rice_evt_m6"
        ]
    )


    # -----------------------------------------------------------------------------
    # Friction heterogeneity pre-trend
    # -----------------------------------------------------------------------------

    pretrend_friction = joint_wald_test(
        res_event,
        [
            "rice_evt_x_fr_m9",
            "rice_evt_x_fr_m6"
        ]
    )


    print("\n" + "=" * 70)
    print("JOINT PRE-TREND TESTS")
    print("=" * 70)


    print("\nTreatment pre-trend:")

    print(
        f"Wald statistic = "
        f"{pretrend_treatment['Statistic']:.3f}"
    )

    print(
        f"p-value = "
        f"{pretrend_treatment['p-value']:.3f}"
    )


    print("\nFriction heterogeneity pre-trend:")

    print(
        f"Wald statistic = "
        f"{pretrend_friction['Statistic']:.3f}"
    )

    print(
        f"p-value = "
        f"{pretrend_friction['p-value']:.3f}"
    )

    # =============================================================================
    # 20. DYNAMIC TREATMENT EFFECTS BY FRICTION LEVEL
    # =============================================================================


    dynamic_results = []


    for friction_level in [-1, 0, 1]:

        for name, event_time in event_map.items():

            base_var = f"rice_evt_{name}"
            interaction_var = (
                f"rice_evt_x_fr_{name}"
            )

            b = res_event.params
            V = res_event.cov

            effect = (
                b[base_var]
                + friction_level
                * b[interaction_var]
            )

            variance = (
                V.loc[base_var, base_var]
                + friction_level**2
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

            se = np.sqrt(variance)

            critical = student_t.ppf(
                0.975,
                res_event.df_resid
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
                    effect - critical * se,

                "CI_high":
                    effect + critical * se
            })


    dynamic_table = pd.DataFrame(
        dynamic_results
    )


    print("\n" + "=" * 85)
    print("DYNAMIC MGPP EFFECTS BY FRICTION LEVEL")
    print("=" * 85)

    print(
        dynamic_table
        .round(3)
        .to_string(index=False)
    )

    # =============================================================================
    # 21. EVENT-STUDY FIGURE BY MARKET FRICTION
    #    (points only, no connecting lines)
    # =============================================================================

    import matplotlib.pyplot as plt
    from pathlib import Path


    # Add omitted reference period: event = -2
    reference_rows = pd.DataFrame({
        "Friction": [-1, 0, 1],
        "Event": [-2, -2, -2],
        "Effect": [0, 0, 0],
        "SE": [0, 0, 0],
        "CI_low": [0, 0, 0],
        "CI_high": [0, 0, 0]
    })

    plot_data = pd.concat(
        [dynamic_table, reference_rows],
        ignore_index=True
    ).sort_values(["Friction", "Event"])


    # Output folder
    OUTPUTDIR = Path(outputdir)

    OUTPUTDIR.mkdir(
        parents=True,
        exist_ok=True
    )


    fig, ax = plt.subplots(figsize=(9, 6))

    labels = {
        -1: "Low friction (-1 SD)",
         0: "Mean friction",
         1: "High friction (+1 SD)"
    }

    # X offset to avoid overlap at same event time
    offsets = {
        -1: -0.18,
         0:  0.00,
         1:  0.18
    }


    for friction_level in [-1, 0, 1]:

        temp = plot_data[
            plot_data["Friction"] == friction_level
        ].copy()

        # reference period and non-reference period separately
        temp_ref = temp[temp["Event"] == -2].copy()
        temp_nonref = temp[temp["Event"] != -2].copy()

        # shifted x positions
        x_nonref = temp_nonref["Event"] + offsets[friction_level]
        x_ref = temp_ref["Event"] + offsets[friction_level]

        # asymmetric confidence interval for non-reference periods
        yerr = np.vstack([
            temp_nonref["Effect"] - temp_nonref["CI_low"],
            temp_nonref["CI_high"] - temp_nonref["Effect"]
        ])

        # points + CI only (no lines)
        ax.errorbar(
            x_nonref,
            temp_nonref["Effect"],
            yerr=yerr,
            fmt="o",
            capsize=4,
            linestyle="none",
            label=labels[friction_level]
        )

        # reference period point only
        ax.scatter(
            x_ref,
            temp_ref["Effect"],
            marker="D",
            s=55
        )


    # Horizontal zero line
    ax.axhline(
        0,
        linewidth=1,
        linestyle="--"
    )

    # Vertical line at policy start
    ax.axvline(
        0,
        linewidth=1,
        linestyle=":"
    )


    # Annotate the omitted reference period
    ax.annotate(
        "Reference period\n(event = -2)",
        xy=(-2, 0),
        xytext=(-3.6, 0.55),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10
    )


    ax.set_xticks([-9, -6, -2, 0, 3, 5])

    ax.set_xlabel(
        "Years relative to MGPP implementation"
    )

    ax.set_ylabel(
        "Effect on asinh household farm hours"
    )

    ax.set_title(
        "Dynamic Effects of MGPP by Pre-policy Market Friction"
    )

    ax.legend()

    fig.tight_layout()

    # Save
    fig.savefig(
        OUTPUTDIR / "event_study_friction_points.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTPUTDIR / "event_study_friction_points.pdf",
        bbox_inches="tight"
    )

    plt.show()

    results = {
        "event_map": event_map,
        "event_sample": event_sample,
        "event_xvars": event_xvars,
        "res_event": res_event,
        "reg_event": reg_event,
        "event_table": event_table,
        "dynamic_table": dynamic_table,
        "pretrend_treatment": pretrend_treatment,
        "pretrend_friction": pretrend_friction,
        "OUTPUTDIR": OUTPUTDIR,
    }

    return hh, results


