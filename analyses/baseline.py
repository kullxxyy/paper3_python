from .._deps import *

def run_baseline_and_friction(hh, run_fe, Xdemog):
    """
    CELL 4.
    Run baseline DID, core market-friction heterogeneity, robustness across
    alternative friction indices, and marginal treatment effects.

    Returns:
        hh_out, results
    """
    hh = hh.copy()

    # =============================================================================
    # 13. BASELINE DID
    # =============================================================================


    # -----------------------------------------------------------------------------
    # Model 1: DID without controls
    # sample_hh
    # -----------------------------------------------------------------------------

    base_noctrl = hh[
        hh["sample_hh"].eq(1)
    ].copy()


    res_base1, reg_base1 = run_fe(
        data=base_noctrl,
        yvar="asinh_hh_farm_hours",
        xvars=[
            "rice_did"
        ]
    )


    # -----------------------------------------------------------------------------
    # Model 2: DID with demographic controls
    # sample_hh_ctrl
    # -----------------------------------------------------------------------------

    base_ctrl = hh[
        hh["sample_hh_ctrl"].eq(1)
    ].copy()


    res_base2, reg_base2 = run_fe(
        data=base_ctrl,
        yvar="asinh_hh_farm_hours",
        xvars=[
            "rice_did",
            "hhsize",
            "child_share",
            "elderly_share",
            "male_share"
        ]
    )


    # =============================================================================
    # BASELINE RESULTS
    # =============================================================================

    print("\n" + "=" * 75)
    print("BASELINE DID RESULTS")
    print("=" * 75)


    baseline_table = pd.DataFrame({

        "Model": [
            "DID only",
            "DID + controls"
        ],

        "DID": [
            res_base1.params["rice_did"],
            res_base2.params["rice_did"]
        ],

        "SE": [
            res_base1.std_errors["rice_did"],
            res_base2.std_errors["rice_did"]
        ],

        "p-value": [
            res_base1.pvalues["rice_did"],
            res_base2.pvalues["rice_did"]
        ],

        "N": [
            res_base1.nobs,
            res_base2.nobs
        ],

        "Households": [
            reg_base1.index
            .get_level_values("hhid")
            .nunique(),

            reg_base2.index
            .get_level_values("hhid")
            .nunique()
        ],

        "Communities": [
            reg_base1["COMMID"].nunique(),
            reg_base2["COMMID"].nunique()
        ]
    })


    print(
        baseline_table
        .round(3)
        .to_string(index=False)
    )

    # =============================================================================
    # 14. CORE FRICTION HETEROGENEITY
    # =============================================================================


    hh["post_x_friction_core"] = (
        hh["rice_post"]
        * hh["friction_core"]
    )


    hh["did_x_friction_core"] = (
        hh["rice_did"]
        * hh["friction_core"]
    )


    friction_sample = hh[
        hh["sample_friction_hh"].eq(1)
    ].copy()


    res_friction, reg_friction = run_fe(
        data=friction_sample,
        yvar="asinh_hh_farm_hours",
        xvars=[
            "rice_did",
            "post_x_friction_core",
            "did_x_friction_core",
            "hhsize",
            "child_share",
            "elderly_share",
            "male_share"
        ]
    )


    print("\n" + "=" * 75)
    print("CORE FRICTION HETEROGENEITY")
    print("=" * 75)

    print(res_friction.summary)


    print("\nMain coefficients:")

    print(
        f"DID              = "
        f"{res_friction.params['rice_did']:.3f}"
    )

    print(
        f"Post × Friction  = "
        f"{res_friction.params['post_x_friction_core']:.3f}"
    )

    print(
        f"DID × Friction   = "
        f"{res_friction.params['did_x_friction_core']:.3f}"
    )

    print(
        f"Interaction SE    = "
        f"{res_friction.std_errors['did_x_friction_core']:.3f}"
    )

    print(
        f"Interaction p     = "
        f"{res_friction.pvalues['did_x_friction_core']:.3f}"
    )

    print(
        f"N                 = "
        f"{res_friction.nobs}"
    )

    print(
        "Households        =",
        reg_friction.index
        .get_level_values("hhid")
        .nunique()
    )

    print(
        "Communities       =",
        reg_friction["COMMID"]
        .nunique()
    )

    # =============================================================================
    # 15. ROBUSTNESS ACROSS FRICTION INDICES
    # =============================================================================


    friction_indices = [
        "friction_core",
        "friction_pca",
        "friction_equal_rich",
        "friction_pca_rich"
    ]


    results = []


    for friction in friction_indices:

        temp = hh.copy()


        # -------------------------------------------------------------------------
        # Same basic sample:
        # sample_hh_ctrl + friction available
        # -------------------------------------------------------------------------

        temp = temp[
            temp["sample_hh_ctrl"].eq(1)
            & temp[friction].notna()
        ].copy()


        # Interactions
        temp["post_x_friction"] = (
            temp["rice_post"]
            * temp[friction]
        )

        temp["did_x_friction"] = (
            temp["rice_did"]
            * temp[friction]
        )


        # FE regression
        res, reg = run_fe(
            data=temp,
            yvar="asinh_hh_farm_hours",
            xvars=[
                "rice_did",
                "post_x_friction",
                "did_x_friction",
                "hhsize",
                "child_share",
                "elderly_share",
                "male_share"
            ]
        )


        results.append({

            "Index":
                friction,

            "DID":
                res.params["rice_did"],

            "DID SE":
                res.std_errors["rice_did"],

            "DID × Friction":
                res.params["did_x_friction"],

            "Interaction SE":
                res.std_errors["did_x_friction"],

            "p-value":
                res.pvalues["did_x_friction"],

            "N":
                res.nobs,

            "Households":
                reg.index
                .get_level_values("hhid")
                .nunique(),

            "Communities":
                reg["COMMID"]
                .nunique()
        })


    results_table = pd.DataFrame(
        results
    )


    print("\n" + "=" * 105)
    print("ROBUSTNESS ACROSS FRICTION INDICES")
    print("=" * 105)


    print(
        results_table
        .round(3)
        .to_string(index=False)
    )

    # =============================================================================
    # 16. MARGINAL TREATMENT EFFECTS AT DIFFERENT FRICTION LEVELS
    # =============================================================================

    from scipy.stats import t as student_t


    def marginal_effect(
        result,
        friction_level,
        did_var="rice_did",
        interaction_var="did_x_friction_core"
    ):

        b = result.params
        V = result.cov

        # Treatment effect
        effect = (
            b[did_var]
            + friction_level * b[interaction_var]
        )

        # Variance of linear combination
        variance = (
            V.loc[did_var, did_var]
            + friction_level**2
            * V.loc[interaction_var, interaction_var]
            + 2 * friction_level
            * V.loc[did_var, interaction_var]
        )

        se = np.sqrt(variance)

        # t-statistic
        t_stat = effect / se

        df_resid = result.df_resid

        p_value = (
            2
            * (
                1
                - student_t.cdf(
                    abs(t_stat),
                    df_resid
                )
            )
        )

        critical = student_t.ppf(
            0.975,
            df_resid
        )

        lower = effect - critical * se
        upper = effect + critical * se

        return {
            "Friction": friction_level,
            "Effect": effect,
            "SE": se,
            "p-value": p_value,
            "CI_low": lower,
            "CI_high": upper
        }


    # -1 SD, mean, +1 SD
    marginal_results = []

    for f in [-1, 0, 1]:

        marginal_results.append(
            marginal_effect(
                res_friction,
                friction_level=f
            )
        )


    marginal_table = pd.DataFrame(
        marginal_results
    )


    print("\n" + "=" * 85)
    print("MARGINAL EFFECTS OF MGPP BY MARKET FRICTION")
    print("=" * 85)

    print(
        marginal_table
        .round(3)
        .to_string(index=False)
    )

    results = {
        "baseline_table": baseline_table,
        "res_base1": res_base1,
        "reg_base1": reg_base1,
        "res_base2": res_base2,
        "reg_base2": reg_base2,
        "friction_sample": friction_sample,
        "res_friction": res_friction,
        "reg_friction": reg_friction,
        "results_table": results_table,
        "marginal_table": marginal_table,
    }

    return hh, results


