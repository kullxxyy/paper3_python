from .._deps import *

def run_cluster_inference_and_bootstrap(
    baseline_results,
    event_results,
):
    """
    CELL 6.
    Run small-cluster inference and wild-cluster bootstrap.

    Requires results from CELL 4 and CELL 5.
    """

    res_friction = baseline_results["res_friction"]
    reg_friction = baseline_results["reg_friction"]
    friction_sample = baseline_results["friction_sample"]

    event_sample = event_results["event_sample"]
    event_xvars = event_results["event_xvars"]

    # =============================================================================
    # 22. SMALL-CLUSTER DEGREE-OF-FREEDOM CORRECTION
    # =============================================================================

    from scipy.stats import t as student_t


    def cluster_df_inference(
        result,
        variable,
        n_clusters
    ):

        beta = result.params[variable]

        se = result.std_errors[variable]

        t_stat = beta / se

        df_cluster = n_clusters - 1


        p_value = (
            2
            * (
                1
                - student_t.cdf(
                    abs(t_stat),
                    df_cluster
                )
            )
        )


        critical = student_t.ppf(
            0.975,
            df_cluster
        )


        ci_low = (
            beta
            - critical * se
        )

        ci_high = (
            beta
            + critical * se
        )


        return {
            "Coefficient": beta,
            "SE": se,
            "t": t_stat,
            "Cluster df": df_cluster,
            "p-value": p_value,
            "CI_low": ci_low,
            "CI_high": ci_high
        }


    n_clusters = (
        reg_friction["COMMID"]
        .nunique()
    )


    small_cluster_result = (
        cluster_df_inference(
            result=res_friction,
            variable="did_x_friction_core",
            n_clusters=n_clusters
        )
    )


    print("\n" + "=" * 75)
    print("SMALL-CLUSTER INFERENCE")
    print("=" * 75)

    print(
        pd.DataFrame(
            [small_cluster_result]
        )
        .round(3)
        .to_string(index=False)
    )

    # =============================================================================
    # 23. WILD CLUSTER BOOTSTRAP
    # =============================================================================

    from linearmodels.panel import PanelOLS


    def wild_cluster_bootstrap(
        data,
        yvar,
        xvars,
        target_var,
        cluster_var="COMMID",
        entity_var="hhid",
        time_var="wave",
        reps=999,
        seed=12345
    ):

        rng = np.random.default_rng(
            seed
        )


        # -------------------------------------------------------------------------
        # Regression sample
        # -------------------------------------------------------------------------

        cols = (
            [
                entity_var,
                time_var,
                cluster_var,
                yvar
            ]
            + xvars
        )


        reg = (
            data[cols]
            .dropna()
            .copy()
        )


        # Drop singleton households
        counts = (
            reg.groupby(entity_var)
            .size()
        )

        valid_entities = counts[
            counts >= 2
        ].index

        reg = reg[
            reg[entity_var]
            .isin(valid_entities)
        ].copy()


        reg = (
            reg.set_index(
                [
                    entity_var,
                    time_var
                ]
            )
            .sort_index()
        )


        clusters = reg[
            cluster_var
        ]


        # =====================================================================
        # 1. Unrestricted model
        # =====================================================================

        unrestricted = PanelOLS(
            reg[yvar],
            reg[xvars],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )


        res_unrestricted = unrestricted.fit(
            cov_type="clustered",
            clusters=clusters
        )


        beta_original = (
            res_unrestricted
            .params[target_var]
        )


        se_original = (
            res_unrestricted
            .std_errors[target_var]
        )


        t_original = (
            beta_original
            / se_original
        )


        # =====================================================================
        # 2. Restricted model under H0:
        # target coefficient = 0
        # =====================================================================

        restricted_vars = [
            x
            for x in xvars
            if x != target_var
        ]


        restricted = PanelOLS(
            reg[yvar],
            reg[restricted_vars],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )


        res_restricted = restricted.fit()


        # fitted value INCLUDING fixed effects
        # yhat = y - residual
        residual = (
            res_restricted.resids
        )


        fitted = (
            reg[yvar]
            - residual
        )


        # Unique clusters
        cluster_ids = (
            clusters
            .dropna()
            .unique()
        )


        bootstrap_t = []


        # =====================================================================
        # 3. Bootstrap repetitions
        # =====================================================================

        for b in range(reps):

            # Rademacher cluster weights:
            # -1 or +1
            weights = dict(

                zip(

                    cluster_ids,

                    rng.choice(
                        [-1, 1],
                        size=len(cluster_ids)
                    )

                )

            )


            cluster_weight = (
                clusters
                .map(weights)
            )


            # Wild bootstrap outcome
            y_boot = (
                fitted
                + residual
                * cluster_weight
            )


            boot_model = PanelOLS(
                y_boot,
                reg[xvars],
                entity_effects=True,
                time_effects=True,
                drop_absorbed=True
            )


            try:

                boot_result = boot_model.fit(
                    cov_type="clustered",
                    clusters=clusters
                )


                beta_boot = (
                    boot_result
                    .params[target_var]
                )


                se_boot = (
                    boot_result
                    .std_errors[target_var]
                )


                t_boot = (
                    beta_boot
                    / se_boot
                )


                bootstrap_t.append(
                    t_boot
                )


            except Exception:

                continue


        bootstrap_t = np.array(
            bootstrap_t
        )


        # =====================================================================
        # 4. Bootstrap p-value
        # =====================================================================

        p_boot = (

            np.sum(
                np.abs(bootstrap_t)
                >= abs(t_original)
            )
            + 1

        ) / (

            len(bootstrap_t)
            + 1

        )


        return {

            "Coefficient":
                beta_original,

            "Cluster SE":
                se_original,

            "t-stat":
                t_original,

            "Wild bootstrap p":
                p_boot,

            "Bootstrap reps":
                len(bootstrap_t),

            "Clusters":
                len(cluster_ids)
        }

    # =============================================================================
    # 24. WILD BOOTSTRAP: DID × FRICTION
    # =============================================================================


    bootstrap_vars = [

        "rice_did",

        "post_x_friction_core",

        "did_x_friction_core",

        "hhsize",

        "child_share",

        "elderly_share",

        "male_share"
    ]


    bootstrap_result = (
        wild_cluster_bootstrap(

            data=friction_sample,

            yvar=
            "asinh_hh_farm_hours",

            xvars=
            bootstrap_vars,

            target_var=
            "did_x_friction_core",

            cluster_var=
            "COMMID",

            reps=999,

            seed=12345
        )
    )


    print("\n" + "=" * 80)
    print("WILD CLUSTER BOOTSTRAP")
    print("Target: DID × Market Friction")
    print("=" * 80)


    print(

        pd.DataFrame(
            [bootstrap_result]
        )

        .round(3)

        .to_string(
            index=False
        )

    )

    # =============================================================================
    # 25. WILD BOOTSTRAP FOR EVENT-STUDY HETEROGENEITY
    # =============================================================================


    event_bootstrap_results = []


    for target in [

        "rice_evt_x_fr_p0",

        "rice_evt_x_fr_p3",

        "rice_evt_x_fr_p5"

    ]:

        result = wild_cluster_bootstrap(

            data=event_sample,

            yvar=
            "asinh_hh_farm_hours",

            xvars=
            event_xvars,

            target_var=
            target,

            cluster_var=
            "COMMID",

            reps=999,

            seed=12345
        )


        result[
            "Variable"
        ] = target


        event_bootstrap_results.append(
            result
        )


    event_bootstrap_table = (
        pd.DataFrame(
            event_bootstrap_results
        )
    )


    print("\n" + "=" * 95)
    print("WILD CLUSTER BOOTSTRAP: EVENT-STUDY HETEROGENEITY")
    print("=" * 95)


    print(

        event_bootstrap_table[
            [
                "Variable",
                "Coefficient",
                "Cluster SE",
                "t-stat",
                "Wild bootstrap p",
                "Clusters"
            ]
        ]

        .round(3)

        .to_string(
            index=False
        )

    )

    return {
        "small_cluster_result": small_cluster_result,
        "bootstrap_result": bootstrap_result,
        "event_bootstrap_table": event_bootstrap_table,
    }


