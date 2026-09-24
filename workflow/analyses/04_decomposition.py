if __package__ in (None, ""):
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from workflow._deps import *
else:
    from .._deps import *

def run_farm_decomposition(hh, run_fe, Xdemog):
    """
    CELL 7.
    Run the supporting event-study decomposition:
      1) number of family farm workers
      2) farm hours per worker

    Requires only CELL 1 -> CELL 2 -> CELL 3.
    """
    hh = hh.copy()

    event_map = {
        "m9": -9,
        "m6": -6,
        "p0": 0,
        "p3": 3,
        "p5": 5,
    }

    # =============================================================================
    # 26. FARM SUPPORTING DECOMPOSITION
    #
    # Exact counterpart of Stata PART 12:
    #
    # 1. Number of family farm workers
    # 2. Farm hours per worker
    #
    # Event-study specification only.
    # Reference period: event = -2
    # Household FE + wave FE
    # Clustered SE at COMMID
    # =============================================================================


    # -----------------------------------------------------------------------------
    # 26.1 CREATE SUPPORTING OUTCOMES
    # -----------------------------------------------------------------------------

    hh["asinh_hh_n_farm_workers"] = np.arcsinh(
        hh["hh_n_farm_workers"]
    )

    hh["asinh_hh_hours_per_worker"] = np.arcsinh(
        hh["hh_hours_per_worker"]
    )


    # Event-study variables
    support_event_vars = [
        "rice_evt_m9",
        "rice_evt_m6",
        "rice_evt_p0",
        "rice_evt_p3",
        "rice_evt_p5"
    ]


    support_xvars = (
        support_event_vars
        + Xdemog
    )


    # =============================================================================
    # 26.2 NUMBER OF FARM WORKERS
    #
    # Stata:
    #
    # reghdfe asinh_hh_n_farm_workers
    #     rice_evt_m9 rice_evt_m6
    #     rice_evt_p0 rice_evt_p3 rice_evt_p5
    #     $Xdemog
    #     if sample_hh_ctrl==1
    #     & !missing(asinh_hh_n_farm_workers),
    #     absorb(hhid wave)
    #     vce(cluster COMMID)
    # =============================================================================


    worker_sample = hh[
        hh["sample_hh_ctrl"].eq(1)
        & hh["asinh_hh_n_farm_workers"].notna()
    ].copy()


    res_workers, reg_workers = run_fe(
        data=worker_sample,
        yvar="asinh_hh_n_farm_workers",
        xvars=support_xvars
    )


    print("\n" + "=" * 85)
    print("FARM WORKER COUNT EVENT STUDY")
    print("=" * 85)

    print(res_workers.summary)

    # =============================================================================
    # 26.3 STATA-STYLE JOINT F TEST
    # =============================================================================

    from scipy.stats import f as f_dist


    def cluster_joint_f_test(
        result,
        variables,
        n_clusters
    ):

        names = list(
            result.params.index
        )

        q = len(variables)

        R = np.zeros(
            (
                q,
                len(names)
            )
        )

        for i, var in enumerate(variables):

            R[
                i,
                names.index(var)
            ] = 1


        beta = (
            result.params
            .to_numpy()
            .reshape(-1, 1)
        )

        V = (
            result.cov
            .to_numpy()
        )


        Rb = R @ beta

        RVRT = (
            R
            @ V
            @ R.T
        )


        # Wald statistic
        W = (
            Rb.T
            @ np.linalg.inv(RVRT)
            @ Rb
        ).item()


        # Convert to F statistic
        F_stat = W / q

        df1 = q
        df2 = n_clusters - 1


        p_value = (
            1
            - f_dist.cdf(
                F_stat,
                df1,
                df2
            )
        )


        return {
            "F": F_stat,
            "df1": df1,
            "df2": df2,
            "p-value": p_value
        }

    # =============================================================================
    # 26.4 WORKER COUNT PRETREND / POST TESTS
    # =============================================================================

    n_clusters_workers = (
        reg_workers["COMMID"]
        .nunique()
    )


    worker_pre = cluster_joint_f_test(
        result=res_workers,
        variables=[
            "rice_evt_m9",
            "rice_evt_m6"
        ],
        n_clusters=n_clusters_workers
    )


    worker_post = cluster_joint_f_test(
        result=res_workers,
        variables=[
            "rice_evt_p0",
            "rice_evt_p3",
            "rice_evt_p5"
        ],
        n_clusters=n_clusters_workers
    )


    print("\n" + "-" * 75)
    print("WORKER COUNT JOINT TESTS")
    print("-" * 75)

    print(
        f"Pre-trend: "
        f"F({worker_pre['df1']}, "
        f"{worker_pre['df2']}) = "
        f"{worker_pre['F']:.3f}, "
        f"p = {worker_pre['p-value']:.4f}"
    )

    print(
        f"Joint post: "
        f"F({worker_post['df1']}, "
        f"{worker_post['df2']}) = "
        f"{worker_post['F']:.3f}, "
        f"p = {worker_post['p-value']:.4f}"
    )


    print("\nSample:")
    print(
        "N =",
        res_workers.nobs
    )

    print(
        "Households =",
        reg_workers.index
        .get_level_values("hhid")
        .nunique()
    )

    print(
        "Communities =",
        reg_workers["COMMID"]
        .nunique()
    )

    # =============================================================================
    # 26.5 HOURS PER FARM WORKER EVENT STUDY
    #
    # Stata:
    #
    # reghdfe asinh_hh_hours_per_worker
    #     rice_evt_m9 rice_evt_m6
    #     rice_evt_p0 rice_evt_p3 rice_evt_p5
    #     $Xdemog
    #     if sample_hh_ctrl==1
    #     & hh_n_farm_workers>0
    #     & !missing(asinh_hh_hours_per_worker),
    #     absorb(hhid wave)
    #     vce(cluster COMMID)
    # =============================================================================


    intensive_sample = hh[
        hh["sample_hh_ctrl"].eq(1)
        & (hh["hh_n_farm_workers"] > 0)
        & hh["asinh_hh_hours_per_worker"].notna()
    ].copy()


    res_intensive, reg_intensive = run_fe(
        data=intensive_sample,
        yvar="asinh_hh_hours_per_worker",
        xvars=support_xvars
    )


    print("\n" + "=" * 85)
    print("FARM HOURS PER WORKER EVENT STUDY")
    print("=" * 85)

    print(res_intensive.summary)

    # =============================================================================
    # 26.6 HOURS PER WORKER PRETREND / POST TESTS
    # =============================================================================

    n_clusters_intensive = (
        reg_intensive["COMMID"]
        .nunique()
    )


    intensive_pre = cluster_joint_f_test(
        result=res_intensive,
        variables=[
            "rice_evt_m9",
            "rice_evt_m6"
        ],
        n_clusters=n_clusters_intensive
    )


    intensive_post = cluster_joint_f_test(
        result=res_intensive,
        variables=[
            "rice_evt_p0",
            "rice_evt_p3",
            "rice_evt_p5"
        ],
        n_clusters=n_clusters_intensive
    )


    print("\n" + "-" * 75)
    print("HOURS PER WORKER JOINT TESTS")
    print("-" * 75)


    print(
        f"Pre-trend: "
        f"F({intensive_pre['df1']}, "
        f"{intensive_pre['df2']}) = "
        f"{intensive_pre['F']:.3f}, "
        f"p = {intensive_pre['p-value']:.4f}"
    )


    print(
        f"Joint post: "
        f"F({intensive_post['df1']}, "
        f"{intensive_post['df2']}) = "
        f"{intensive_post['F']:.3f}, "
        f"p = {intensive_post['p-value']:.4f}"
    )


    print("\nSample:")

    print(
        "N =",
        res_intensive.nobs
    )

    print(
        "Households =",
        reg_intensive.index
        .get_level_values("hhid")
        .nunique()
    )

    print(
        "Communities =",
        reg_intensive["COMMID"]
        .nunique()
    )

    # =============================================================================
    # 26.7 SUPPORTING EVENT-STUDY TABLE
    # =============================================================================

    support_results = []


    for label, result in [

        (
            "Number of farm workers",
            res_workers
        ),

        (
            "Hours per farm worker",
            res_intensive
        )

    ]:

        for name, event_time in event_map.items():

            var = f"rice_evt_{name}"

            support_results.append({

                "Outcome":
                    label,

                "Event":
                    event_time,

                "Coef":
                    result.params[var],

                "SE":
                    result.std_errors[var],

                "p-value":
                    result.pvalues[var]
            })


    support_table = (
        pd.DataFrame(
            support_results
        )
    )


    print("\n" + "=" * 95)
    print("FARM SUPPORTING DECOMPOSITION: EVENT STUDY")
    print("Reference period = -2")
    print("=" * 95)

    print(
        support_table
        .round(3)
        .to_string(index=False)
    )

    return {
        "res_workers": res_workers,
        "reg_workers": reg_workers,
        "res_intensive": res_intensive,
        "reg_intensive": reg_intensive,
        "support_table": support_table,
        "worker_pre": worker_pre,
        "worker_post": worker_post,
        "intensive_pre": intensive_pre,
        "intensive_post": intensive_post,
    }


if __name__ == "__main__":
    from workflow.direct_run import run_pipeline_mode

    run_pipeline_mode("decomposition")
