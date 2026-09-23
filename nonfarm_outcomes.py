from ._deps import *

def build_nonfarm_outcomes(df):
    """
    Supporting data-construction function for CELL 8.

    Construct household primary-job nonfarm weekly hours.

    Definition:
      1. primary_hours_avgweek = primary-job weekly hours
      2. B2 == 0 -> not working -> zero nonfarm hours
      3. job == 5 -> farming primary occupation -> zero nonfarm hours
      4. job in 1,...,13 except 5 -> primary nonfarm worker
      5. Household outcome = sum of individual primary-job nonfarm hours

    Returns:
        df with individual and household nonfarm outcomes.
    """

    df = df.copy()


    # =============================================================================
    # 27. PRIMARY NONFARM LABOR OUTCOME
    # =============================================================================

    # -----------------------------------------------------------------------------
    # 27.1 PRIMARY OCCUPATION
    # -----------------------------------------------------------------------------

    if "job" not in df.columns:

        if "B4" in df.columns:

            df["job"] = df["B4"]

        else:

            raise ValueError(
                "Neither job nor B4 is available."
            )


    required_vars = [
        "B2",
        "job",
        "primary_hours_avgweek"
    ]


    missing_vars = [
        var
        for var in required_vars
        if var not in df.columns
    ]


    if missing_vars:

        raise ValueError(
            "Missing required nonfarm variables: "
            + ", ".join(missing_vars)
        )


    df["B2"] = pd.to_numeric(
        df["B2"],
        errors="coerce"
    )


    df["job"] = pd.to_numeric(
        df["job"],
        errors="coerce"
    )


    # -----------------------------------------------------------------------------
    # 27.2 PRIMARY-JOB WEEKLY HOURS
    # -----------------------------------------------------------------------------

    missing_codes = [
        -9,
        -88,
        -99,
        -999,
        -9999,
        999,
        9999
    ]


    df["primary_weekly_hours_main"] = (
        pd.to_numeric(
            df["primary_hours_avgweek"],
            errors="coerce"
        )
        .replace(
            missing_codes,
            np.nan
        )
    )


    # Logical range
    df["primary_weekly_hours_main"] = (
        df["primary_weekly_hours_main"]
        .where(
            df["primary_weekly_hours_main"]
            .between(0, 168)
        )
    )


    # -----------------------------------------------------------------------------
    # 27.3 PRIMARY NONFARM WORKER
    # -----------------------------------------------------------------------------

    df["primary_nonfarm_worker"] = np.nan


    # Not working
    df.loc[
        df["B2"].eq(0),
        "primary_nonfarm_worker"
    ] = 0


    # Primary occupation = farming
    df.loc[
        df["B2"].eq(1)
        & df["job"].eq(5),
        "primary_nonfarm_worker"
    ] = 0


    # Primary occupation = nonfarm
    df.loc[
        df["B2"].eq(1)
        & df["job"].between(1, 13)
        & ~df["job"].eq(5),
        "primary_nonfarm_worker"
    ] = 1


    # -----------------------------------------------------------------------------
    # 27.4 INDIVIDUAL PRIMARY NONFARM HOURS
    # -----------------------------------------------------------------------------

    df["primary_nonfarm_hours"] = np.nan


    # Not working -> zero
    df.loc[
        df["B2"].eq(0),
        "primary_nonfarm_hours"
    ] = 0


    # Farming primary occupation -> zero nonfarm hours
    df.loc[
        df["B2"].eq(1)
        & df["job"].eq(5),
        "primary_nonfarm_hours"
    ] = 0


    # Nonfarm primary occupation -> observed weekly hours
    df.loc[
        df["primary_nonfarm_worker"].eq(1)
        & df["primary_weekly_hours_main"].notna(),
        "primary_nonfarm_hours"
    ] = (
        df["primary_weekly_hours_main"]
    )


    # -----------------------------------------------------------------------------
    # 27.5 INCOMPLETE NONFARM INFORMATION
    # -----------------------------------------------------------------------------

    df["primary_nf_incomplete"] = 0


    # Working but occupation unavailable / invalid
    df.loc[
        df["B2"].eq(1)
        & (
            df["job"].isna()
            | ~df["job"].between(1, 13)
        ),
        "primary_nf_incomplete"
    ] = 1


    # Nonfarm worker but weekly hours missing
    df.loc[
        df["primary_nonfarm_worker"].eq(1)
        & df["primary_weekly_hours_main"].isna(),
        "primary_nf_incomplete"
    ] = 1


    # -----------------------------------------------------------------------------
    # 27.6 HOUSEHOLD × WAVE AGGREGATION
    # -----------------------------------------------------------------------------

    g = df.groupby(
        ["hhid", "wave"],
        sort=False
    )


    df["hh_primary_nonfarm_hours"] = (
        g["primary_nonfarm_hours"]
        .transform(
            lambda x: x.sum(min_count=1)
        )
    )


    df["hh_n_primary_nonfarm_workers"] = (
        g["primary_nonfarm_worker"]
        .transform(
            lambda x: x.sum(min_count=1)
        )
    )


    df["hh_primary_nf_incomplete"] = (
        g["primary_nf_incomplete"]
        .transform("max")
    )


    # -----------------------------------------------------------------------------
    # 27.7 ASINH TRANSFORMATION
    # -----------------------------------------------------------------------------

    df["asinh_hh_primary_nonfarm_hours"] = (
        np.arcsinh(
            df["hh_primary_nonfarm_hours"]
        )
    )


    # -----------------------------------------------------------------------------
    # 27.8 DIAGNOSTICS
    # -----------------------------------------------------------------------------

    hh_check = (
        df.sort_values(
            ["hhid", "wave"]
        )
        .drop_duplicates(
            ["hhid", "wave"]
        )
    )


    print("\n" + "=" * 80)
    print("PRIMARY NONFARM OUTCOME CONSTRUCTION")
    print("=" * 80)


    print("\nHousehold primary nonfarm hours:")

    print(
        hh_check[
            "hh_primary_nonfarm_hours"
        ]
        .describe()
        .round(3)
    )


    print("\nIncomplete household nonfarm information:")

    print(
        hh_check[
            "hh_primary_nf_incomplete"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


    return df


