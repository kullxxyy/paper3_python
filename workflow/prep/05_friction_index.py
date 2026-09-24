if __package__ in (None, ""):
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from workflow._deps import *
else:
    from .._deps import *

def build_friction_indices(df, pre_waves):
    """
    CELL 2.
    Construct core equal-weight, core PCA, rich equal-weight, and rich PCA
    pre-policy market-friction measures, then merge them back to df.

    Returns:
        df_out, metadata
    """
    df = df.copy()

    # Make this function safe to rerun on an already prepared df.
    rerun_cols = [
        "pre_econ",
        "pre_trans",
        "pre_comm",
        "friction_core",
        "friction_pca",
        "friction_equal_rich",
        "friction_pca_rich",
    ]
    existing = [c for c in rerun_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    # =============================================================================
    # 7. ORIGINAL EQUAL-WEIGHT MARKET FRICTION INDEX
    # =============================================================================

    friction_vars = [
        "econ",
        "trans",
        "comm"
    ]


    # -----------------------------------------------------------------------------
    # Step 1:
    # individual data -> community × wave
    # -----------------------------------------------------------------------------

    cw = (

        df.groupby(
            ["COMMID", "wave"],
            as_index=False
        )[friction_vars]

        .mean()

    )


    # Keep pre-policy years
    pre = cw[
        cw["wave"].isin(pre_waves)
    ].copy()


    # -----------------------------------------------------------------------------
    # Step 2:
    # pre-policy community averages
    #
    # Require at least 2 of 3 pre-policy waves
    # -----------------------------------------------------------------------------

    community = pd.DataFrame(
        {
            "COMMID":
            pre["COMMID"]
            .dropna()
            .unique()
        }
    )


    community = community.set_index(
        "COMMID"
    )


    for var in friction_vars:

        temp = (
            pre.groupby("COMMID")[var]
            .agg(["mean", "count"])
        )

        community[f"pre_{var}"] = (
            temp["mean"]
            .where(
                temp["count"] >= 2
            )
        )


    community = community.reset_index()


    # -----------------------------------------------------------------------------
    # Step 3:
    # Communities represented in main estimation sample
    # -----------------------------------------------------------------------------

    main_communities = (

        df.loc[
            df["sample_main"].eq(1),
            "COMMID"
        ]

        .dropna()
        .unique()

    )


    community["in_main_sample"] = (
        community["COMMID"]
        .isin(main_communities)
    )


    # -----------------------------------------------------------------------------
    # Step 4:
    # Standardize using main-sample communities
    # -----------------------------------------------------------------------------

    for var in friction_vars:

        reference = community.loc[
            community["in_main_sample"]
            & community[f"pre_{var}"].notna(),
            f"pre_{var}"
        ]

        mean = reference.mean()
        sd = reference.std(ddof=1)

        community[f"z_pre_{var}"] = (
            community[f"pre_{var}"]
            - mean
        ) / sd


    # Equal-weight integration index
    zvars = [
        "z_pre_econ",
        "z_pre_trans",
        "z_pre_comm"
    ]


    community["integration_raw"] = (

        community[zvars]
        .mean(axis=1)

        .where(
            community[zvars]
            .notna()
            .all(axis=1)
        )

    )


    # Standardize composite using main sample
    reference = community.loc[
        community["in_main_sample"]
        & community["integration_raw"].notna(),
        "integration_raw"
    ]


    community["integration_core"] = (

        community["integration_raw"]
        - reference.mean()

    ) / reference.std(ddof=1)


    # Higher = more friction
    community["friction_core"] = (
        -community["integration_core"]
    )


    # =============================================================================
    # 8. PCA MARKET FRICTION INDEX
    # =============================================================================

    features = [
        "pre_econ",
        "pre_trans",
        "pre_comm"
    ]


    # Communities used to estimate PCA
    fit_sample = (

        community["in_main_sample"]

        & community[features]
        .notna()
        .all(axis=1)

    )


    # Communities where PCA score can be calculated
    complete_sample = (

        community[features]
        .notna()
        .all(axis=1)

    )


    # -----------------------------------------------------------------------------
    # Standardize
    # -----------------------------------------------------------------------------

    scaler = StandardScaler()


    X_fit = scaler.fit_transform(
        community.loc[
            fit_sample,
            features
        ]
    )


    # -----------------------------------------------------------------------------
    # PCA
    # -----------------------------------------------------------------------------

    pca = PCA(
        n_components=1
    )

    pca.fit(X_fit)


    # Calculate PCA score for all complete communities
    community.loc[
        complete_sample,
        "pca_score"
    ] = pca.transform(

        scaler.transform(
            community.loc[
                complete_sample,
                features
            ]
        )

    )[:, 0]


    # -----------------------------------------------------------------------------
    # Orient PCA:
    # higher score = better market integration
    # -----------------------------------------------------------------------------

    corr = np.corrcoef(

        community.loc[
            fit_sample,
            "pca_score"
        ],

        community.loc[
            fit_sample,
            "integration_raw"
        ]

    )[0, 1]


    if corr < 0:

        community["pca_score"] = (
            -community["pca_score"]
        )


    # -----------------------------------------------------------------------------
    # Standardize PCA score
    # -----------------------------------------------------------------------------

    reference = community.loc[
        fit_sample,
        "pca_score"
    ]


    community["integration_pca"] = (

        community["pca_score"]
        - reference.mean()

    ) / reference.std(ddof=1)


    # Higher = more friction
    community["friction_pca"] = (
        -community["integration_pca"]
    )


    # =============================================================================
    # 9. MERGE FRICTION INDICES BACK
    # =============================================================================

    df = df.merge(

        community[
            [
                "COMMID",
                "pre_econ",
                "pre_trans",
                "pre_comm",
                "friction_core",
                "friction_pca"
            ]
        ],

        on="COMMID",
        how="left"

    )

    # =============================================================================
    # 10. CHECK RESULTS
    # =============================================================================

    print("\n" + "=" * 60)
    print("PCA-BASED MARKET FRICTION INDEX")
    print("=" * 60)


    # PCA loadings
    pca_result = pd.DataFrame({
        "Variable": features,
        "Loading": pca.components_[0]
    })

    print("\nPCA loadings")
    print(
        pca_result
        .round(3)
        .to_string(index=False)
    )


    # Explained variance
    print("\nExplained variance")
    print(
        f"PC1: {pca.explained_variance_ratio_[0]:.3f}"
    )


    # Correlation
    corr_friction = (
        community[
            ["friction_core", "friction_pca"]
        ]
        .corr()
        .iloc[0, 1]
    )

    print("\nCorrelation between indices")
    print(
        f"Equal-weight vs PCA: {corr_friction:.3f}"
    )


    # Descriptive statistics
    print("\nDescriptive statistics")

    print(
        community[
            ["friction_core", "friction_pca"]
        ]
        .describe()
        .round(3)
    )


    print("\n" + "=" * 60)
    print("FINISHED")
    print("=" * 60)

    # =============================================================================
    # 8. RICH PCA-BASED MARKET FRICTION INDEX
    # =============================================================================


    # More comprehensive pre-policy market environment
    friction_vars = [
        "econ",
        "trans",
        "comm",
        "market",
        "mart",
        "denc"
    ]

    pre_waves = [1997, 2000, 2004]


    # =============================================================================
    # 8.1 COMMUNITY × WAVE DATA
    # =============================================================================

    cw = (
        df.groupby(
            ["COMMID", "wave"],
            as_index=False
        )[friction_vars]
        .mean()
    )


    # Pre-policy only
    pre = cw[
        cw["wave"].isin(pre_waves)
    ].copy()


    # =============================================================================
    # 8.2 PRE-POLICY COMMUNITY AVERAGES
    # =============================================================================

    community = pd.DataFrame(
        {"COMMID": pre["COMMID"].dropna().unique()}
    )

    community = community.set_index("COMMID")


    for var in friction_vars:

        temp = (
            pre.groupby("COMMID")[var]
            .agg(["mean", "count"])
        )

        # Require at least 2 pre-policy waves
        community[f"pre_{var}"] = (
            temp["mean"]
            .where(temp["count"] >= 2)
        )


    community = community.reset_index()


    # =============================================================================
    # 8.3 MAIN-SAMPLE COMMUNITIES
    # =============================================================================

    main_communities = (
        df.loc[
            df["sample_main"] == 1,
            "COMMID"
        ]
        .dropna()
        .unique()
    )


    community["in_main_sample"] = (
        community["COMMID"]
        .isin(main_communities)
    )


    # =============================================================================
    # 8.4 PCA INPUT VARIABLES
    # =============================================================================

    features = [
        "pre_econ",
        "pre_trans",
        "pre_comm",
        "pre_market",
        "pre_mart",
        "pre_denc"
    ]


    # Communities used to estimate PCA
    fit_sample = (
        community["in_main_sample"]
        & community[features].notna().all(axis=1)
    )


    # Communities for which a score can be generated
    complete_sample = (
        community[features]
        .notna()
        .all(axis=1)
    )


    # =============================================================================
    # 8.5 STANDARDIZE VARIABLES
    # =============================================================================

    scaler = StandardScaler()


    X_fit = scaler.fit_transform(
        community.loc[
            fit_sample,
            features
        ]
    )


    # Standardized values for all complete communities
    X_all = scaler.transform(
        community.loc[
            complete_sample,
            features
        ]
    )


    # =============================================================================
    # 8.6 EQUAL-WEIGHT RICH INDEX
    # =============================================================================

    # Mean standardized market environment
    community.loc[
        complete_sample,
        "integration_equal_rich"
    ] = X_all.mean(axis=1)


    # =============================================================================
    # 8.7 PCA
    # =============================================================================

    pca = PCA(
        n_components=1
    )

    pca.fit(X_fit)


    community.loc[
        complete_sample,
        "pca_score_rich"
    ] = (
        pca.transform(X_all)[:, 0]
    )


    # =============================================================================
    # 8.8 ORIENT THE PCA SCORE
    #
    # Higher PC1 should mean better market integration
    # =============================================================================

    corr = np.corrcoef(

        community.loc[
            fit_sample,
            "pca_score_rich"
        ],

        community.loc[
            fit_sample,
            "integration_equal_rich"
        ]

    )[0, 1]


    if corr < 0:

        community["pca_score_rich"] = (
            -community["pca_score_rich"]
        )


    # =============================================================================
    # 8.9 STANDARDIZE PCA INDEX
    # =============================================================================

    reference = community.loc[
        fit_sample,
        "pca_score_rich"
    ]


    community["integration_pca_rich"] = (

        community["pca_score_rich"]
        - reference.mean()

    ) / reference.std(ddof=1)


    # Reverse sign:
    # higher value = MORE market friction
    community["friction_pca_rich"] = (
        -community["integration_pca_rich"]
    )


    # Equal-weight rich friction index
    reference_equal = community.loc[
        fit_sample,
        "integration_equal_rich"
    ]


    community["friction_equal_rich"] = -(

        community["integration_equal_rich"]
        - reference_equal.mean()

    ) / reference_equal.std(ddof=1)


    # =============================================================================
    # 8.10 MERGE BACK INTO MAIN DATA
    # =============================================================================

    # Avoid duplicate columns if code is rerun
    for var in [
        "friction_pca_rich",
        "friction_equal_rich"
    ]:
        if var in df.columns:
            df.drop(columns=var, inplace=True)


    df = df.merge(

        community[
            [
                "COMMID",
                "friction_pca_rich",
                "friction_equal_rich"
            ]
        ],

        on="COMMID",
        how="left"
    )


    # =============================================================================
    # 8.11 RESULTS
    # =============================================================================

    print("\n" + "=" * 65)
    print("RICH PCA MARKET FRICTION INDEX")
    print("=" * 65)


    # PCA loadings
    loading_table = pd.DataFrame({

        "Variable": features,

        "Loading":
            pca.components_[0]

    })


    print("\nPCA loadings")

    print(
        loading_table
        .round(3)
        .to_string(index=False)
    )


    # Explained variance
    print(
        "\nVariance explained by PC1:"
    )

    print(
        f"{pca.explained_variance_ratio_[0]:.3f}"
    )


    # Correlation between rich equal-weight and PCA
    corr_rich = (
        community[
            [
                "friction_equal_rich",
                "friction_pca_rich"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )


    print(
        "\nCorrelation: Equal-weight rich vs PCA rich"
    )

    print(
        f"{corr_rich:.3f}"
    )


    # Main-sample descriptive statistics
    print(
        "\nMain-sample descriptive statistics"
    )

    print(

        community.loc[
            community["in_main_sample"],
            [
                "friction_equal_rich",
                "friction_pca_rich"
            ]
        ]

        .describe()
        .round(3)

    )

    print("\n" + "=" * 65)
    print("FINISHED")
    print("=" * 65)

    compare = (
        df[
            [
                "COMMID",
                "friction_core",
                "friction_pca",
                "friction_equal_rich",
                "friction_pca_rich"
            ]
        ]
        .drop_duplicates("COMMID")
    )

    print("\nCorrelation across friction measures")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print(
        compare[
            [
                "friction_core",
                "friction_pca",
                "friction_equal_rich",
                "friction_pca_rich"
            ]
        ]
        .corr()
        .round(3)
    )

    metadata = {
        "community": community,
        "compare": compare,
        "pca_result": pca_result,
        "loading_table": loading_table,
    }

    return df, metadata


if __name__ == "__main__":
    from workflow.direct_run import run_preparation_step

    run_preparation_step("friction")
