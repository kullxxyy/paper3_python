from ._deps import *

def build_household_panel(df):
    """
    CELL 3.
    Create one observation per household-wave, define estimation samples,
    and create the FE regression helper.

    Returns:
        hh, run_fe
    """
    # =============================================================================
    # 11. MAIN HOUSEHOLD PANEL SAMPLE
    # =============================================================================

    from linearmodels.panel import PanelOLS


    # -----------------------------------------------------------------------------
    # One observation per household × wave
    # Equivalent to Stata tag_hhwave == 1
    # -----------------------------------------------------------------------------

    hh = (
        df.sort_values(["hhid", "wave"])
          .drop_duplicates(["hhid", "wave"])
          .copy()
    )


    # -----------------------------------------------------------------------------
    # Exact Stata sample logic
    #
    # sample_hh_raw =
    # tag_hhwave==1
    # & sample_main==1
    # & !missing(asinh_hh_farm_hours)
    # & !missing(COMMID)
    # -----------------------------------------------------------------------------

    hh["sample_hh_raw"] = (
        hh["sample_main"].eq(1)
        & hh["asinh_hh_farm_hours"].notna()
        & hh["COMMID"].notna()
    ).astype(int)


    # Number of usable waves for each household
    hh["n_hh_obs"] = (
        hh.groupby("hhid")["sample_hh_raw"]
          .transform("sum")
    )


    # At least two usable observations
    hh["sample_hh"] = (
        hh["sample_hh_raw"].eq(1)
        & hh["n_hh_obs"].ge(2)
    ).astype(int)


    # Controlled sample
    hh["sample_hh_ctrl"] = (
        hh["sample_hh"].eq(1)
        & hh["controls_complete"].eq(1)
    ).astype(int)


    # Core-friction sample
    hh["sample_friction_hh"] = (
        hh["sample_hh_ctrl"].eq(1)
        & hh["friction_core"].notna()
    ).astype(int)


    # =============================================================================
    # SAMPLE CHECK
    # =============================================================================

    print("\n" + "=" * 70)
    print("HOUSEHOLD PANEL SAMPLE")
    print("=" * 70)

    print(
        f"sample_hh      = {hh['sample_hh'].sum()}"
    )

    print(
        f"sample_hh_ctrl = {hh['sample_hh_ctrl'].sum()}"
    )

    print(
        f"friction sample = {hh['sample_friction_hh'].sum()}"
    )

    print("\nSample by wave:")

    print(
        pd.crosstab(
            hh.loc[hh["sample_hh"] == 1, "wave"],
            hh.loc[hh["sample_hh"] == 1, "rice_treated"]
        )
    )

    # =============================================================================
    # 12. FIXED-EFFECT REGRESSION FUNCTION
    # =============================================================================


    def run_fe(data, yvar, xvars):
        """
        Household FE + wave FE
        Clustered SE at COMMID level.

        Singleton households are removed AFTER
        defining the actual regression sample.
        """

        reg = (
            data[
                [
                    "hhid",
                    "wave",
                    "COMMID",
                    yvar
                ] + xvars
            ]
            .dropna()
            .copy()
        )

        # -------------------------------------------------------------------------
        # Drop singleton households
        # This is closer to reghdfe behavior.
        # -------------------------------------------------------------------------

        while True:

            counts = (
                reg.groupby("hhid")
                   .size()
            )

            singleton_hh = counts[
                counts < 2
            ].index

            if len(singleton_hh) == 0:
                break

            reg = reg[
                ~reg["hhid"].isin(singleton_hh)
            ].copy()


        # Panel index
        reg = (
            reg.set_index(
                ["hhid", "wave"]
            )
            .sort_index()
        )


        model = PanelOLS(
            reg[yvar],
            reg[xvars],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )


        result = model.fit(
            cov_type="clustered",
            clusters=reg["COMMID"]
        )


        return result, reg

    return hh, run_fe


