from ._deps import *

def prepare_data(
    datadir=Path("/Users/lenovo/PhD papers/Paper 3_new2/data/CHNS_data_analysis"),
    logdir=Path("/Users/lenovo/PhD papers/Paper 3_new2/python_code/logs"),
):
    """
    CELL 1.
    Load CHNS data and construct farm outcomes, main sample, policy variables,
    event-study variables, and household demographic controls.

    Returns a dictionary containing df, waves, pre_waves, events, and Xdemog.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # =============================================================================
    # LOG FILE
    # =============================================================================

    import sys
    import atexit
    from pathlib import Path
    from datetime import datetime

    class Tee:
        """
        Print output to both:
        1. PyCharm console
        2. log file
        """

        def __init__(self, *streams):
            self.streams = streams

        def write(self, text):
            for stream in self.streams:
                stream.write(text)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    # Log folder
    LOGDIR = Path(logdir)

    LOGDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Same file overwritten every run
    LOGFILE = LOGDIR / "paper3_python.log"

    # Open log
    _log_file = open(
        LOGFILE,
        "w",
        encoding="utf-8",
        buffering=1
    )

    # Save original console
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr

    # Console + log simultaneously
    sys.stdout = Tee(
        _original_stdout,
        _log_file
    )

    sys.stderr = Tee(
        _original_stderr,
        _log_file
    )

    def close_log():

        sys.stdout = _original_stdout
        sys.stderr = _original_stderr

        _log_file.close()

    atexit.register(close_log)

    print("=" * 80)
    print("PAPER 3 PYTHON LOG")
    print("=" * 80)
    print(f"Log file: {LOGFILE}")
    print(
        "Run time:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    print("=" * 80)


    # =============================================================================
    # 1. LOAD DATA
    # =============================================================================

    DATADIR = Path(datadir)

    df = pd.read_stata(
        DATADIR / "paper3_data.dta",
        convert_categoricals=False
    )


    # Study waves
    waves = [1997, 2000, 2004, 2006, 2009, 2011]
    pre_waves = [1997, 2000, 2004]

    df = df[df["wave"].isin(waves)].copy()


    # Standardize variable names
    if "T1" in df.columns and "t1" not in df.columns:
        df.rename(columns={"T1": "t1"}, inplace=True)

    if "COMMID" not in df.columns and "commid" in df.columns:
        df.rename(columns={"commid": "COMMID"}, inplace=True)

    elif "COMMID" in df.columns and "commid" in df.columns:
        df["COMMID"] = df["COMMID"].fillna(df["commid"])


    # Hunan = treatment
    # Guizhou = control
    df = df[df["t1"].isin([43, 52])].copy()

    df["rice_treated"] = (
        df["t1"] == 43
    ).astype(int)


    # =============================================================================
    # 2. INDIVIDUAL FARM HOURS
    # =============================================================================

    missing_codes = [
        -9, -88, -99, -999, -9999,
        999, 9999
    ]


    df["farming_days"] = pd.to_numeric(
        df["E4B"],
        errors="coerce"
    ).replace(
        missing_codes,
        np.nan
    )


    df["farming_hours"] = pd.to_numeric(
        df["E4C"],
        errors="coerce"
    ).replace(
        missing_codes,
        np.nan
    )


    # Logical range
    df["farming_days"] = df["farming_days"].where(
        df["farming_days"].between(0, 7)
    )

    df["farming_hours"] = df["farming_hours"].where(
        df["farming_hours"].between(0, 24)
    )


    # Farm participation
    df["farm_participation"] = (
        df["E2A"]
        .where(df["E2A"].isin([0, 1]))
    )


    d = df["farming_days"]
    h = df["farming_hours"]


    # Response patterns
    pair_complete = (
        d.notna()
        & h.notna()
    )

    partial_missing = (
        d.isna()
        ^ h.isna()
    )

    both_missing = (
        d.isna()
        & h.isna()
    )

    zero_component = (
        d.eq(0)
        | h.eq(0)
    )

    positive_component = (
        d.gt(0)
        | h.gt(0)
    )


    # E2A says no farming,
    # but E4 suggests positive farming
    conflict = (

        df["farm_participation"].eq(0)

        & (

            (
                pair_complete
                & (d * h > 0)
            )

            |

            (
                partial_missing
                & positive_component
                & ~zero_component
            )

        )
    )


    # Weekly farm hours
    df["farm_weekly_hours"] = d * h


    # Explicit zero
    df.loc[
        df["farm_weekly_hours"].isna()
        & zero_component,
        "farm_weekly_hours"
    ] = 0


    # E2A confirms nonparticipation
    df.loc[
        df["farm_weekly_hours"].isna()
        & df["farm_participation"].eq(0)
        & ~conflict,
        "farm_weekly_hours"
    ] = 0


    # Both E4 variables missing:
    # treat as zero unless E2A confirms farming
    df.loc[
        both_missing
        & df["farm_participation"].ne(1)
        & ~conflict,
        "farm_weekly_hours"
    ] = 0


    # Conflict stays missing
    df.loc[
        conflict,
        "farm_weekly_hours"
    ] = np.nan


    df["asinh_farm_weekly_hours"] = np.arcsinh(
        df["farm_weekly_hours"]
    )


    df["farm_worker"] = np.where(
        df["farm_weekly_hours"].notna(),
        (df["farm_weekly_hours"] > 0).astype(int),
        np.nan
    )


    # =============================================================================
    # 3. HOUSEHOLD FARM OUTCOMES
    # =============================================================================

    g = df.groupby(
        ["hhid", "wave"],
        sort=False
    )


    df["farm_missing"] = (
        df["farm_weekly_hours"]
        .isna()
        .astype(int)
    )


    df["hh_farm_missing"] = (
        g["farm_missing"]
        .transform("max")
    )


    df["hh_any_farm_worker"] = (
        g["farm_worker"]
        .transform("max")
    )


    df["hh_farm_weekly_hours"] = (
        g["farm_weekly_hours"]
        .transform(
            lambda x: x.sum(min_count=1)
        )
    )


    # Do not use incomplete household totals
    df.loc[
        df["hh_farm_missing"].eq(1),
        "hh_farm_weekly_hours"
    ] = np.nan


    df["asinh_hh_farm_hours"] = np.arcsinh(
        df["hh_farm_weekly_hours"]
    )


    df["hh_n_farm_workers"] = (
        g["farm_worker"]
        .transform(
            lambda x: x.sum(min_count=1)
        )
    )


    df.loc[
        df["hh_farm_missing"].eq(1),
        "hh_n_farm_workers"
    ] = np.nan


    df["hh_hours_per_worker"] = (
        df["hh_farm_weekly_hours"]
        / df["hh_n_farm_workers"]
    ).where(
        df["hh_n_farm_workers"] > 0
    )


    # =============================================================================
    # 4. MAIN SAMPLE
    # =============================================================================

    df["pre_farm"] = (
        df["hh_any_farm_worker"]
        .where(
            df["wave"].isin(pre_waves)
        )
    )


    df["pre_policy_farm_hh"] = (
        df.groupby("hhid")["pre_farm"]
        .transform("max")
    )


    df["sample_main"] = (
        df["pre_policy_farm_hh"] == 1
    ).astype(int)


    # =============================================================================
    # 5. POLICY VARIABLES
    # =============================================================================

    df["outcome_year"] = (
        df["wave"] - 1
    )


    df["rice_post"] = (
        df["outcome_year"] >= 2005
    ).astype(int)


    df["rice_did"] = (
        df["rice_treated"]
        * df["rice_post"]
    )


    df["rice_event"] = (
        df["outcome_year"] - 2005
    )


    # Event -2 = reference period
    events = {
        "m9": -9,
        "m6": -6,
        "p0": 0,
        "p3": 3,
        "p5": 5
    }


    for name, event_time in events.items():

        df[f"rice_evt_{name}"] = (

            df["rice_treated"]

            * (
                df["rice_event"]
                == event_time
            )

        ).astype(int)


    # =============================================================================
    # 6. HOUSEHOLD CONTROLS
    # =============================================================================

    valid_age = df["age"].between(
        0,
        110
    )


    df["child"] = np.where(
        valid_age,
        df["age"] < 18,
        np.nan
    )


    df["elderly"] = np.where(
        valid_age,
        df["age"] >= 65,
        np.nan
    )


    df["male"] = np.where(
        df["GENDER"].isin([1, 2]),
        df["GENDER"] == 1,
        np.nan
    )


    g = df.groupby(
        ["hhid", "wave"],
        sort=False
    )


    for var in [
        "child",
        "elderly",
        "male"
    ]:

        df[f"n_{var}"] = (
            g[var]
            .transform("sum")
        )

        df[f"{var}_share"] = (

            df[f"n_{var}"]
            / df["hhsize"]

        ).where(
            df["hhsize"] > 0
        )


    Xdemog = [
        "hhsize",
        "child_share",
        "elderly_share",
        "male_share"
    ]

    # Same as Stata: controls_complete
    df["controls_complete"] = (
        df[Xdemog]
        .notna()
        .all(axis=1)
        .astype(int)
    )

    return {
        "df": df,
        "waves": waves,
        "pre_waves": pre_waves,
        "events": events,
        "Xdemog": Xdemog,
    }

