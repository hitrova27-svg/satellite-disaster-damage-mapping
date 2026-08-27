from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import censusdata

YEAR = 2017
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY")

ACS_VARS = [
    "B19013_001E",  # median household income
    "B17020_001E",  # poverty universe total
    "B17020_002E",  # below poverty
    "B25077_001E",  # median home value
    "B25003_001E",  # total housing units
    "B25003_003E",  # renter occupied units
    "B25035_001E",  # median year structure built
    "B01003_001E",  # population total
]


def get_geography(state_fips: str):
    return censusdata.censusgeo(
        [
            ("state", state_fips),
            ("county", "*"),
            ("tract", "*"),
            ("block group", "*"),
        ]
    )


def build_acs_df() -> pd.DataFrame:
    if not CENSUS_API_KEY:
        raise RuntimeError(
            "Census API key is required for ACS block-group downloads. "
            "Set the CENSUS_API_KEY environment variable or pass it explicitly to censusdata.download()."
        )

    rows = []

    for state in [f"{i:02d}" for i in range(1, 57)]:
        try:
            table = censusdata.download(
                "acs5",
                YEAR,
                get_geography(state),
                ACS_VARS,
                key=CENSUS_API_KEY,
            )
        except Exception as exc:
            print(f"Skipping state {state}: {exc}")
            continue

        if table.empty:
            continue

        for idx, row in table.iterrows():
            geo_parts = dict(idx.geo)
            geoid = (
                f"{geo_parts['state']}"
                f"{geo_parts['county']}"
                f"{geo_parts['tract']}"
                f"{geo_parts['block group']}"
            )

            income = pd.to_numeric(row["B19013_001E"], errors="coerce")
            pov_total = pd.to_numeric(row["B17020_001E"], errors="coerce")
            pov_below = pd.to_numeric(row["B17020_002E"], errors="coerce")
            home_value = pd.to_numeric(row["B25077_001E"], errors="coerce")
            housing_total = pd.to_numeric(row["B25003_001E"], errors="coerce")
            renter_total = pd.to_numeric(row["B25003_003E"], errors="coerce")
            med_year = pd.to_numeric(row["B25035_001E"], errors="coerce")
            pop = pd.to_numeric(row["B01003_001E"], errors="coerce")

            rows.append(
                {
                    "GEOID": geoid,
                    "income_2017": income,
                    "pov_rate_2017": (pov_below / pov_total) if pd.notna(pov_total) and pov_total else np.nan,
                    "median_home_value_2017": home_value,
                    "renter_pct_2017": (renter_total / housing_total) if pd.notna(housing_total) and housing_total else np.nan,
                    "median_year_built_2017": med_year,
                    "pop_density_2017": pop,
                }
            )

    if not rows:
        raise ValueError(
            "ACS query returned zero rows. This means the Census API is rejecting the request for this environment. "
            "The most common cause is a missing Census API key."
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["GEOID"]).sort_values("GEOID").reset_index(drop=True)
    return df[[
        "GEOID",
        "income_2017",
        "pov_rate_2017",
        "median_home_value_2017",
        "renter_pct_2017",
        "median_year_built_2017",
        "pop_density_2017",
    ]]


def build_exposure_df(acs_df: pd.DataFrame) -> pd.DataFrame:
    exposure = acs_df[["GEOID"]].copy()
    exposure["exposure_proxy"] = 0.0
    return exposure


def main() -> None:
    acs_df = build_acs_df()
    acs_path = DATA_DIR / "acs.csv"
    acs_df.to_csv(acs_path, index=False)

    exposure_df = build_exposure_df(acs_df)
    exposure_path = DATA_DIR / "hazard_exposure.csv"
    exposure_df.to_csv(exposure_path, index=False)

    print(f"Saved ACS CSV: {acs_path} ({len(acs_df):,} rows)")
    print(f"Saved hazard exposure CSV: {exposure_path} ({len(exposure_df):,} rows)")
    print(acs_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
