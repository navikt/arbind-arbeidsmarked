"""Map monthly county bankruptcies to NAV regions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import FYLKE_TIL_NAV
from src.common.validation import require_columns
from src.datagrunnlag.arbeidsmarkedsdata import normaliser_navn

_RAW = Path("data/raw")
_PROCESSED = Path("data/processed")


def main() -> None:
    """Persist native county and standardized NAV-region bankruptcy panels."""
    raw = require_columns(
        pd.read_csv(_RAW / "ssb_konkurser_fylke_maaned.csv", dtype={"fylkeskode": str}),
        ["fylkeskode", "fylke", "maaned", "foretakskonkurser"],
        source="ssb_konkurser_fylke_maaned.csv",
    )
    county_to_region = {
        normaliser_navn(county): region for county, region in FYLKE_TIL_NAV.items()
    }
    county_to_region["troms og finnmark"] = "Nav Troms og Finnmark"
    raw["nav_region"] = raw["fylke"].map(normaliser_navn).map(county_to_region)
    unmapped = raw.loc[raw["nav_region"].isna(), "fylke"].unique()
    if len(unmapped):
        print(
            "  Ikke aggregert til Nav-region: "
            f"{', '.join(sorted(unmapped))}. Viken kan ikke fordeles uten en kilde på lavere nivå."
        )
    result = (
        raw.dropna(subset=["nav_region"])
        .groupby(["maaned", "nav_region"], as_index=False)["foretakskonkurser"]
        .sum()
        .sort_values(["maaned", "nav_region"])
    )
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    raw.to_csv(_PROCESSED / "konkurser_fylke_maaned.csv", index=False)
    result.to_csv(_PROCESSED / "konkurser_nav_region_maaned.csv", index=False)
    print(f"-> {_PROCESSED / 'konkurser_nav_region_maaned.csv'} ({len(result)} rader)")


if __name__ == "__main__":
    main()
