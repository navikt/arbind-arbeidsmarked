"""Standardize annual municipal SSB labour-market measures to NAV regions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.validation import require_columns, require_no_nulls
from src.datagrunnlag.arbeidsmarkedsdata import (
    beregn_arbeidsmarkedsmaal,
    legg_til_nav_region,
)

_RAW = Path("data/raw")
_PROCESSED = Path("data/processed")


def main() -> None:
    """Calculate measures and attach each municipality's NAV region."""
    raw = require_columns(
        pd.read_csv(
            _RAW / "ssb_arbeidsmarkedsmaal_kommune.csv", dtype={"kommunenummer": str}
        ),
        [
            "kommunenummer",
            "kommune",
            "aar",
            "sysselsetting_bosted",
            "sysselsetting_arbeidssted",
            "privat_sysselsetting_arbeidssted",
            "sysselsettingsrate",
        ],
        source="ssb_arbeidsmarkedsmaal_kommune.csv",
    )
    # Four-digit historical region codes are returned with zero values by the
    # SSB API. They are not current municipality observations.
    raw = raw[
        (raw["sysselsetting_bosted"] > 0) & (raw["sysselsetting_arbeidssted"] > 0)
    ].copy()
    municipality = legg_til_nav_region(beregn_arbeidsmarkedsmaal(raw))
    municipality = require_no_nulls(
        municipality, ["nav_region"], source="arbeidsmarkedsmål"
    )
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    municipality.to_csv(_PROCESSED / "arbeidsmarkedsmaal_kommune.csv", index=False)
    print(
        f"-> {_PROCESSED / 'arbeidsmarkedsmaal_kommune.csv'} ({len(municipality)} rader)"
    )


if __name__ == "__main__":
    main()
