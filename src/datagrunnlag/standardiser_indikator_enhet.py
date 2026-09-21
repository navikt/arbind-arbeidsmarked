"""Standardize monthly office-level work-indicator observations to a wide panel.

This is a data-foundation output only. Existing analysis inputs remain unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.validation import require_columns, require_no_nulls

_RAW = Path("data/raw")
_PROCESSED = Path("data/processed")


def standardiser_indikator_enhet(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the four outcomes and their metrics to one office-month observation."""
    required = [
        "beholdningsmaaned",
        "org_sted",
        "utfall",
        "indikator",
        "forventet",
        "faktisk",
        "antall_personer",
        "nedbrytning",
    ]
    source = require_columns(df, required, source="indikator_data_enhet.csv")
    source = source[source["nedbrytning"] == "Alle"].copy()
    source["beholdningsmaaned"] = pd.to_datetime(source["beholdningsmaaned"], utc=True)
    duplicate_keys = source.duplicated(["beholdningsmaaned", "org_sted", "utfall"])
    if duplicate_keys.any():
        raise ValueError(
            "Indikatordata inneholder flere observasjoner per enhet, måned og utfall."
        )
    result = source.pivot(
        index=["beholdningsmaaned", "org_sted"],
        columns="utfall",
        values=["indikator", "forventet", "faktisk", "antall_personer"],
    )
    result.columns = [f"{metric}_{outcome}" for metric, outcome in result.columns]
    return (
        result.reset_index()
        .sort_values(["beholdningsmaaned", "org_sted"])
        .reset_index(drop=True)
    )


def main() -> None:
    """Read the office extract and write its standardized monthly panel."""
    result = standardiser_indikator_enhet(
        pd.read_csv(_RAW / "indikator_data_enhet.csv")
    )
    result = require_no_nulls(
        result,
        ["beholdningsmaaned", "org_sted"],
        source="arbeidsindikator_enhet_maaned",
    )
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    destination = _PROCESSED / "arbeidsindikator_enhet_maaned.csv"
    result.to_csv(destination, index=False)
    print(f"-> {destination} ({len(result)} rader)")


if __name__ == "__main__":
    main()
