"""Beregn Distriktsindeksen 2025 fra SSB-rådata på kommunenivå.

Kilder: SSB 07459 (befolkning), 07984 (sysselsetting) og 13470 (privat
sysselsetting etter næring), samt sentralitet fra den offisielle
beregningsfilen. Indeksen bruker 2015–2025 befolkningsvekst, 2014–2024
sysselsettingsvekst, omvendt Herfindahl og sentralitet med vektene 40/40/10/10.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.validation import require_columns, require_no_nulls, require_nonempty
from src.datagrunnlag.distriktsindeks import beregn_distriktsindeks, beregn_herfindahl

_RAW = Path("data/raw")
_PROCESSED = Path("data/processed")


def _vekst(df: pd.DataFrame, start: int, end: int, name: str) -> pd.DataFrame:
    wide = df.pivot(index="kommunenummer", columns="aar", values="value")
    required = {start, end}
    if not required.issubset(wide.columns):
        raise ValueError(
            f"Mangler år {sorted(required - set(wide.columns))} for {name}."
        )
    return ((wide[end] - wide[start]) / wide[start] * 100).rename(name).reset_index()


def main() -> None:
    """Standardiser rådata og skriv komplett kommunedatasett."""
    population = require_columns(
        pd.read_csv(
            _RAW / "distriktsindeks_befolkning.csv", dtype={"kommunenummer": str}
        ),
        ["kommunenummer", "aar", "value"],
        source="distriktsindeks_befolkning.csv",
    )
    employment = require_columns(
        pd.read_csv(
            _RAW / "distriktsindeks_sysselsetting.csv", dtype={"kommunenummer": str}
        ),
        ["kommunenummer", "aar", "value"],
        source="distriktsindeks_sysselsetting.csv",
    )
    industries = require_columns(
        pd.read_csv(_RAW / "distriktsindeks_naering.csv", dtype={"kommunenummer": str}),
        ["kommunenummer", "sysselsatte"],
        source="distriktsindeks_naering.csv",
    )
    centrality = require_columns(
        pd.read_csv(
            _RAW / "distriktsindeks_sentralitet.csv", dtype={"kommunenummer": str}
        ),
        ["kommunenummer", "kommunenavn", "sentralitet"],
        source="distriktsindeks_sentralitet.csv",
    )

    herfindahl = beregn_herfindahl(require_nonempty(industries, source="næringsdata"))
    result = (
        centrality.merge(
            _vekst(population, 2015, 2025, "befolkningsvekst"), on="kommunenummer"
        )
        .merge(
            _vekst(employment, 2014, 2024, "sysselsettingsvekst"), on="kommunenummer"
        )
        .merge(herfindahl, on="kommunenummer")
    )
    result = require_no_nulls(
        result,
        ["kommunenummer", "kommunenavn", "sentralitet"],
        source="distriktsindeks",
    )
    result = (
        beregn_distriktsindeks(result)
        .sort_values("distriktsindeks")
        .reset_index(drop=True)
    )
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    result.to_csv(_PROCESSED / "distriktsindeks_2025.csv", index=False)
    print(f"-> {_PROCESSED / 'distriktsindeks_2025.csv'} ({len(result)} kommuner)")


if __name__ == "__main__":
    main()
