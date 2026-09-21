"""Fetch the finest public municipality labour-market measures from SSB.

The current municipality register statistics are annual fourth-quarter data;
they must not be expanded to artificial monthly observations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.datagrunnlag.fetch_distriktsindeks_data import _municipal_rows
from src.datagrunnlag.fetch_ssb_data import _fetch_json_stat2, _json_stat2_to_df

_API = "https://data.ssb.no/api/pxwebapi/v2/tables"
_RAW = Path("data/raw")
_YEARS = "2021,2022,2023,2024,2025"


def _fetch(url: str) -> pd.DataFrame:
    return _municipal_rows(_json_stat2_to_df(_fetch_json_stat2(url)))


def hent_arbeidsmarkedsdata() -> pd.DataFrame:
    """Fetch source components for the five selected annual municipal measures."""
    employment = _fetch(
        f"{_API}/07984/data?lang=no&outputFormat=json-stat2"
        f"&valuecodes[Region]=*&valuecodes[NACE2007]=00-99&valuecodes[Kjonn]=0"
        f"&valuecodes[Alder]=15-74&valuecodes[ContentsCode]=Sysselsatte,SysselsatteArb"
        f"&valuecodes[Tid]={_YEARS}&heading=Region,ContentsCode,Tid&stub=NACE2007,Kjonn,Alder"
    )
    employment = (
        employment.pivot(
            index=["kommunenummer", "kommunenavn", "Tid_code"],
            columns="ContentsCode_code",
            values="value",
        )
        .reset_index()
        .rename(
            columns={
                "kommunenavn": "kommune",
                "Tid_code": "aar",
                "Sysselsatte": "sysselsetting_bosted",
                "SysselsatteArb": "sysselsetting_arbeidssted",
            }
        )
    )
    rates = _fetch(
        f"{_API}/06445/data?lang=no&outputFormat=json-stat2&valuecodes[Region]=*"
        f"&valuecodes[Kjonn]=0&valuecodes[Alder]=15-74&valuecodes[ContentsCode]=Sysselsatte"
        f"&valuecodes[Tid]={_YEARS}&heading=Region,Tid&stub=Kjonn,Alder,ContentsCode"
    ).rename(
        columns={
            "kommunenavn": "kommune",
            "Tid_code": "aar",
            "value": "sysselsettingsrate",
        }
    )
    private = _fetch(
        f"{_API}/13472/data?lang=no&outputFormat=json-stat2&valuecodes[Region]=*"
        f"&valuecodes[NACE2007]=00-99&valuecodes[Sektor]=A%2BB%2BD%2BE.0-1%2B9"
        f"&valuecodes[ContentsCode]=SysselEtterArbste&valuecodes[Tid]={_YEARS}"
        "&heading=Region,Tid&stub=NACE2007,Sektor,ContentsCode"
    ).rename(
        columns={
            "kommunenavn": "kommune",
            "Tid_code": "aar",
            "value": "privat_sysselsetting_arbeidssted",
        }
    )
    keys = ["kommunenummer", "kommune", "aar"]
    return employment.merge(
        rates[keys + ["sysselsettingsrate"]], on=keys, validate="one_to_one"
    ).merge(
        private[keys + ["privat_sysselsetting_arbeidssted"]],
        on=keys,
        validate="one_to_one",
    )


def main() -> None:
    """Fetch source components and write the raw municipality panel."""
    _RAW.mkdir(parents=True, exist_ok=True)
    result = hent_arbeidsmarkedsdata()
    destination = _RAW / "ssb_arbeidsmarkedsmaal_kommune.csv"
    result.to_csv(destination, index=False)
    print(f"-> {destination} ({len(result)} rader)")


if __name__ == "__main__":
    main()
