"""Fetch monthly enterprise bankruptcies by county from SSB table 08551."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.datagrunnlag.fetch_ssb_data import _fetch_json_stat2, _json_stat2_to_df

_API = "https://data.ssb.no/api/pxwebapi/v2/tables"
_RAW = Path("data/raw")
_COUNTY_CODES = "03,11,15,18,30,31,32,33,34,39,40,42,46,50,54,55,56"


def hent_konkurser() -> pd.DataFrame:
    """Fetch all-industry enterprise bankruptcies from 2021 onwards."""
    url = (
        f"{_API}/08551/data?lang=no&outputFormat=json-stat2&valuecodes[Region]={_COUNTY_CODES}"
        "&valuecodes[NACE2007]=00-99&valuecodes[ContentsCode]=Foretakskonkurser"
        "&valuecodes[Tid]=*&heading=Region,Tid&stub=NACE2007,ContentsCode"
    )
    result = _json_stat2_to_df(_fetch_json_stat2(url))
    result = result[result["Tid_code"] >= "2021M01"].copy()
    return result.rename(
        columns={
            "Region_code": "fylkeskode",
            "Region_label": "fylke",
            "Tid_code": "maaned",
            "value": "foretakskonkurser",
        }
    )[["fylkeskode", "fylke", "maaned", "foretakskonkurser"]]


def main() -> None:
    """Fetch bankruptcy observations and write the raw county panel."""
    _RAW.mkdir(parents=True, exist_ok=True)
    result = hent_konkurser()
    destination = _RAW / "ssb_konkurser_fylke_maaned.csv"
    result.to_csv(destination, index=False)
    print(f"-> {destination} ({len(result)} rader)")


if __name__ == "__main__":
    main()
