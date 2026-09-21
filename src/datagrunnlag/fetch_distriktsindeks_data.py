"""Hent SSB-dataene som trengs for å beregne Distriktsindeksen 2025.

Henter befolkning (07459), sysselsetting etter arbeidssted (07984) og
privat sysselsetting etter tosifret næring (13470). Sentralitet ekstraheres fra
Kommunal- og distriktsdepartementets offisielle 2025-beregningsfil, som gjengir
SSBs sentralitetsverdier for kommuneinndelingen som brukes i indeksen.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from src.datagrunnlag.fetch_ssb_data import _fetch_json_stat2, _json_stat2_to_df

_RAW = Path("data/raw")
_API = "https://data.ssb.no/api/pxwebapi/v2/tables"
_OFFISIELL_FIL = (
    "https://www.regjeringen.no/contentassets/c004ffa1cda7474d8e5628b55d4e2e29/"
    "distriktsindeksen-2025.xlsx"
)


def _municipal_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep four-digit municipality codes and preserve leading zeros."""
    result = df[df["Region_code"].str.fullmatch(r"\d{4}", na=False)].copy()
    result["kommunenummer"] = result["Region_code"].str.zfill(4)
    result["kommunenavn"] = result["Region_label"]
    return result


def _fetch_population() -> pd.DataFrame:
    url = (
        f"{_API}/07459/data?lang=no&outputFormat=json-stat2"
        "&valuecodes[Region]=*&valuecodes[Kjonn]=*&valuecodes[Alder]=*"
        "&valuecodes[ContentsCode]=Personer1&valuecodes[Tid]=2015,2025"
        "&heading=Region,Tid&stub=Kjonn,Alder,ContentsCode"
    )
    df = _municipal_rows(_json_stat2_to_df(_fetch_json_stat2(url)))
    return (
        df.groupby(["kommunenummer", "kommunenavn", "Tid_code"], as_index=False)[
            "value"
        ]
        .sum()
        .rename(columns={"Tid_code": "aar"})
    )


def _fetch_employment() -> pd.DataFrame:
    url = (
        f"{_API}/07984/data?lang=no&outputFormat=json-stat2"
        "&valuecodes[Region]=*&valuecodes[NACE2007]=00-99"
        "&valuecodes[Kjonn]=0&valuecodes[Alder]=15-74"
        "&valuecodes[ContentsCode]=SysselsatteArb&valuecodes[Tid]=2014,2024"
        "&heading=Region,Tid&stub=NACE2007,Kjonn,Alder,ContentsCode"
    )
    df = _municipal_rows(_json_stat2_to_df(_fetch_json_stat2(url)))
    return df[["kommunenummer", "kommunenavn", "Tid_code", "value"]].rename(
        columns={"Tid_code": "aar"}
    )


def _fetch_industries() -> pd.DataFrame:
    metadata_url = f"{_API}/13470/metadata?lang=no"
    with urllib.request.urlopen(metadata_url) as response:
        metadata = json.loads(response.read().decode("utf-8"))
    industry_codes = [
        code
        for code in metadata["dimension"]["NACE2007"]["category"]["index"]
        if code.isdigit() and len(code) == 2 and code not in {"00", "99"}
    ]
    url = (
        f"{_API}/13470/data?lang=no&outputFormat=json-stat2"
        f"&valuecodes[Region]=*&valuecodes[NACE2007]={','.join(industry_codes)}"
        "&valuecodes[ContentsCode]=SysselsatteArb&valuecodes[Tid]=2024"
        "&heading=Region,NACE2007&stub=ContentsCode,Tid"
    )
    df = _municipal_rows(_json_stat2_to_df(_fetch_json_stat2(url)))
    df = df[df["NACE2007_code"].str.fullmatch(r"\d{2}", na=False)].copy()
    return df.rename(
        columns={
            "NACE2007_code": "naering",
            "NACE2007_label": "naering_navn",
            "value": "sysselsatte",
        }
    )[["kommunenummer", "kommunenavn", "naering", "naering_navn", "sysselsatte"]]


def _fetch_centrality() -> pd.DataFrame:
    """Extract the published 2025 centrality input from the official workbook."""
    target = _RAW / "distriktsindeksen-2025-offisiell.xlsx"
    request = urllib.request.Request(
        _OFFISIELL_FIL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request) as response, target.open("wb") as output:
        output.write(response.read())
    raw = pd.read_excel(target, sheet_name="Rådata", dtype={"KNR": str})
    return raw.rename(
        columns={
            "KNR": "kommunenummer",
            "KNAVN": "kommunenavn",
            "SSB Sentralitet": "sentralitet",
        }
    )[["kommunenummer", "kommunenavn", "sentralitet"]].assign(
        kommunenummer=lambda x: x["kommunenummer"].str.zfill(4)
    )


def main() -> None:
    """Fetch and save each source extract."""
    _RAW.mkdir(parents=True, exist_ok=True)
    for name, fetcher in {
        "distriktsindeks_befolkning.csv": _fetch_population,
        "distriktsindeks_sysselsetting.csv": _fetch_employment,
        "distriktsindeks_naering.csv": _fetch_industries,
        "distriktsindeks_sentralitet.csv": _fetch_centrality,
    }.items():
        df = fetcher()
        df.to_csv(_RAW / name, index=False)
        print(f"-> {_RAW / name} ({len(df)} rader)")


if __name__ == "__main__":
    main()
