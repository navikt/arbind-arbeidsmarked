"""Shared transformations for local labour-market data."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

_MAPPING = Path("data/mapping/kommune_kontor_fylke.json")
_MUNICIPALITY_ALIASES = {"valer (viken)": "valer (østfold)"}


def normaliser_navn(value: str) -> str:
    """Normalize SSB and NAV geographic labels to their Norwegian name."""
    primary = re.sub(r"\s+\(\d{4}-\d{4}\)$", "", value.split(" - ")[0]).strip()
    result = "".join(
        char
        for char in unicodedata.normalize("NFKD", primary.lower())
        if not unicodedata.combining(char)
    )
    return _MUNICIPALITY_ALIASES.get(result, result)


def les_kommunekart() -> pd.DataFrame:
    """Read one municipality-to-NAV-region row per mapped municipality."""
    with _MAPPING.open(encoding="utf-8") as source:
        items = json.load(source)["results"][0]["items"]
    mapping = pd.DataFrame(items).rename(
        columns={
            "navarende_kommune_navn": "kommune",
            "nav_region": "nav_region",
        }
    )
    mapping["kommune_nokkel"] = mapping["kommune"].map(normaliser_navn)
    conflicts = mapping.groupby("kommune_nokkel")["nav_region"].nunique()
    if (conflicts > 1).any():
        raise ValueError("En kommune kan ikke tilhøre flere Nav-regioner.")
    return mapping.drop_duplicates("kommune_nokkel")[["kommune_nokkel", "nav_region"]]


def beregn_arbeidsmarkedsmaal(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate annual municipality measures from source counts and rates."""
    required = {
        "kommunenummer",
        "kommune",
        "aar",
        "sysselsetting_bosted",
        "sysselsetting_arbeidssted",
        "privat_sysselsetting_arbeidssted",
        "sysselsettingsrate",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mangler kolonner for arbeidsmarkedsmål: {sorted(missing)}")

    result = df.sort_values(["kommunenummer", "aar"]).copy()
    if (result["sysselsetting_bosted"] <= 0).any():
        raise ValueError(
            "Kan ikke beregne arbeidsplassdekning med ikke-positiv bostedssysselsetting."
        )
    if (result["sysselsetting_arbeidssted"] <= 0).any():
        raise ValueError(
            "Kan ikke beregne privatandel med ikke-positiv arbeidsstedssysselsetting."
        )
    result["arbeidsplassdekning"] = (
        result["sysselsetting_arbeidssted"] / result["sysselsetting_bosted"]
    )
    result["andel_privat_sysselsetting"] = (
        result["privat_sysselsetting_arbeidssted"] / result["sysselsetting_arbeidssted"]
    )
    result["vekst_sysselsetting_bosted"] = result.groupby("kommunenummer")[
        "sysselsetting_bosted"
    ].pct_change()
    result["vekst_sysselsetting_arbeidssted"] = result.groupby("kommunenummer")[
        "sysselsetting_arbeidssted"
    ].pct_change()
    return result


def legg_til_nav_region(df: pd.DataFrame) -> pd.DataFrame:
    """Map municipality records to the region in the maintained NAV mapping."""
    result = df.copy()
    result["kommune_nokkel"] = result["kommune"].map(normaliser_navn)
    result = result.merge(
        les_kommunekart(), on="kommune_nokkel", how="left", validate="many_to_one"
    )
    unknown = result.loc[result["nav_region"].isna(), "kommune"].unique()
    if len(unknown):
        raise ValueError(f"Kommuner uten Nav-region i mappingen: {sorted(unknown)}")
    return result.drop(columns="kommune_nokkel")
