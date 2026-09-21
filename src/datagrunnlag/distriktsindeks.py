"""Pure calculations for Distriktsindeksen 2025."""

from __future__ import annotations

import pandas as pd

_WEIGHTS = {
    "sentralitet": 0.40,
    "befolkningsvekst": 0.40,
    "sysselsettingsvekst": 0.10,
    "herfindahl_omvendt": 0.10,
}


def beregn_herfindahl(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate private-sector industry concentration per municipality."""
    required = {"kommunenummer", "sysselsatte"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mangler kolonner for Herfindahl: {sorted(missing)}")
    totals = df.groupby("kommunenummer")["sysselsatte"].transform("sum")
    if (totals <= 0).any():
        raise ValueError("Kan ikke beregne Herfindahl med ikke-positiv sysselsetting.")
    shares = df["sysselsatte"] / totals
    return (
        df.assign(_andel_kvadrert=shares**2)
        .groupby("kommunenummer", as_index=False)["_andel_kvadrert"]
        .sum()
        .rename(columns={"_andel_kvadrert": "herfindahl"})
        .assign(herfindahl_omvendt=lambda x: 1 - x["herfindahl"])
    )


def beregn_distriktsindeks(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize, truncate, weight, and scale the four index components."""
    missing = set(_WEIGHTS) - set(df.columns)
    if missing:
        raise ValueError(f"Mangler indekskomponenter: {sorted(missing)}")
    result = df.copy()
    weighted = pd.Series(0.0, index=result.index)
    for column, weight in _WEIGHTS.items():
        std = result[column].std(ddof=1)
        if pd.isna(std) or std == 0:
            raise ValueError(f"Kan ikke standardisere {column}: standardavvik er null.")
        normalized = (result[column] - result[column].mean()) / std
        result[f"{column}_z"] = normalized
        result[f"{column}_trunkert"] = normalized.clip(-2.5, 2.5)
        weighted += result[f"{column}_trunkert"] * weight

    result["distriktsindeks_standardisert"] = weighted
    low, high = weighted.min(), weighted.max()
    if low == high:
        raise ValueError("Kan ikke indeksere når alle vektede skårer er like.")
    result["distriktsindeks"] = (weighted - low) * 100 / (high - low)
    return result
