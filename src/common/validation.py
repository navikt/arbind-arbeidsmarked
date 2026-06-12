"""Lette valideringssjekker for mellomliggende datasett.

Brukes ved innlesing av ``data/processed``-filer slik at manglende kolonner eller
tomme datasett feiler tydelig i stedet for å gi stille feil eller forvirrende
nedstrøms-feil (jf. CRITIQUE_METHOD_AND_CODE.md).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class DataValidationError(ValueError):
    """Reises når et innlest datasett ikke oppfyller forventede krav."""


def require_columns(
    df: pd.DataFrame, columns: Iterable[str], *, source: str
) -> pd.DataFrame:
    """Sjekk at ``df`` inneholder alle ``columns``.

    Args:
        df: Datasettet som valideres.
        columns: Kolonnenavn som må være til stede.
        source: Kilde (f.eks. filnavn) brukt i feilmeldingen.

    Returns:
        ``df`` uendret, for å muliggjøre kjeding.

    Raises:
        DataValidationError: Hvis én eller flere kolonner mangler.
    """
    mangler = [c for c in columns if c not in df.columns]
    if mangler:
        raise DataValidationError(
            f"{source}: mangler forventede kolonner {mangler}. Fant {list(df.columns)}."
        )
    return df


def require_nonempty(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Sjekk at ``df`` ikke er tomt.

    Raises:
        DataValidationError: Hvis datasettet ikke har noen rader.
    """
    if df.empty:
        raise DataValidationError(f"{source}: datasettet er tomt.")
    return df


def require_no_nulls(
    df: pd.DataFrame, columns: Iterable[str], *, source: str
) -> pd.DataFrame:
    """Sjekk at nøkkelkolonner ikke har manglende verdier.

    Raises:
        DataValidationError: Hvis en av ``columns`` inneholder null-verdier.
    """
    with_nulls = [c for c in columns if c in df.columns and df[c].isna().any()]
    if with_nulls:
        raise DataValidationError(
            f"{source}: nøkkelkolonner med manglende verdier: {with_nulls}."
        )
    return df
