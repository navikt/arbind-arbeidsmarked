"""Tester for rene parse-/transformasjonshjelpere i datagrunnlaget."""

from __future__ import annotations

import pandas as pd

from src.datagrunnlag.fetch_bedrifts_data import _clean_name, _safe_num
from src.datagrunnlag.standardiser_mismatch_data import _tilknytt_undersokelsesaar


def test_safe_num_parses_numbers() -> None:
    assert _safe_num(3) == 3.0
    assert _safe_num("4.5") == 4.5


def test_safe_num_returns_none_for_blanks_formulas_and_text() -> None:
    assert _safe_num(None) is None
    assert _safe_num("=SUM(A1:A2)") is None
    assert _safe_num("ikke et tall") is None


def test_clean_name_strips_leading_dash_and_whitespace() -> None:
    assert _clean_name("  - Industri ") == "Industri"
    assert _clean_name("Bygg og anlegg") == "Bygg og anlegg"
    assert _clean_name(None) == ""


def test_tilknytt_undersokelsesaar_maps_months_to_survey_year() -> None:
    df = pd.DataFrame(
        {
            "beholdningsmaaned": [
                "2021-01-15",  # før første referansemåned (feb 2021) -> 2021
                "2021-03-15",  # >= feb 2021 -> 2021
                "2022-03-15",  # før april 2022 -> 2021
                "2022-05-15",  # >= april 2022 -> 2022
                "2025-04-15",  # >= mars 2025 -> 2025
            ]
        }
    )
    result = _tilknytt_undersokelsesaar(df)
    assert list(result["undersokelsesaar"]) == [2021, 2021, 2021, 2022, 2025]
