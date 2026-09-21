"""Tests for local labour-market measure transformations."""

from __future__ import annotations

import pandas as pd
import pytest

from src.datagrunnlag.arbeidsmarkedsdata import (
    beregn_arbeidsmarkedsmaal,
    normaliser_navn,
)
from src.datagrunnlag.standardiser_indikator_enhet import standardiser_indikator_enhet


def _source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kommunenummer": ["0001", "0001", "0002", "0002"],
            "kommune": ["A", "A", "B", "B"],
            "aar": [2021, 2022, 2021, 2022],
            "sysselsetting_bosted": [100, 110, 100, 90],
            "sysselsetting_arbeidssted": [120, 132, 80, 72],
            "privat_sysselsetting_arbeidssted": [60, 66, 20, 18],
            "sysselsettingsrate": [70.0, 71.0, 60.0, 59.0],
        }
    )


def test_beregn_arbeidsmarkedsmaal_uses_counts_for_ratios_and_growth() -> None:
    result = beregn_arbeidsmarkedsmaal(_source_rows()).set_index(
        ["kommunenummer", "aar"]
    )

    assert result.loc[("0001", 2021), "arbeidsplassdekning"] == pytest.approx(1.2)
    assert result.loc[("0002", 2022), "andel_privat_sysselsetting"] == pytest.approx(
        0.25
    )
    assert result.loc[("0001", 2022), "vekst_sysselsetting_bosted"] == pytest.approx(
        0.1
    )
    assert result.loc[
        ("0002", 2022), "vekst_sysselsetting_arbeidssted"
    ] == pytest.approx(-0.1)


def test_beregn_arbeidsmarkedsmaal_rejects_zero_denominator() -> None:
    source = _source_rows()
    source.loc[0, "sysselsetting_bosted"] = 0

    with pytest.raises(ValueError, match="ikke-positiv"):
        beregn_arbeidsmarkedsmaal(source)


def test_normaliser_navn_uses_norwegian_part_of_multilingual_labels() -> None:
    assert normaliser_navn("Oslo - Oslove") == normaliser_navn("Oslo")


def test_standardiser_indikator_enhet_pivots_monthly_outcomes() -> None:
    source = pd.DataFrame(
        {
            "beholdningsmaaned": ["2025-03-01", "2025-03-01"],
            "org_sted": ["Nav A", "Nav A"],
            "utfall": ["jobb3", "atid3"],
            "indikator": [0.1, 0.2],
            "forventet": [0.4, 0.5],
            "faktisk": [0.5, 0.7],
            "antall_personer": [10, 20],
            "nedbrytning": ["Alle", "Alle"],
        }
    )

    result = standardiser_indikator_enhet(source)

    assert len(result) == 1
    assert result.loc[0, "faktisk_jobb3"] == pytest.approx(0.5)
    assert result.loc[0, "antall_personer_atid3"] == 20
