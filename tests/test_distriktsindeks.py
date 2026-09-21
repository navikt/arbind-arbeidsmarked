"""Tester for beregningene i Distriktsindeksen."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.datagrunnlag import standardiser_distriktsindeks
from src.datagrunnlag.distriktsindeks import beregn_distriktsindeks, beregn_herfindahl


def test_beregn_herfindahl_summerer_kvadrerte_naeringsandeler() -> None:
    result = beregn_herfindahl(
        pd.DataFrame(
            {
                "kommunenummer": ["0001", "0001", "0002", "0002"],
                "sysselsatte": [75, 25, 50, 50],
            }
        )
    )

    assert result.set_index("kommunenummer").loc["0001", "herfindahl"] == pytest.approx(
        0.625
    )
    assert result.set_index("kommunenummer").loc[
        "0002", "herfindahl_omvendt"
    ] == pytest.approx(0.5)


def test_beregn_herfindahl_rejects_non_positive_totals() -> None:
    with pytest.raises(ValueError, match="ikke-positiv"):
        beregn_herfindahl(pd.DataFrame({"kommunenummer": ["0001"], "sysselsatte": [0]}))


def test_beregn_distriktsindeks_weights_and_scales_components() -> None:
    source = pd.DataFrame(
        {
            "kommunenummer": ["0001", "0002", "0003"],
            "sentralitet": [1, 2, 3],
            "befolkningsvekst": [1, 2, 3],
            "sysselsettingsvekst": [1, 2, 3],
            "herfindahl_omvendt": [1, 2, 3],
        }
    )

    result = beregn_distriktsindeks(source).set_index("kommunenummer")

    assert result.loc["0001", "distriktsindeks"] == pytest.approx(0)
    assert result.loc["0003", "distriktsindeks"] == pytest.approx(100)
    assert result.loc["0002", "distriktsindeks"] == pytest.approx(50)


def test_standardiser_distriktsindeks_writes_one_row_per_municipality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    pd.DataFrame(
        {
            "kommunenummer": ["0001", "0001", "0002", "0002", "0003", "0003"],
            "aar": [2015, 2025] * 3,
            "value": [100, 110, 100, 100, 100, 90],
        }
    ).to_csv(raw / "distriktsindeks_befolkning.csv", index=False)
    pd.DataFrame(
        {
            "kommunenummer": ["0001", "0001", "0002", "0002", "0003", "0003"],
            "aar": [2014, 2024] * 3,
            "value": [100, 110, 100, 100, 100, 90],
        }
    ).to_csv(raw / "distriktsindeks_sysselsetting.csv", index=False)
    pd.DataFrame(
        {
            "kommunenummer": ["0001", "0001", "0002", "0002", "0003", "0003"],
            "sysselsatte": [75, 25, 50, 50, 25, 75],
        }
    ).to_csv(raw / "distriktsindeks_naering.csv", index=False)
    pd.DataFrame(
        {
            "kommunenummer": ["0001", "0002", "0003"],
            "kommunenavn": ["A", "B", "C"],
            "sentralitet": [1, 2, 3],
        }
    ).to_csv(raw / "distriktsindeks_sentralitet.csv", index=False)
    monkeypatch.setattr(standardiser_distriktsindeks, "_RAW", raw)
    monkeypatch.setattr(standardiser_distriktsindeks, "_PROCESSED", processed)

    standardiser_distriktsindeks.main()

    result = pd.read_csv(processed / "distriktsindeks_2025.csv")
    assert len(result) == result["kommunenummer"].nunique() == 3
    assert result["distriktsindeks"].between(0, 100).all()
