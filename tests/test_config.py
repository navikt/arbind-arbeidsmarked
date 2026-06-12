"""Tester for sentraliserte antakelser i src/common/config.py."""

from __future__ import annotations

from src.common.config import (
    FYLKE_TIL_NAV,
    NAV_REGIONER,
    REFERANSEMAANED,
    REGION_TIL_NAV,
)


def test_referansemaaned_keys_and_values_are_valid() -> None:
    assert set(REFERANSEMAANED) == {2021, 2022, 2023, 2024, 2025}
    assert all(1 <= m <= 12 for m in REFERANSEMAANED.values())


def test_region_til_nav_values_are_known_nav_regions() -> None:
    assert set(REGION_TIL_NAV.values()) <= set(NAV_REGIONER)
    # 2021–2023 dekker alle Nav-regioner direkte.
    assert set(REGION_TIL_NAV.values()) == set(NAV_REGIONER)


def test_fylke_til_nav_values_are_known_nav_regions() -> None:
    assert set(FYLKE_TIL_NAV.values()) <= set(NAV_REGIONER)
    # Fylkesaggregeringen skal også dekke alle Nav-regioner.
    assert set(FYLKE_TIL_NAV.values()) == set(NAV_REGIONER)


def test_nav_regioner_are_unique_and_prefixed() -> None:
    assert len(NAV_REGIONER) == len(set(NAV_REGIONER))
    assert all(r.startswith("Nav ") for r in NAV_REGIONER)
