"""Fetches data from bedriftsundersøkelsen and saves it as csv files.

Downloads Excel tables from nav.no for years 2021-2025 and produces:
  - mangel_per_fylke.csv          : Estimert mangel etter fylke/region (Tabell 1)
  - mangel_per_yrkesgruppe.csv    : Estimert mangel etter yrkesgruppe (Tabell 2)
  - mangel_per_naring.csv         : Estimert mangel etter næring (Tabell 7/V1/5/6)
  - sysselsettingsbarometer.csv   : Nasjonal tidsserie 2003–2025 (Tabell 3/4)
  - sysselsettingsbarometer_naring.csv : Sysselsettingsbarometer etter næring (Tabell 4/5)
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pandas as pd

_URL = "https://www.nav.no/no/nav-og-samfunn/kunnskap/analyser-fra-nav/arbeid-og-velferd/arbeid-og-velferd/bedriftsundersokelsen"

_BASE_URL = "https://www.nav.no"

# Excel attachment paths per year (from nav.no page)
_EXCEL_URLS: dict[int, str] = {
    2021: "/_/attachment/download/fb0ed9fc-549a-4055-8210-4684c9f1a460:8eb0544f84e9011daf3b4838bbecd60d57be72ec/NAVs%20Bedriftsunders%C3%B8kelse%202021%20-%20Figurer%20og%20tabeller.xlsx",
    2022: "/_/attachment/download/68005875-7d3d-423f-b6fb-3065bc680011:f9f05b6f7ee24221c208302f61afc8bd0b7631ab/Bedriftsunders%C3%B8kelse%202022%20Figurer%20og%20tabeller.xlsx",
    2023: "/_/attachment/download/45e9b32f-56a3-4e58-8121-2f8b46304617:8604789b5ea6886a8af9c774ce731e5b0fdacd08/Bedriftsunders%C3%B8kelse%202023%20-%20Figurer%20og%20tabeller.xlsx",
    2024: "/_/attachment/download/059cfd7a-8b1c-4785-829d-892beb503d6a:6da16c8995c1eb06dfd141689ec7ed12364fc64e/NAV%20bedriftsunders%C3%B8king%202024%20Figurer%20og%20tabeller%20til%20publisering%20(1).xlsx",
    2025: "/_/attachment/download/a3e49df3-7ca5-44b9-8a3c-fe5391271edd:e9608defcb16e72c709a69d64314a4b7f37e216c/Navs%20bedriftsunders%C3%B8kelse%202025%20Figurer%20og%20tabeller.xlsx",
}

# Which sheet contains each table type per year
# (mangel_per_naring, sysselsettingsbarometer_naring)
_SHEET_MAP: dict[str, dict[int, str]] = {
    "mangel_fylke": {
        2021: "Tabell 1",
        2022: "Tabell 1",
        2023: "Tabell 1",
        2024: "Tabell 1",
        2025: "Tabell 1",
    },
    "mangel_yrke": {
        2021: "Tabell 2",
        2022: "Tabell 2",
        2023: "Tabell 2",
        2024: "Tabell 2",
        2025: "Tabell 2",
    },
    "barometer": {
        2021: "Tabell 3",
        2022: "Tabell 3",
        2023: "Tabell 3",
        2024: "Tabell 3",
        2025: "Tabell 4",
    },
    "barometer_naring": {
        2021: "Tabell 5",
        2022: "Tabell 4",
        2023: "Tabell 4",
        2024: "Tabell 4",
        2025: "Tabell 5",
    },
    "mangel_naring": {
        2021: "Tabell 7",
        2022: "Tabell V1",
        2023: "Tabell V1",
        2024: "Tabell 5",
        2025: "Tabell 6",
    },
}

_RAW_DIR = Path("data/raw/bedriftsundersokelsen")
_PROCESSED_DIR = Path("data/processed")


def download_excel_files() -> None:
    """Download Excel files for all years to the raw data directory."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year, path in _EXCEL_URLS.items():
        dest = _RAW_DIR / f"bedriftsundersokelsen_{year}.xlsx"
        if dest.exists():
            print(f"  {year}: already downloaded, skipping")
            continue
        url = _BASE_URL + path
        print(f"  {year}: downloading...")
        urllib.request.urlretrieve(url, dest)
        print(f"  {year}: saved to {dest}")


def _load_sheet(year: int, sheet: str) -> list[list[object]]:
    """Load a sheet from the Excel file for a given year."""
    import openpyxl

    wb = openpyxl.load_workbook(_RAW_DIR / f"bedriftsundersokelsen_{year}.xlsx")
    ws = wb[sheet]
    return [
        list(row)
        for row in ws.iter_rows(values_only=True)
        if any(c is not None for c in row)
    ]


def _safe_num(val: object) -> float | None:
    """Convert a cell value to float, returning None for formulas and blanks."""
    if val is None:
        return None
    if isinstance(val, str) and val.startswith("="):
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None


def _clean_name(name: str | None) -> str:
    """Normalize whitespace and strip leading dashes from category names."""
    if name is None:
        return ""
    return re.sub(r"^\s*-\s*", "", str(name).strip())


# ---------------------------------------------------------------------------
# Table 1 – Mangel per fylke/region
# ---------------------------------------------------------------------------


def _parse_mangel_fylke(year: int) -> pd.DataFrame:
    """Parse Tabell 1 (mangel på arbeidskraft etter fylke) for a given year."""
    rows = _load_sheet(year, _SHEET_MAP["mangel_fylke"][year])
    # Row 0: title, row 1: headers, rows 2+: data
    data = []
    for row in rows[2:]:
        region = _clean_name(
            row[0] if isinstance(row[0], (str, type(None))) else str(row[0])
        )
        if not region or region.lower() in ("totalt", "i alt"):
            continue
        data.append(
            {
                "aar": year,
                "region": region,
                "mangel_antall": _safe_num(row[1]),
                "ki_nedre": _safe_num(row[2]),
                "ki_ovre": _safe_num(row[3]),
                "stramhetsindikator": _safe_num(row[4]),
                "andel_alvorlige_rekrutteringsproblemer_pst": _safe_num(row[5]),
            }
        )
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Table 2 – Mangel per yrkesgruppe
# ---------------------------------------------------------------------------


def _parse_mangel_yrkesgruppe(year: int) -> pd.DataFrame:
    """Parse Tabell 2 (mangel etter yrkesgruppe) for a given year."""
    rows = _load_sheet(year, _SHEET_MAP["mangel_yrke"][year])
    data = []
    for row in rows[2:]:
        yrke = _clean_name(
            row[0] if isinstance(row[0], (str, type(None))) else str(row[0])
        )
        if not yrke or yrke.lower() in ("totalt", "i alt"):
            continue
        data.append(
            {
                "aar": year,
                "yrkesgruppe": yrke,
                "mangel_antall": _safe_num(row[1]),
                "ledige_og_tiltak": _safe_num(row[2]),
            }
        )
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Table 3 – Sysselsettingsbarometer (nasjonal tidsserie)
# ---------------------------------------------------------------------------


def _parse_barometer(year: int) -> pd.DataFrame:
    """Parse the national sysselsettingsbarometer time series."""
    rows = _load_sheet(year, _SHEET_MAP["barometer"][year])
    data = []
    for row in rows[2:]:
        if row[0] is None:
            continue
        try:
            survey_year = int(str(row[0]))
        except TypeError, ValueError:
            continue
        data.append(
            {
                "aar": survey_year,
                "nedgang_pst": _safe_num(row[1]),
                "uendret_pst": _safe_num(row[2]),
                "okning_pst": _safe_num(row[3]),
                "nettoandel": _safe_num(row[4]),
            }
        )
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Table 4/5 – Sysselsettingsbarometer etter næring
# ---------------------------------------------------------------------------


def _parse_barometer_naring(year: int) -> pd.DataFrame:
    """Parse the sysselsettingsbarometer by næring for a given year."""
    rows = _load_sheet(year, _SHEET_MAP["barometer_naring"][year])
    data = []
    for row in rows[2:]:
        naring = _clean_name(
            row[0] if isinstance(row[0], (str, type(None))) else str(row[0])
        )
        if not naring or naring.lower() in ("totalt", "i alt"):
            continue
        data.append(
            {
                "aar": year,
                "naring": naring,
                "nedgang_pst": _safe_num(row[1]),
                "uendret_pst": _safe_num(row[2]),
                "okning_pst": _safe_num(row[3]),
                "nettoandel": _safe_num(row[4]),
            }
        )
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Table 7/V1/5/6 – Mangel per næring
# ---------------------------------------------------------------------------


# The næring table in 2021 has sub-industry rows prefixed with "-".
# We include them with is_subnaring=True so callers can filter.
def _parse_mangel_naring(year: int) -> pd.DataFrame:
    """Parse the mangel per næring table for a given year."""
    rows = _load_sheet(year, _SHEET_MAP["mangel_naring"][year])
    data = []
    for row in rows[2:]:
        raw_name = row[0]
        if raw_name is None:
            continue
        raw_str = str(raw_name).strip()
        if raw_str.lower() in ("totalt", "i alt"):
            continue
        is_sub = raw_str.startswith("-")
        naring = _clean_name(raw_str)
        if not naring:
            continue
        data.append(
            {
                "aar": year,
                "naring": naring,
                "er_undernaring": is_sub,
                "mangel_antall": _safe_num(row[1]),
                "ki_nedre": _safe_num(row[2]),
                "ki_ovre": _safe_num(row[3]),
                "stramhetsindikator": _safe_num(row[4]),
                "andel_alvorlige_rekrutteringsproblemer_pst": _safe_num(row[5]),
            }
        )
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Download Excel files and produce processed CSV tables."""
    years = list(_EXCEL_URLS.keys())

    print("Downloading Excel files...")
    download_excel_files()

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing mangel per fylke/region (Tabell 1)...")
    df_fylke = pd.concat([_parse_mangel_fylke(y) for y in years], ignore_index=True)
    dest = _PROCESSED_DIR / "mangel_per_fylke.csv"
    df_fylke.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_fylke)} rows)")

    print("Parsing mangel per yrkesgruppe (Tabell 2)...")
    df_yrke = pd.concat(
        [_parse_mangel_yrkesgruppe(y) for y in years], ignore_index=True
    )
    dest = _PROCESSED_DIR / "mangel_per_yrkesgruppe.csv"
    df_yrke.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_yrke)} rows)")

    print("Parsing sysselsettingsbarometer nasjonal (Tabell 3/4)...")
    # Use the 2025 file for the most complete time series (includes all prior years)
    df_barometer = _parse_barometer(2025)
    dest = _PROCESSED_DIR / "sysselsettingsbarometer.csv"
    df_barometer.to_csv(dest, index=False)
    print(
        f"  -> {dest} ({len(df_barometer)} rows, {df_barometer['aar'].min()}–{df_barometer['aar'].max()})"
    )

    print("Parsing sysselsettingsbarometer etter næring (Tabell 4/5)...")
    df_bar_naring = pd.concat(
        [_parse_barometer_naring(y) for y in years], ignore_index=True
    )
    dest = _PROCESSED_DIR / "sysselsettingsbarometer_naring.csv"
    df_bar_naring.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_bar_naring)} rows)")

    print("Parsing mangel per næring (Tabell 7/V1/5/6)...")
    df_naring = pd.concat([_parse_mangel_naring(y) for y in years], ignore_index=True)
    dest = _PROCESSED_DIR / "mangel_per_naring.csv"
    df_naring.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_naring)} rows)")

    print("\nFerdig!")


if __name__ == "__main__":
    main()
