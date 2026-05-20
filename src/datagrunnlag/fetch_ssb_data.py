"""Henter kvartalsvis sektordata fra SSB for mismatch-analyse.

Tabell 11587: Ledige stillingar etter næring (etterspørsel)
Tabell 11154: Sysselsatte etter næring (tilbud)

Lagrer:
  data/raw/ssb_ledige_stillinger.csv  — ledige stillinger per næring per kvartal
  data/raw/ssb_sysselsatte.csv        — sysselsatte per næring per kvartal

Kjøres:
  uv run python src/fetch_ssb_data.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

_RAW_DIR = Path("data/raw")

# SSB PxWeb API v2 base
_API_BASE = "https://data.ssb.no/api/pxwebapi/v2/tables"


def _fetch_json_stat2(url: str) -> dict[str, Any]:
    """Hent JSON-stat2 fra SSB API."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        result: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return result


def _json_stat2_to_df(js: dict[str, Any]) -> pd.DataFrame:
    """Konverter JSON-stat2-respons til en flat DataFrame."""
    dims = list(js["dimension"].keys())
    values = js["value"]

    # Bygg indeks-tupler for alle celler
    import itertools

    categories = []
    for dim in dims:
        cat = js["dimension"][dim]["category"]
        idx = cat["index"]
        label = cat["label"]
        # Sorter etter indeksverdi
        codes = sorted(idx.keys(), key=lambda c: idx[c])
        categories.append([(c, label[c]) for c in codes])

    records = []
    for i, combo in enumerate(itertools.product(*categories)):
        row = {}
        for dim, (code, label) in zip(dims, combo):
            row[f"{dim}_code"] = code
            row[f"{dim}_label"] = label
        row["value"] = values[i] if i < len(values) else None
        records.append(row)

    return pd.DataFrame(records)


def fetch_ledige_stillinger() -> pd.DataFrame:
    """Hent ledige stillinger (tabell 11587) etter næring, kvartalsvis."""
    url = (
        f"{_API_BASE}/11587/data?lang=no"
        "&outputFormat=json-stat2"
        "&valuecodes[Tid]=*"
        "&valuecodes[NACE2007]=*"
        "&codelist[NACE2007]=vs_NACE2007ledstillNN3"
        "&valuecodes[ContentsCode]=LedigeStillinger"
        "&heading=Tid,ContentsCode"
        "&stub=NACE2007"
    )
    print("  Henter tabell 11587 (ledige stillinger)...")
    js = _fetch_json_stat2(url)
    df = _json_stat2_to_df(js)

    # Filtrer bort aggregat-raden (alle næringer)
    df = df[df["NACE2007_code"] != "01-96"].copy()

    df = df.rename(
        columns={
            "NACE2007_code": "nace_code",
            "NACE2007_label": "naering",
            "Tid_code": "kvartal",
            "value": "ledige_stillinger",
        }
    )
    return df[["kvartal", "nace_code", "naering", "ledige_stillinger"]].copy()


def fetch_sysselsatte() -> pd.DataFrame:
    """Hent sysselsatte (tabell 11154) etter næring, kvartalsvis."""
    url = (
        f"{_API_BASE}/11154/data?lang=no"
        "&outputFormat=json-stat2"
        "&valuecodes[Tid]=*"
        "&valuecodes[NACE2007]=*"
        "&valuecodes[ContentsCode]=Sysselsatte"
        "&valuecodes[Kjonn]=0"
        "&heading=Tid,ContentsCode"
        "&stub=Kjonn,NACE2007"
    )
    print("  Henter tabell 11154 (sysselsatte)...")
    js = _fetch_json_stat2(url)
    df = _json_stat2_to_df(js)

    # Filtrer bort aggregat (alle næringer) og kjønnskolonner
    df = df[df["NACE2007_code"] != "00-99"].copy()

    df = df.rename(
        columns={
            "NACE2007_code": "nace_code",
            "NACE2007_label": "naering",
            "Tid_code": "kvartal",
            "value": "sysselsatte_1000",
        }
    )
    # Konverter fra 1000 personer til personer
    df["sysselsatte"] = df["sysselsatte_1000"] * 1000
    return df[["kvartal", "nace_code", "naering", "sysselsatte"]].copy()


def main() -> None:
    """Hent SSB-data og lagre som CSV."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    df_still = fetch_ledige_stillinger()
    dest = _RAW_DIR / "ssb_ledige_stillinger.csv"
    df_still.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_still)} rader)")

    df_syss = fetch_sysselsatte()
    dest = _RAW_DIR / "ssb_sysselsatte.csv"
    df_syss.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_syss)} rader)")

    print("\nFerdig!")


if __name__ == "__main__":
    main()
