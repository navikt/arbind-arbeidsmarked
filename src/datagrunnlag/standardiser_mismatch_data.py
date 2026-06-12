"""Standardiserer mismatch-data: yrkesvis tilbud/etterspørsel → nasjonale mismatch-mål.

Leser:
  data/processed/mangel_per_yrkesgruppe.csv  — etterspørsel (mangel) og tilbud (ledige+tiltak) per yrke per år
  data/raw/indikator_data_nasjonalt.csv      — månedlige nasjonale indikatorverdier

Beregner per år (2021–2025):
  - Jackman–Roper mismatch-indeks M_t
  - Aggregert stramhetsratio τ_t = V_t / U_t
  - Mismatch-ledighet U^mis_t og andelen U^mis_t / U_t
  - Yrkesvis stramhetsratio (for deskriptiv analyse)

Kobler mismatch-mål til månedlige indikatorer som step-funksjon og lagrer:
  data/processed/mismatch_nasjonal.csv        — månedlig panel med mismatch-mål
  data/processed/mismatch_yrke.csv            — yrkesvis stramhetsratio per år
  data/processed/mismatch_aar.csv             — årlige mismatch-mål

Kjøres:
  uv run python -m src.datagrunnlag.standardiser_mismatch_data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import REFERANSEMAANED
from src.common.validation import require_columns

_PROCESSED = Path("data/processed")
_RAW = Path("data/raw")

# Kategori uten etterspørselsmotstykke — ekskluderes
_EKSKLUDER_YRKER = {"Ingen yrkesbakgrunn eller uoppgitt"}


def _beregn_mismatch_per_aar(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn årlige mismatch-mål fra yrkesgruppedataene."""
    df = df[~df["yrkesgruppe"].isin(_EKSKLUDER_YRKER)].copy()
    df = df.dropna(subset=["mangel_antall", "ledige_og_tiltak"])

    records = []
    for aar, grp in df.groupby("aar"):
        v = grp["mangel_antall"].values  # etterspørsel (mangel)
        u = grp["ledige_og_tiltak"].values  # tilbud (ledige + tiltak)
        V = v.sum()
        U = u.sum()

        # Jackman–Roper mismatch-indeks
        mismatch_indeks = (
            0.5 * np.sum(np.abs(v / V - u / U)) if V > 0 and U > 0 else np.nan
        )

        # Cobb-Douglas mismatch-indeks (alternativt mål)
        cd_elasticities = [0.3, 0.5, 0.7]
        cd_values = {}
        for alpha in cd_elasticities:
            key = f"mismatch_indeks_cd_{alpha:.0%}".replace("%", "")
            cd_values[key] = (
                1 - np.sum((v / V) ** alpha * (u / U) ** (1 - alpha))
                if V > 0 and U > 0
                else np.nan
            )

        # Aggregert stramhetsratio
        stramhetsratio = V / U if U > 0 else np.nan

        # Mismatch-ledighet: overskudd av ledige i yrker der tilbud > etterspørsel
        surplus = np.maximum(u - v, 0)
        mismatch_ledighet = surplus.sum()
        mismatch_ledighet_andel = mismatch_ledighet / U if U > 0 else np.nan

        records.append(
            {
                "aar": aar,
                "total_mangel": V,
                "total_ledige": U,
                "mismatch_indeks": round(mismatch_indeks, 4),
                **{k: round(v, 4) for k, v in cd_values.items()},
                "stramhetsratio": round(stramhetsratio, 4),
                "mismatch_ledighet": mismatch_ledighet,
                "mismatch_ledighet_andel": round(mismatch_ledighet_andel, 4),
                "antall_yrker": len(grp),
            }
        )

    return pd.DataFrame(records)


def _beregn_stramhet_per_yrke(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn stramhetsratio per yrke per år."""
    df = df[~df["yrkesgruppe"].isin(_EKSKLUDER_YRKER)].copy()
    df = df.dropna(subset=["mangel_antall", "ledige_og_tiltak"])

    df["stramhetsratio_yrke"] = df["mangel_antall"] / df["ledige_og_tiltak"]
    df["overskudd_ledige"] = np.maximum(df["ledige_og_tiltak"] - df["mangel_antall"], 0)
    df["overskudd_mangel"] = np.maximum(df["mangel_antall"] - df["ledige_og_tiltak"], 0)

    return df.sort_values(["aar", "yrkesgruppe"]).reset_index(drop=True)


def _tilknytt_undersokelsesaar(df_ind: pd.DataFrame) -> pd.DataFrame:
    """Tilknytt hvert månedlige observasjon til nærmeste undersøkelsesår.

    Logikk: en observasjon tilhører undersøkelsesåret dersom den faller
    på eller etter referansemåneden for det året og før referansemåneden
    for neste år. For observasjoner før første undersøkelse brukes 2021.
    """
    df = df_ind.copy()
    df["beholdningsmaaned"] = pd.to_datetime(df["beholdningsmaaned"], utc=True)
    df["aar_obs"] = df["beholdningsmaaned"].dt.year
    df["maaned_obs"] = df["beholdningsmaaned"].dt.month

    # Bygg grenseverdier (år, måned) for hvert undersøkelsesår
    cutoffs = sorted(REFERANSEMAANED.items())  # [(2021, 2), (2022, 4), ...]

    def _finn_undersokelsesaar(row: pd.Series) -> int:
        y, m = row["aar_obs"], row["maaned_obs"]
        tilhoerer = cutoffs[0][0]  # default: første år
        for survey_year, ref_month in cutoffs:
            if y > survey_year or (y == survey_year and m >= ref_month):
                tilhoerer = survey_year
        return tilhoerer

    df["undersokelsesaar"] = df.apply(_finn_undersokelsesaar, axis=1)
    return df


def main() -> None:
    """Les data, beregn mismatch, koble med indikatorer, skriv resultat."""
    print("Leser mangel_per_yrkesgruppe.csv...")
    df_yrke = require_columns(
        pd.read_csv(_PROCESSED / "mangel_per_yrkesgruppe.csv"),
        ["aar"],
        source="mangel_per_yrkesgruppe.csv",
    )
    print(f"  {len(df_yrke)} rader, år {df_yrke['aar'].min()}–{df_yrke['aar'].max()}")

    # Yrkesvis stramhet
    print("Beregner yrkesvis stramhet...")
    df_yrke_stramhet = _beregn_stramhet_per_yrke(df_yrke)
    dest = _PROCESSED / "mismatch_yrke.csv"
    df_yrke_stramhet.to_csv(dest, index=False)
    print(f"  -> {dest}")

    # Årlige mismatch-mål
    print("Beregner årlige mismatch-mål...")
    df_mismatch_aar = _beregn_mismatch_per_aar(df_yrke)
    dest = _PROCESSED / "mismatch_aar.csv"
    df_mismatch_aar.to_csv(dest, index=False)
    print(f"  -> {dest}")
    print(df_mismatch_aar.to_string(index=False))

    # Les og tilknytt nasjonale indikatorer
    print("\nLeser indikator_data_nasjonalt.csv...")
    df_ind = require_columns(
        pd.read_csv(_RAW / "indikator_data_nasjonalt.csv"),
        ["beholdningsmaaned"],
        source="indikator_data_nasjonalt.csv",
    )
    print(f"  {len(df_ind)} rader")

    print("Tilknytter undersøkelsesår...")
    df_ind = _tilknytt_undersokelsesaar(df_ind)

    # Slå sammen mismatch-mål med månedlige indikatorer
    print("Slår sammen med mismatch-mål...")
    mismatch_cols = [
        "aar",
        "mismatch_indeks",
        "mismatch_indeks_cd_30",
        "mismatch_indeks_cd_50",
        "mismatch_indeks_cd_70",
        "stramhetsratio",
        "mismatch_ledighet",
        "mismatch_ledighet_andel",
        "total_mangel",
        "total_ledige",
    ]
    df_merged = df_ind.merge(
        df_mismatch_aar[mismatch_cols],
        left_on="undersokelsesaar",
        right_on="aar",
        how="left",
        suffixes=("", "_mismatch"),
    ).drop(columns=["aar_mismatch"], errors="ignore")

    # Rund av
    float_cols = df_merged.select_dtypes("float64").columns
    df_merged[float_cols] = df_merged[float_cols].round(4)

    dest = _PROCESSED / "mismatch_nasjonal.csv"
    df_merged.to_csv(dest, index=False)
    print(f"\n  -> {dest} ({len(df_merged)} rader)")

    # Vis en liten oversikt
    print("\nEksempel (jobb3, noen måneder):")
    sample = df_merged[df_merged["utfall"] == "jobb3"].head(5)
    print(
        sample[
            [
                "beholdningsmaaned",
                "utfall",
                "faktisk",
                "indikator",
                "undersokelsesaar",
                "mismatch_indeks",
                "stramhetsratio",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
