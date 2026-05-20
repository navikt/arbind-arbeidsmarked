"""Standardiserer indikator- og bedriftsundersøkelsesdata til samme geografi og tidsperiode.

Geografisk standardisering:
  - Bedriftsundersøkelsen 2021–2023 bruker Nav-regioner direkte (Øst-Viken osv.)
  - Bedriftsundersøkelsen 2024–2025 bruker individuelle fylker (Østfold, Akershus osv.)
    som aggregeres til tilsvarende Nav-regioner via vektet gjennomsnitt / sum.
  - Indikatordata er allerede på Nav-regionnivå (org_sted).

Tidsperiode:
  Bedriftsundersøkelsen gjennomføres om våren med referansemåned per år:
    2021: februar, 2022: april, 2023: april, 2024: mars, 2025: mars
  Indikatordata hentes for tilsvarende referansemåned samme år.

Produserer:
  data/processed/sammenliknet_fylke.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_PROCESSED = Path("data/processed")
_RAW = Path("data/raw")

# Referansemåned i indikatordata for hvert bedriftsundersøkelse-år
_REFERANSEMAANED: dict[int, int] = {
    2021: 2,  # februar
    2022: 4,  # april
    2023: 4,  # april
    2024: 3,  # mars
    2025: 3,  # mars
}

# Direkte navnmapping: bedriftsundersøkelse-regioner (2021–2023) → Nav-region
_REGION_TIL_NAV: dict[str, str] = {
    "Øst-Viken": "Nav Øst-Viken",
    "Vest-Viken": "Nav Vest-Viken",
    "Oslo": "Nav Oslo",
    "Innlandet": "Nav Innlandet",
    "Vestfold og Telemark": "Nav Vestfold og Telemark",
    "Agder": "Nav Agder",
    "Rogaland": "Nav Rogaland",
    "Vestland": "Nav Vestland",
    "Møre og Romsdal": "Nav Møre og Romsdal",
    "Trøndelag": "Nav Trøndelag",
    "Nordland": "Nav Nordland",
    "Troms og Finnmark": "Nav Troms og Finnmark",
}

# Fylke → Nav-region (2024–2025 bruker individuelle fylker)
_FYLKE_TIL_NAV: dict[str, str] = {
    "Østfold": "Nav Øst-Viken",
    "Akershus": "Nav Øst-Viken",
    "Oslo": "Nav Oslo",
    "Innlandet": "Nav Innlandet",
    "Buskerud": "Nav Vest-Viken",
    "Vestfold": "Nav Vestfold og Telemark",
    "Telemark": "Nav Vestfold og Telemark",
    "Agder": "Nav Agder",
    "Rogaland": "Nav Rogaland",
    "Vestland": "Nav Vestland",
    "Møre og Romsdal": "Nav Møre og Romsdal",
    "Trøndelag": "Nav Trøndelag",
    "Nordland": "Nav Nordland",
    "Troms": "Nav Troms og Finnmark",
    "Finnmark": "Nav Troms og Finnmark",
}

_RATE_COLS = ["stramhetsindikator", "andel_alvorlige_rekrutteringsproblemer_pst"]
_SUM_COLS = ["mangel_antall", "ki_nedre", "ki_ovre"]


def _weighted_mean(df: pd.DataFrame, col: str, weight_col: str) -> float | Any:
    """Compute weighted mean; falls back to simple mean if all weights are 0 or NaN."""
    w = df[weight_col].fillna(0)
    if w.sum() == 0:
        return df[col].mean()
    return (df[col] * w).sum() / w.sum()


def _standardiser_bedrifts_geografi(df: pd.DataFrame) -> pd.DataFrame:
    """Kartlegg bedriftsundersøkelse-regioner/fylker til Nav-regioner og aggreger."""
    frames = []

    for aar, grp in df.groupby("aar"):
        if aar <= 2023:
            grp = grp.copy()
            grp["nav_region"] = grp["region"].map(_REGION_TIL_NAV)
            ukjente = grp[grp["nav_region"].isna()]["region"].unique()
            if len(ukjente):
                print(f"  Advarsel {aar}: ukjente regioner: {ukjente}")
            frames.append(grp.dropna(subset=["nav_region"]))
        else:
            # 2024+: aggreger fylker til Nav-regioner
            grp = grp.copy()
            grp["nav_region"] = grp["region"].map(_FYLKE_TIL_NAV)
            ukjente = grp[grp["nav_region"].isna()]["region"].unique()
            if len(ukjente):
                print(f"  Advarsel {aar}: ukjente fylker: {ukjente}")
            grp = grp.dropna(subset=["nav_region"])

            agg_rows = []
            for nav_reg, reg_grp in grp.groupby("nav_region"):
                row: dict[str, object] = {"aar": aar, "nav_region": nav_reg}
                for col in _SUM_COLS:
                    row[col] = reg_grp[col].sum()
                for col in _RATE_COLS:
                    row[col] = _weighted_mean(reg_grp, col, "mangel_antall")
                agg_rows.append(row)

            frames.append(pd.DataFrame(agg_rows))

    result = pd.concat(frames, ignore_index=True)
    # Behold bare de kolonnene vi trenger
    keep = ["aar", "nav_region", *_SUM_COLS, *_RATE_COLS]
    existing = [c for c in keep if c in result.columns]
    return result[existing].sort_values(["aar", "nav_region"]).reset_index(drop=True)


def _hent_indikator_per_referansemaaned(df_ind: pd.DataFrame) -> pd.DataFrame:
    """Filtrer indikatordata til referansemåneden for hvert undersøkelsesår og pivot utfall."""
    df_ind = df_ind.copy()
    df_ind["beholdningsmaaned"] = pd.to_datetime(df_ind["beholdningsmaaned"], utc=True)
    df_ind["aar"] = df_ind["beholdningsmaaned"].dt.year
    df_ind["maaned"] = df_ind["beholdningsmaaned"].dt.month

    rows = []
    for aar, ref_mnd in _REFERANSEMAANED.items():
        utsnitt = df_ind[(df_ind["aar"] == aar) & (df_ind["maaned"] == ref_mnd)]
        if utsnitt.empty:
            print(f"  Advarsel: ingen indikatordata for {aar} måned {ref_mnd}")
            continue
        rows.append(utsnitt)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    # Pivot utfall-kolonnene til brede kolonner: faktisk_jobb3, faktisk_jobb12 osv.
    pivot = df.pivot_table(
        index=["aar", "org_sted"],
        columns="utfall",
        values=["faktisk", "forventet", "indikator", "antall_personer"],
        aggfunc="mean",
    )
    pivot.columns = [f"{val}_{utfall}" for val, utfall in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"org_sted": "nav_region"})

    return pivot.sort_values(["aar", "nav_region"]).reset_index(drop=True)


def main() -> None:
    """Les rådata, standardiser geografi og tid, og skriv sammenliknet tabell."""
    print("Leser bedriftsundersøkelse-data...")
    df_bedrifts = pd.read_csv(_PROCESSED / "mangel_per_fylke.csv")
    print(
        f"  {len(df_bedrifts)} rader, år {df_bedrifts['aar'].min()}–{df_bedrifts['aar'].max()}"
    )

    print("Standardiserer geografi til Nav-regioner...")
    df_bedrifts_std = _standardiser_bedrifts_geografi(df_bedrifts)
    print(f"  {len(df_bedrifts_std)} rader etter aggregering")

    print("Leser indikatordata...")
    df_ind = pd.read_csv(_RAW / "indikator_data.csv")
    print(f"  {len(df_ind)} rader")

    print("Filtrerer indikator til referansemåneder...")
    df_ind_std = _hent_indikator_per_referansemaaned(df_ind)
    print(f"  {len(df_ind_std)} rader etter filtrering")

    print("Slår sammen til felles tabell...")
    df = (
        pd.merge(
            df_bedrifts_std,
            df_ind_std,
            on=["aar", "nav_region"],
            how="outer",
        )
        .sort_values(["aar", "nav_region"])
        .reset_index(drop=True)
    )

    # Rund av for lesbarhet
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].round(4)

    dest = _PROCESSED / "sammenliknet_fylke.csv"
    df.to_csv(dest, index=False)
    print(f"\n  -> {dest} ({len(df)} rader, {df.columns.tolist()})")

    # Vis en liten oversikt
    print("\nEksempel (Nav Øst-Viken):")
    print(
        df[df["nav_region"] == "Nav Øst-Viken"]
        .filter(
            regex="aar|nav_region|mangel_antall|stramhets|faktisk_jobb12|faktisk_atid12"
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
