"""Standardiserer SSB-sektordata til felles næringsinndeling og beregner mismatch-mål.

Leser:
  data/raw/ssb_ledige_stillinger.csv  — ledige stillinger per næring per kvartal (11587)
  data/raw/ssb_sysselsatte.csv        — sysselsatte per næring per kvartal (11154)

Beregner per kvartal:
  - Stramhetsindikator per næring: v / (u + v)
  - Jackman–Roper mismatch-indeks
  - Cobb–Douglas mismatch-indeks (α = 0.3, 0.5, 0.7)
  - Aggregert stramhetsratio og stramhetsindikator

Lagrer:
  data/processed/ssb_mismatch_kvartal.csv   — kvartalsvise mismatch-mål
  data/processed/ssb_stramhet_naering.csv   — stramhetsindikator per næring per kvartal

Kjøres:
  uv run python -m src.datagrunnlag.standardiser_ssb_mismatch_data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_RAW = Path("data/raw")
_PROCESSED = Path("data/processed")

# ── Mapping fra detaljerte NACE-koder til felles næringer ─────────────────────

# 11154 (sysselsatte) → felles næringsgruppe
_MAP_SYSSELSATTE: dict[str, str] = {
    "01-02": "01-03",
    "03": "01-03",
    "05-09": "05-09",
    "10-12": "10-33",
    "13-15": "10-33",
    "16": "10-33",
    "17": "10-33",
    "18": "10-33",
    "19-22": "10-33",
    "23": "10-33",
    "24-25": "10-33",
    "26-28": "10-33",
    "29-30": "10-33",
    "31-33": "10-33",
    "35": "35-39",
    "36-39": "35-39",
    "41-43": "41-43",
    "45": "45-47",
    "46": "45-47",
    "47": "45-47",
    "49": "49-53",
    "50": "49-53",
    "51": "49-53",
    "52": "49-53",
    "53": "49-53",
    "55": "55-56",
    "56": "55-56",
    "58-63": "58-63",
    "64-66": "64-66",
    "68": "68",
    "69-75": "69-75",
    "77-82": "77-82",
    "84": "84",
    "85": "85",
    "86-88": "86-88",
    "90-93": "90-93",
    "94-99": "94-96",
}

# 11587 (ledige stillinger) → felles næringsgruppe
_MAP_STILLINGER: dict[str, str] = {
    "01-03": "01-03",
    "05-09": "05-09",
    "10-33": "10-33",
    "35-39": "35-39",
    "41-43": "41-43",
    "45-47": "45-47",
    "49-53": "49-53",
    "55-56": "55-56",
    "58-63": "58-63",
    "64-66": "64-66",
    "68": "68",
    "69-75": "69-75",
    "77-82": "77-82",
    "84": "84",
    "85": "85",
    "86": "86-88",
    "87": "86-88",
    "88": "86-88",
    "90-93": "90-93",
    "94-96": "94-96",
}

_NAERING_LABELS: dict[str, str] = {
    "01-03": "Jordbruk, skogbruk og fiske",
    "05-09": "Bergverksdrift og utvinning",
    "10-33": "Industri",
    "35-39": "Elektrisitet, vann og renovasjon",
    "41-43": "Bygge- og anleggsvirksomhet",
    "45-47": "Varehandel",
    "49-53": "Transport og lagring",
    "55-56": "Overnatting og servering",
    "58-63": "Informasjon og kommunikasjon",
    "64-66": "Finans og forsikring",
    "68": "Omsetning av fast eiendom",
    "69-75": "Faglig og teknisk tjenesteyting",
    "77-82": "Forretningsmessig tjenesteyting",
    "84": "Offentlig administrasjon",
    "85": "Undervisning",
    "86-88": "Helse- og sosialtjenester",
    "90-93": "Kultur og underholdning",
    "94-96": "Annen tjenesteyting",
}


def _map_and_aggregate(
    df: pd.DataFrame,
    mapping: dict[str, str],
    value_col: str,
) -> pd.DataFrame:
    """Map nace_code til felles gruppe og aggreger verdier."""
    df = df.copy()
    df["naering_felles"] = df["nace_code"].map(mapping)
    # Dropp koder uten mapping
    df = df.dropna(subset=["naering_felles"])
    return df.groupby(["kvartal", "naering_felles"])[value_col].sum().reset_index()


def _beregn_mismatch_per_kvartal(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn kvartalsvise mismatch-mål fra sammenslåtte sektordata."""
    records = []
    for kvartal, grp in df.groupby("kvartal"):
        v = grp["ledige_stillinger"].values
        u = grp["sysselsatte"].values
        V = v.sum()
        U = u.sum()

        # Jackman–Roper mismatch-indeks
        mismatch_indeks = (
            0.5 * np.sum(np.abs(v / V - u / U)) if V > 0 and U > 0 else np.nan
        )

        # Cobb-Douglas mismatch-indeks
        cd_values = {}
        for alpha in [0.3, 0.5, 0.7]:
            key = f"mismatch_indeks_cd_{int(alpha * 100)}"
            cd_values[key] = (
                1 - np.sum((v / V) ** alpha * (u / U) ** (1 - alpha))
                if V > 0 and U > 0
                else np.nan
            )

        # Aggregert stramhet
        stramhetsratio = V / U if U > 0 else np.nan
        stramhetsindikator = V / (U + V) if (U + V) > 0 else np.nan

        records.append(
            {
                "kvartal": kvartal,
                "total_stillinger": V,
                "total_sysselsatte": U,
                "mismatch_indeks": round(mismatch_indeks, 4),
                **{k: round(v, 4) for k, v in cd_values.items()},
                "stramhetsratio": round(stramhetsratio, 6),
                "stramhetsindikator": round(stramhetsindikator, 6),
                "antall_naeringer": len(grp),
            }
        )

    return pd.DataFrame(records)


def _beregn_stramhet_per_naering(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn stramhetsindikator per næring per kvartal."""
    df = df.copy()
    df["stramhetsratio"] = df["ledige_stillinger"] / df["sysselsatte"]
    df["stramhetsindikator"] = df["ledige_stillinger"] / (
        df["sysselsatte"] + df["ledige_stillinger"]
    )
    df["naering_label"] = df["naering_felles"].map(_NAERING_LABELS)
    return df.sort_values(["kvartal", "naering_felles"]).reset_index(drop=True)


def main() -> None:
    """Les SSB-data, standardiser, beregn mismatch-mål."""
    _PROCESSED.mkdir(parents=True, exist_ok=True)

    print("Leser SSB-data...")
    df_still = pd.read_csv(_RAW / "ssb_ledige_stillinger.csv")
    df_syss = pd.read_csv(_RAW / "ssb_sysselsatte.csv")
    print(
        f"  Stillinger: {len(df_still)} rader, {df_still['nace_code'].nunique()} næringer"
    )
    print(
        f"  Sysselsatte: {len(df_syss)} rader, {df_syss['nace_code'].nunique()} næringer"
    )

    print("\nMapper til felles næringsgrupper...")
    df_v = _map_and_aggregate(df_still, _MAP_STILLINGER, "ledige_stillinger")
    df_u = _map_and_aggregate(df_syss, _MAP_SYSSELSATTE, "sysselsatte")

    # Slå sammen på kvartal og næringsgruppe
    df_merged = df_v.merge(df_u, on=["kvartal", "naering_felles"], how="inner")
    # Filtrer bort rader med manglende data
    df_merged = df_merged.dropna(subset=["ledige_stillinger", "sysselsatte"])
    df_merged = df_merged[
        (df_merged["ledige_stillinger"] > 0) & (df_merged["sysselsatte"] > 0)
    ]
    print(
        f"  Sammenslåtte data: {len(df_merged)} rader, {df_merged['kvartal'].nunique()} kvartaler"
    )

    # Stramhet per næring
    print("\nBeregner stramhetsindikator per næring...")
    df_naering = _beregn_stramhet_per_naering(df_merged)
    dest = _PROCESSED / "ssb_stramhet_naering.csv"
    df_naering.to_csv(dest, index=False)
    print(f"  -> {dest}")

    # Kvartalsvise mismatch-mål
    print("\nBeregner kvartalsvise mismatch-mål...")
    df_mismatch = _beregn_mismatch_per_kvartal(df_merged)
    dest = _PROCESSED / "ssb_mismatch_kvartal.csv"
    df_mismatch.to_csv(dest, index=False)
    print(f"  -> {dest}")
    print(df_mismatch.to_string(index=False))

    print("\nFerdig!")


if __name__ == "__main__":
    main()
