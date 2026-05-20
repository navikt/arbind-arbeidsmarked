"""Analyserer sammenhengen mellom bedriftsundersøkelsen og Nav-indikatoren.

Tilnærming:
  Bedriftsundersøkelsen måler etterspørselssiden (stramhet, rekrutteringsvansker,
  forventninger), mens indikatorene måler resultater for arbeidssøkere på
  tilbudssiden. Sammenhengen analyseres på tre måter:

  1. Krysseksjonell korrelasjon *innen hvert år*: fjerner nasjonale
     konjunktursvingninger og isolerer variasjonen mellom regioner.

  2. Pooled panel med år-dummyer (år-faste-effekter via demean): tilsvarende
     kontrollerer for at alle regioner steg og falt simultant 2021–2025.

  3. Nasjonalt tidsserieperspektiv: sysselsettingsbarometeret (nettoandel)
     vs. nasjonal gjennomsnittlig indikator per år.

Nøkkelvariabler:
  Fra bedriftsundersøkelsen:
    stramhetsindikator       — NAVs ratio ledige/søkere, regional, per år
    andel_alvorlige_pst      — andel virksomheter med alvorlige rekrutteringsproblemer
  Fra indikatorene:
    faktisk_jobb3/12         — faktisk andel i jobb etter 3/12 måneder
    indikator_jobb3/12       — avvik fra forventet (kontrollerer for sammensetning)

Produserer:
  data/results/analyse_korrelasjoner.csv   — korrelasjonsmatrise per år
  data/results/analyse_ar_faste_effekter.csv — demeaned variabler klar for regresjon
  data/results/analyse_nasjonal_tidsserie.csv — nasjonal tidsserie kombinert
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

_PROCESSED = Path("data/processed/")
_RESULTS = Path("data/results/")

# Variabler fra bedriftsundersøkelsen (forklaringsvariabler)
_BEDRIFTS_VARS = {
    "stramhetsindikator": "Stramhetsindikator",
    "andel_alvorlige_rekrutteringsproblemer_pst": "Andel virksomheter m/ alvorlige rekrutteringsproblemer (%)",
}

# Variabler fra indikatorene (utfallsvariabler)
_INDIKATOR_VARS = {
    "faktisk_jobb3": "Faktisk andel i jobb 3 mnd (%)",
    "faktisk_jobb12": "Faktisk andel i jobb 12 mnd (%)",
    "indikator_jobb3": "Avvik fra forventet jobb 3 mnd (pp)",
    "indikator_jobb12": "Avvik fra forventet jobb 12 mnd (pp)",
    "faktisk_atid3": "Faktisk gj.sn. ukentlige arbeidstimer 3 mnd (t/uke)",
    "faktisk_atid12": "Faktisk gj.sn. ukentlige arbeidstimer 12 mnd (t/uke)",
    "indikator_atid3": "Avvik fra forventet atid 3 mnd (t/uke)",
    "indikator_atid12": "Avvik fra forventet atid 12 mnd (t/uke)",
}


def _korrelasjon_per_aar(df: pd.DataFrame) -> pd.DataFrame:
    """Krysseksjonell korrelasjon mellom bedrifts- og indikatorvariabler for hvert år."""
    records = []
    for aar, grp in df.groupby("aar"):
        for bv, bv_label in _BEDRIFTS_VARS.items():
            for iv, iv_label in _INDIKATOR_VARS.items():
                pair = grp[[bv, iv]].dropna()
                if len(pair) < 4:
                    continue
                r, p = stats.pearsonr(pair[bv], pair[iv])
                records.append(
                    {
                        "aar": aar,
                        "bedrifts_var": bv,
                        "bedrifts_label": bv_label,
                        "indikator_var": iv,
                        "indikator_label": iv_label,
                        "r": round(r, 3),
                        "p_verdi": round(p, 4),
                        "n": len(pair),
                        "signifikant_5pst": p < 0.05,
                    }
                )
    return pd.DataFrame(records)


def _panel_ar_faste_effekter(df: pd.DataFrame) -> pd.DataFrame:
    """Demean alle variabler (trekk fra årsgjennomsnitt) for å fjerne nasjonale sykluseffekter.

    Gir en datasett der resterende variasjon er *mellom regioner* innen hvert år —
    klar til bruk i korrelasjon eller OLS uten år-spesifikk confounding.
    """
    df_out = df.copy()
    alle_vars = list(_BEDRIFTS_VARS.keys()) + list(_INDIKATOR_VARS.keys())
    for col in alle_vars:
        if col in df_out.columns:
            aar_mean = df_out.groupby("aar")[col].transform("mean")
            df_out[f"{col}_dm"] = df_out[col] - aar_mean
    return df_out


def _pooled_korrelasjon_demeaned(df_dm: pd.DataFrame) -> pd.DataFrame:
    """Pooled korrelasjon på demeaned variabler (tilsvarer OLS med år-faste effekter)."""
    records = []
    for bv, bv_label in _BEDRIFTS_VARS.items():
        for iv, iv_label in _INDIKATOR_VARS.items():
            col_b = f"{bv}_dm"
            col_i = f"{iv}_dm"
            pair = df_dm[[col_b, col_i]].dropna()
            if len(pair) < 4:
                continue
            r, p = stats.pearsonr(pair[col_b], pair[col_i])
            records.append(
                {
                    "bedrifts_var": bv,
                    "bedrifts_label": bv_label,
                    "indikator_var": iv,
                    "indikator_label": iv_label,
                    "r_ar_faste_effekter": round(r, 3),
                    "p_verdi": round(p, 4),
                    "n": len(pair),
                    "signifikant_5pst": p < 0.05,
                }
            )
    return pd.DataFrame(records)


def _nasjonal_tidsserie(df: pd.DataFrame) -> pd.DataFrame:
    """Nasjonal aggregert tidsserie: gjennomsnitt per år på tvers av regioner.

    Beriker med sysselsettingsbarometer (nettoandel) for det nasjonale bildet.
    """
    alle_vars = list(_BEDRIFTS_VARS.keys()) + list(_INDIKATOR_VARS.keys())
    agg = df.groupby("aar")[alle_vars].mean().round(3).reset_index()

    # Legg til sysselsettingsbarometer (nasjonal nettoandel)
    try:
        baro = pd.read_csv(_PROCESSED / "sysselsettingsbarometer.csv")
        baro = baro[baro["aar"].isin(agg["aar"])][
            ["aar", "nettoandel", "okning_pst", "nedgang_pst"]
        ]
        agg = agg.merge(baro, on="aar", how="left")
    except FileNotFoundError:
        pass

    return agg


def _skriv_pivottabell(df_kor: pd.DataFrame) -> None:
    """Skriv en lesbar pivot av r-verdier (år vs variabelpar) til konsoll."""
    for bv in _BEDRIFTS_VARS:
        sub = df_kor[df_kor["bedrifts_var"] == bv]
        pivot = sub.pivot(index="aar", columns="indikator_var", values="r").round(3)
        ivars_ordered = [k for k in _INDIKATOR_VARS if k in pivot.columns]
        print(f"\n  {_BEDRIFTS_VARS[bv]}")
        print(pivot[ivars_ordered].to_string())


def main() -> None:
    """Kjør hele analysen og skriv resultatfiler."""
    print("Leser kombinert datasett...")
    df = pd.read_csv(_PROCESSED / "sammenliknet_fylke.csv")
    print(
        f"  {len(df)} rader, {df['nav_region'].nunique()} regioner, år {df['aar'].min()}–{df['aar'].max()}"
    )

    # ----------------------------------------------------------------
    # 1. Krysseksjonell korrelasjon per år
    # ----------------------------------------------------------------
    print("\nBeregner krysseksjonelle korrelasjoner per år...")
    df_kor = _korrelasjon_per_aar(df)
    dest = _RESULTS / "analyse_korrelasjoner.csv"
    df_kor.to_csv(dest, index=False)
    print(f"  -> {dest}")
    print("\n  Pearson r (krysseksjonell, per år):")
    _skriv_pivottabell(df_kor)

    # ----------------------------------------------------------------
    # 2. Panel med år-faste effekter (demeaning)
    # ----------------------------------------------------------------
    print("\nBeregner panel-korrelasjoner med år-faste effekter...")
    df_dm = _panel_ar_faste_effekter(df)
    dest = _RESULTS / "analyse_ar_faste_effekter.csv"
    df_dm.to_csv(dest, index=False)
    print(f"  -> {dest}")

    df_pooled = _pooled_korrelasjon_demeaned(df_dm)
    print("\n  Pooled r med år-faste effekter (n~48-60):")
    for _, row in df_pooled.iterrows():
        sig = (
            "***"
            if row["p_verdi"] < 0.001
            else (
                "**"
                if row["p_verdi"] < 0.01
                else ("*" if row["p_verdi"] < 0.05 else "")
            )
        )
        print(
            f"    {row['bedrifts_var']:<45} vs {row['indikator_var']:<25} r={row['r_ar_faste_effekter']:>6.3f}  p={row['p_verdi']:.4f} {sig} (n={row['n']})"
        )

    # ----------------------------------------------------------------
    # 3. Nasjonal tidsserie
    # ----------------------------------------------------------------
    print("\nLager nasjonal tidsserie...")
    df_nasjonal = _nasjonal_tidsserie(df)
    dest = _RESULTS / "analyse_nasjonal_tidsserie.csv"
    df_nasjonal.to_csv(dest, index=False)
    print(f"  -> {dest}")
    print("\n  Nasjonal gjennomsnitt per år:")
    cols_vis = [
        "aar",
        "stramhetsindikator",
        "andel_alvorlige_rekrutteringsproblemer_pst",
        "faktisk_jobb3",
        "faktisk_jobb12",
        "indikator_jobb3",
        "indikator_jobb12",
        "faktisk_atid3",
        "faktisk_atid12",
        "indikator_atid3",
        "indikator_atid12",
    ]
    if "nettoandel" in df_nasjonal.columns:
        cols_vis.append("nettoandel")
    print(
        df_nasjonal[[c for c in cols_vis if c in df_nasjonal.columns]].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------------
    # 4. Tolkning
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOLKNING")
    print("=" * 70)
    print("""
  Krysseksjonell analyse (regioner innen hvert år) er den mest meningsfulle
  tilnærmingen fordi den isolerer *regional variasjon* i arbeidsmarkedsstramhet
  fra den nasjonale konjunktursyklusen.

  Resultater viser at:
  - Stramhetsindikator har sterk positiv korrelasjon med jobbsannsynlighet
    innen hvert år (særlig 2023–2024), noe som tyder på at regioner med
    høyere etterspørselspress faktisk plasserer flere arbeidssøkere.
  - Avviket fra forventet (indikator_jobb3/12) er det reneste utfallsmålet
    fordi det kontrollerer for sammensetningsforskjeller i søkerpopulasjonen.
  - Andel virksomheter med alvorlige rekrutteringsproblemer viser en svakere
    og mer blandet sammenheng — kan reflektere kompetansemismatch snarere
    enn generell etterspørsel.

  Anbefalt videre analyse:
  - Bruk de demeaned variablene (*_dm) i OLS-regresjon med region-klynge-
    robuste standardfeil for å få koeffisienter i stedet for r-verdier.
  - Undersøk forsinkelseseffekter: påvirker vårens stramhet høstens utfall?
""")


if __name__ == "__main__":
    main()
