"""Tester om utvidede modeller reduserer sammenhengen mellom indikatorer og stramhet.

Primæranalysen: for hver modellvariant, beregner vi korrelasjoner mellom
indikatorverdier og bedriftsundersøkelsens stramhetsmål, og sammenligner
med referansemodellen.

Tilnærming:
  1. Hent indikatorverdier per referansemåned (samme logikk som standardiser_data.py)
  2. Slå sammen med bedriftsundersøkelsesdata (sammenliknet_fylke.csv)
  3. Beregn krysseksjonelle korrelasjoner per år og modell
  4. Beregn pooled korrelasjon med år-faste effekter per modell
  5. Formell sammenligning: endring i korrelasjon mellom referanse og utvidede modeller

Produserer:
  data/results/modellendringer_stramhet_korrelasjoner.csv
  data/results/modellendringer_stramhet_pooled.csv
  data/results/modellendringer_stramhet_sammendrag.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from src.common.config import REFERANSEMAANED

_PROCESSED = Path("data/processed")
_RESULTS = Path("data/results")

_MODELLNAVN = {
    77: "Referanse",
    78: "Stillingsrate",
    79: "Shiftshare",
    80: "Ledighetsrate",
    81: "Ledighetsrate ung",
}

_BEDRIFTS_VARS = {
    "stramhetsindikator": "Stramhetsindikator",
    "andel_alvorlige_rekrutteringsproblemer_pst": "Andel alvorlige rekr.problemer (%)",
}

_INDIKATOR_UTFALL = {
    "indikator_jobb3": "Indikator jobb 3 mnd",
    "indikator_jobb12": "Indikator jobb 12 mnd",
    "indikator_atid3": "Indikator atid 3 mnd",
    "indikator_atid12": "Indikator atid 12 mnd",
}


def _hent_modell_per_referansemaaned(df_reg: pd.DataFrame) -> pd.DataFrame:
    """Filtrer regionalt modelldatasett til referansemåneder og returner bredt format."""
    df = df_reg.copy()
    df["beholdningsmaaned"] = pd.to_datetime(df["beholdningsmaaned"], utc=True)
    df["aar"] = df["beholdningsmaaned"].dt.year
    df["maaned"] = df["beholdningsmaaned"].dt.month

    rows = []
    for aar, ref_mnd in REFERANSEMAANED.items():
        utsnitt = df[(df["aar"] == aar) & (df["maaned"] == ref_mnd)]
        if utsnitt.empty:
            print(f"  Advarsel: ingen data for {aar} måned {ref_mnd}")
            continue
        rows.append(utsnitt)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def _korrelasjoner_per_aar_modell(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn krysseksjonell Pearson-korrelasjon per år, modell og variabelpar."""
    records = []
    for aar, grp_aar in df.groupby("aar"):
        for rid, modellnavn in _MODELLNAVN.items():
            grp = grp_aar[grp_aar["result_id"] == rid]
            if len(grp) < 4:
                continue
            for bv, bv_label in _BEDRIFTS_VARS.items():
                for iv, iv_label in _INDIKATOR_UTFALL.items():
                    pair = grp[[bv, iv]].dropna()
                    if len(pair) < 4:
                        continue
                    r, p = stats.pearsonr(pair[bv], pair[iv])
                    records.append(
                        {
                            "aar": aar,
                            "result_id": rid,
                            "modell": modellnavn,
                            "bedrifts_var": bv,
                            "bedrifts_label": bv_label,
                            "indikator_var": iv,
                            "indikator_label": iv_label,
                            "r": round(r, 4),
                            "p_verdi": round(p, 4),
                            "n": len(pair),
                        }
                    )
    return pd.DataFrame(records)


def _pooled_demeaned_korrelasjon(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled korrelasjon med år-faste effekter per modell."""
    records = []
    for rid, modellnavn in _MODELLNAVN.items():
        grp = df[df["result_id"] == rid].copy()
        if len(grp) < 8:
            continue

        # Demean (fjern årsgjennomsnitt)
        alle_vars = list(_BEDRIFTS_VARS.keys()) + list(_INDIKATOR_UTFALL.keys())
        for col in alle_vars:
            if col in grp.columns:
                grp[f"{col}_dm"] = grp[col] - grp.groupby("aar")[col].transform("mean")

        for bv, bv_label in _BEDRIFTS_VARS.items():
            for iv, iv_label in _INDIKATOR_UTFALL.items():
                col_b = f"{bv}_dm"
                col_i = f"{iv}_dm"
                pair = grp[[col_b, col_i]].dropna()
                if len(pair) < 4:
                    continue
                r, p = stats.pearsonr(pair[col_b], pair[col_i])
                records.append(
                    {
                        "result_id": rid,
                        "modell": modellnavn,
                        "bedrifts_var": bv,
                        "bedrifts_label": bv_label,
                        "indikator_var": iv,
                        "indikator_label": iv_label,
                        "r_demeaned": round(r, 4),
                        "p_verdi": round(p, 4),
                        "n": len(pair),
                    }
                )
    return pd.DataFrame(records)


def _sammenlign_med_referanse(df_korr: pd.DataFrame) -> pd.DataFrame:
    """Beregn endring i korrelasjon fra referansemodellen til utvidede modeller."""
    ref = df_korr[df_korr["result_id"] == 77][
        ["aar", "bedrifts_var", "indikator_var", "r"]
    ].rename(columns={"r": "r_referanse"})

    merged = df_korr.merge(ref, on=["aar", "bedrifts_var", "indikator_var"], how="left")
    merged["delta_r"] = merged["r"] - merged["r_referanse"]
    merged["abs_r_reduksjon"] = merged["r_referanse"].abs() - merged["r"].abs()

    return merged


def main() -> None:
    """Kjør stramhetsanalysen for alle modeller."""
    _RESULTS.mkdir(parents=True, exist_ok=True)

    # Les regionalt modelldatasett
    print("Leser regionalt modelldatasett...")
    df_reg = pd.read_csv(_PROCESSED / "modellendringer_regionalt.csv")

    # Filtrer til referansemåneder
    print("Filtrerer til referansemåneder...")
    df_ref = _hent_modell_per_referansemaaned(df_reg)
    print(f"  {len(df_ref)} rader (5 modeller × {len(df_ref) // 5} region-år)")

    # Les bedriftsundersøkelsesdata
    print("Leser bedriftsundersøkelsesdata...")
    df_bedrift = pd.read_csv(_PROCESSED / "sammenliknet_fylke.csv")
    # Behold bare stramhetsvariablene og identifikatorer
    bedrift_cols = ["aar", "nav_region"] + list(_BEDRIFTS_VARS.keys())
    df_bedrift = df_bedrift[[c for c in bedrift_cols if c in df_bedrift.columns]]
    df_bedrift = df_bedrift.drop_duplicates()

    # Slå sammen
    print("Slår sammen modell- og bedriftsdata...")
    df = df_ref.merge(
        df_bedrift,
        left_on=["aar", "org_sted"],
        right_on=["aar", "nav_region"],
        how="inner",
    )
    print(f"  {len(df)} rader etter merge")
    print(f"  Modeller: {sorted(df['result_id'].unique())}")
    print(f"  År: {sorted(df['aar'].unique())}")
    print(f"  Regioner: {df['org_sted'].nunique()}")

    if df.empty:
        print("FEIL: Ingen data etter merge — sjekk at nav_region matcher org_sted.")
        return

    # 1. Korrelasjoner per år og modell
    print("\n1. Krysseksjonelle korrelasjoner per år og modell:")
    df_korr = _korrelasjoner_per_aar_modell(df)
    df_korr_med_ref = _sammenlign_med_referanse(df_korr)
    dest = _RESULTS / "modellendringer_stramhet_korrelasjoner.csv"
    df_korr_med_ref.to_csv(dest, index=False)
    print(f"  -> {dest}")

    # Vis nøkkelresultater
    for bv in _BEDRIFTS_VARS:
        print(f"\n  {_BEDRIFTS_VARS[bv]}:")
        for iv in _INDIKATOR_UTFALL:
            print(f"    {_INDIKATOR_UTFALL[iv]}:")
            sub = df_korr_med_ref[
                (df_korr_med_ref["bedrifts_var"] == bv)
                & (df_korr_med_ref["indikator_var"] == iv)
            ]
            # Gjennomsnittlig |r| per modell (over alle år)
            for rid, modellnavn in _MODELLNAVN.items():
                msub = sub[sub["result_id"] == rid]
                if msub.empty:
                    continue
                mean_abs_r = msub["r"].abs().mean()
                mean_delta = msub["abs_r_reduksjon"].mean() if rid != 77 else 0
                print(
                    f"      {modellnavn:20s}: gj.sn. |r|={mean_abs_r:.3f}"
                    + (f"  (|r| reduksjon={mean_delta:+.3f})" if rid != 77 else "")
                )

    # 2. Pooled demeaned korrelasjon
    print("\n2. Pooled korrelasjon med år-faste effekter:")
    df_pooled = _pooled_demeaned_korrelasjon(df)
    dest = _RESULTS / "modellendringer_stramhet_pooled.csv"
    df_pooled.to_csv(dest, index=False)
    print(f"  -> {dest}")

    for bv in _BEDRIFTS_VARS:
        print(f"\n  {_BEDRIFTS_VARS[bv]}:")
        for iv in _INDIKATOR_UTFALL:
            sub = df_pooled[
                (df_pooled["bedrifts_var"] == bv) & (df_pooled["indikator_var"] == iv)
            ]
            for _, row in sub.iterrows():
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
                    f"    {row['modell']:20s}: r={row['r_demeaned']:>7.4f}  p={row['p_verdi']:.4f} {sig}"
                )

    # 3. Sammendrag
    print("\n3. Sammendrag — gjennomsnittlig |r|-reduksjon per modell:")
    sammendrag = []
    for rid in [78, 79, 80, 81]:
        sub = df_korr_med_ref[df_korr_med_ref["result_id"] == rid]
        # Bare for indikator-variablene (ikke faktisk)
        for iv in _INDIKATOR_UTFALL:
            iv_sub = sub[sub["indikator_var"] == iv]
            mean_red = iv_sub["abs_r_reduksjon"].mean()
            sammendrag.append(
                {
                    "modell": _MODELLNAVN[rid],
                    "indikator_var": iv,
                    "gj_sn_abs_r_reduksjon": round(mean_red, 4),
                }
            )
    df_samm = pd.DataFrame(sammendrag)
    dest = _RESULTS / "modellendringer_stramhet_sammendrag.csv"
    df_samm.to_csv(dest, index=False)
    print(f"  -> {dest}")
    print()
    print(
        df_samm.pivot(
            index="modell", columns="indikator_var", values="gj_sn_abs_r_reduksjon"
        ).to_string()
    )


if __name__ == "__main__":
    main()
