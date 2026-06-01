"""Analyserer regionale indikatorforskjeller mellom modellvarianter.

Undersøker:
  1. Spearman-rangkorrelasjon mellom modellpar per måned
  2. Gjennomsnittlig absolutt rangendring og største endringer
  3. Regional spredning (std på tvers av regioner) per måned og modell

Produserer:
  data/results/modellendringer_regionalt_rangkorrelasjon.csv
  data/results/modellendringer_regionalt_rangendring.csv
  data/results/modellendringer_regionalt_spredning.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_PROCESSED = Path("data/processed")
_RESULTS = Path("data/results")

_MODELLNAVN = {
    77: "Referanse",
    78: "Stillingsrate",
    79: "Shiftshare",
    80: "Ledighetsrate",
    81: "Ledighetsrate ung",
}
_UTFALL = ["indikator_jobb3", "indikator_jobb12", "indikator_atid3", "indikator_atid12"]


def _rangkorrelasjon_per_maaned(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn Spearman-rangkorrelasjon mellom referanse og hver modell per måned."""
    records = []
    for utfall in _UTFALL:
        for mnd in df["beholdningsmaaned"].unique():
            ref = df[
                (df["result_id"] == 77) & (df["beholdningsmaaned"] == mnd)
            ].set_index("org_sted")[utfall]
            for rid in [78, 79, 80, 81]:
                mod = df[
                    (df["result_id"] == rid) & (df["beholdningsmaaned"] == mnd)
                ].set_index("org_sted")[utfall]
                felles = ref.index.intersection(mod.index)
                if len(felles) < 4:
                    continue
                rho, p = stats.spearmanr(ref.loc[felles], mod.loc[felles])
                records.append(
                    {
                        "beholdningsmaaned": mnd,
                        "utfall": utfall,
                        "result_id": rid,
                        "modell": _MODELLNAVN[rid],
                        "spearman_rho": round(rho, 4),
                        "p_verdi": round(p, 4),
                        "n_regioner": len(felles),
                    }
                )
    return pd.DataFrame(records)


def _rangendring(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn rangendringer per region-måned mellom referanse og modeller."""
    records = []
    for utfall in _UTFALL:
        for mnd in df["beholdningsmaaned"].unique():
            ref = (
                df[(df["result_id"] == 77) & (df["beholdningsmaaned"] == mnd)]
                .set_index("org_sted")[utfall]
                .rank(ascending=False)
            )
            for rid in [78, 79, 80, 81]:
                mod = (
                    df[(df["result_id"] == rid) & (df["beholdningsmaaned"] == mnd)]
                    .set_index("org_sted")[utfall]
                    .rank(ascending=False)
                )
                felles = ref.index.intersection(mod.index)
                for region in felles:
                    r_ref = ref[region]
                    r_mod = mod[region]
                    if pd.isna(r_ref) or pd.isna(r_mod):
                        continue
                    diff = int(r_mod - r_ref)
                    records.append(
                        {
                            "beholdningsmaaned": mnd,
                            "utfall": utfall,
                            "result_id": rid,
                            "modell": _MODELLNAVN[rid],
                            "org_sted": region,
                            "rang_referanse": int(r_ref),
                            "rang_modell": int(r_mod),
                            "rang_endring": diff,
                        }
                    )
    return pd.DataFrame(records)


def _regional_spredning(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn spredning (std, range) på tvers av regioner per måned og modell."""
    records = []
    for utfall in _UTFALL:
        for rid, navn in _MODELLNAVN.items():
            for mnd in df["beholdningsmaaned"].unique():
                serie = df[(df["result_id"] == rid) & (df["beholdningsmaaned"] == mnd)][
                    utfall
                ].dropna()
                if len(serie) < 2:
                    continue
                records.append(
                    {
                        "beholdningsmaaned": mnd,
                        "utfall": utfall,
                        "result_id": rid,
                        "modell": navn,
                        "std_regioner": round(serie.std(), 5),
                        "range_regioner": round(serie.max() - serie.min(), 5),
                        "n_regioner": len(serie),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    """Kjør regional sammenlikning."""
    _RESULTS.mkdir(parents=True, exist_ok=True)

    print("Leser regionalt datasett...")
    df = pd.read_csv(_PROCESSED / "modellendringer_regionalt.csv")

    # 1. Rangkorrelasjon
    print("\n1. Spearman rangkorrelasjon med referansemodellen:")
    df_rang = _rangkorrelasjon_per_maaned(df)
    dest = _RESULTS / "modellendringer_regionalt_rangkorrelasjon.csv"
    df_rang.to_csv(dest, index=False)
    print(f"  -> {dest}")

    for utfall in _UTFALL:
        print(f"\n  {utfall}:")
        for rid in [78, 79, 80, 81]:
            sub = df_rang[(df_rang["utfall"] == utfall) & (df_rang["result_id"] == rid)]
            print(
                f"    {_MODELLNAVN[rid]:20s}: gj.sn. ρ={sub['spearman_rho'].mean():.3f}, "
                f"min ρ={sub['spearman_rho'].min():.3f}, "
                f"andel ρ<0.9={np.mean(sub['spearman_rho'] < 0.9):.0%}"
            )

    # 2. Rangendring
    print("\n2. Rangendringer:")
    df_endring = _rangendring(df)
    dest = _RESULTS / "modellendringer_regionalt_rangendring.csv"
    df_endring.to_csv(dest, index=False)
    print(f"  -> {dest}")

    for utfall in _UTFALL:
        print(f"\n  {utfall}:")
        sub = df_endring[df_endring["utfall"] == utfall]
        for rid in [78, 79, 80, 81]:
            msub = sub[sub["result_id"] == rid]
            mean_abs = msub["rang_endring"].abs().mean()
            andel_ge3 = np.mean(msub["rang_endring"].abs() >= 3)
            max_endring = msub["rang_endring"].abs().max()
            print(
                f"    {_MODELLNAVN[rid]:20s}: gj.sn. |endring|={mean_abs:.1f}, "
                f"max |endring|={max_endring}, andel ≥3 plasser={andel_ge3:.0%}"
            )

    # 3. Regional spredning
    print("\n3. Regional spredning:")
    df_spred = _regional_spredning(df)
    dest = _RESULTS / "modellendringer_regionalt_spredning.csv"
    df_spred.to_csv(dest, index=False)
    print(f"  -> {dest}")

    for utfall in _UTFALL:
        print(f"\n  {utfall}:")
        ref_std = df_spred[
            (df_spred["utfall"] == utfall) & (df_spred["result_id"] == 77)
        ]["std_regioner"].mean()
        for rid, navn in _MODELLNAVN.items():
            sub = df_spred[
                (df_spred["utfall"] == utfall) & (df_spred["result_id"] == rid)
            ]
            mean_std = sub["std_regioner"].mean()
            endring = ((mean_std - ref_std) / ref_std * 100) if ref_std > 0 else 0
            print(
                f"    {navn:20s}: gj.sn. std={mean_std:.5f} ({endring:+.0f}% vs referanse)"
            )


if __name__ == "__main__":
    main()
