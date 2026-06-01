"""Analyserer nasjonale indikatorforskjeller mellom modellvarianter.

Undersøker:
  1. Tidsserieplot av indikatorverdier for alle modeller
  2. Avvik fra referansemodellen (delta-plot)
  3. Volatilitet (std, range, IQR) per modell
  4. Korrelasjonsmatrise mellom modellenes indikatorer

Produserer:
  data/results/modellendringer_nasjonalt_volatilitet.csv
  data/results/modellendringer_nasjonalt_korrelasjon.csv
  data/results/modellendringer_nasjonalt_delta.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def _beregn_volatilitet(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn volatilitetsstatistikk per modell og utfallsvariabel."""
    records = []
    for utfall in _UTFALL:
        for rid, navn in _MODELLNAVN.items():
            serie = df[df["result_id"] == rid][utfall].dropna()
            records.append(
                {
                    "utfall": utfall,
                    "result_id": rid,
                    "modell": navn,
                    "gjennomsnitt": round(serie.mean(), 5),
                    "std": round(serie.std(), 5),
                    "min": round(serie.min(), 5),
                    "max": round(serie.max(), 5),
                    "range": round(serie.max() - serie.min(), 5),
                    "iqr": round(serie.quantile(0.75) - serie.quantile(0.25), 5),
                    "n": len(serie),
                }
            )
    return pd.DataFrame(records)


def _beregn_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn avvik fra referansemodellen for hver måned."""
    ref = df[df["result_id"] == 77].set_index("beholdningsmaaned")
    records = []

    for rid in [78, 79, 80, 81]:
        mod = df[df["result_id"] == rid].set_index("beholdningsmaaned")
        for utfall in _UTFALL:
            # Merge på felles måneder
            felles = ref.index.intersection(mod.index)
            for mnd in felles:
                delta = mod.loc[mnd, utfall] - ref.loc[mnd, utfall]
                records.append(
                    {
                        "beholdningsmaaned": mnd,
                        "result_id": rid,
                        "modell": _MODELLNAVN[rid],
                        "utfall": utfall,
                        "indikator_referanse": ref.loc[mnd, utfall],
                        "indikator_modell": mod.loc[mnd, utfall],
                        "delta": delta,
                    }
                )
    return pd.DataFrame(records)


def _beregn_korrelasjon(df: pd.DataFrame) -> pd.DataFrame:
    """Korrelasjonsmatrise mellom modellenes indikatorer per utfall."""
    records = []
    for utfall in _UTFALL:
        # Pivot: måneder som rader, modeller som kolonner
        pivot = df.pivot_table(
            index="beholdningsmaaned", columns="result_id", values=utfall
        )
        pivot.columns = [_MODELLNAVN[c] for c in pivot.columns]
        corr = pivot.corr().round(4)

        for m1 in corr.columns:
            for m2 in corr.columns:
                records.append(
                    {
                        "utfall": utfall,
                        "modell_1": m1,
                        "modell_2": m2,
                        "pearson_r": corr.loc[m1, m2],
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    """Kjør nasjonal sammenlikning."""
    _RESULTS.mkdir(parents=True, exist_ok=True)

    print("Leser nasjonalt datasett...")
    df = pd.read_csv(_PROCESSED / "modellendringer_nasjonalt.csv")

    # 1. Volatilitet
    print("\n1. Volatilitet per modell:")
    df_vol = _beregn_volatilitet(df)
    dest = _RESULTS / "modellendringer_nasjonalt_volatilitet.csv"
    df_vol.to_csv(dest, index=False)
    print(f"  -> {dest}")

    for utfall in _UTFALL:
        print(f"\n  {utfall}:")
        sub = df_vol[df_vol["utfall"] == utfall].sort_values("result_id")
        ref_std = sub[sub["result_id"] == 77]["std"].values[0]
        for _, row in sub.iterrows():
            endring = ((row["std"] - ref_std) / ref_std * 100) if ref_std > 0 else 0
            print(
                f"    {row['modell']:20s}: std={row['std']:.5f}  range={row['range']:.4f}  ({endring:+.0f}% vs referanse)"
            )

    # 2. Delta fra referanse
    print("\n2. Avvik fra referansemodellen:")
    df_delta = _beregn_delta(df)
    dest = _RESULTS / "modellendringer_nasjonalt_delta.csv"
    df_delta.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_delta)} rader)")

    for utfall in _UTFALL:
        sub = df_delta[df_delta["utfall"] == utfall]
        print(f"\n  {utfall}:")
        for rid in [78, 79, 80, 81]:
            msub = sub[sub["result_id"] == rid]
            print(
                f"    {_MODELLNAVN[rid]:20s}: gj.sn. delta={msub['delta'].mean():.5f}, "
                f"std delta={msub['delta'].std():.5f}, max |delta|={msub['delta'].abs().max():.4f}"
            )

    # 3. Korrelasjon mellom modeller
    print("\n3. Korrelasjon mellom modeller:")
    df_korr = _beregn_korrelasjon(df)
    dest = _RESULTS / "modellendringer_nasjonalt_korrelasjon.csv"
    df_korr.to_csv(dest, index=False)
    print(f"  -> {dest}")

    for utfall in _UTFALL:
        sub = df_korr[df_korr["utfall"] == utfall]
        pivot = sub.pivot(index="modell_1", columns="modell_2", values="pearson_r")
        print(f"\n  {utfall}:")
        print(f"  {pivot.to_string()}")


if __name__ == "__main__":
    main()
