"""Analyserer koeffisienter for de ulike modellvariantene.

Undersøker:
  1. Koeffisientstørrelse og signifikans for de nye arbeidsmarkedsvariablene
  2. Monotonitet i desil-koeffisientene (stigende med desil → forventet gradient)
  3. Endring i tilstrømming-koeffisientene når andre LM-variabler legges til

Produserer:
  data/results/modellendringer_koeffisienter_lm.csv   — koeffisienter for LM-variabler
  data/results/modellendringer_tilstromming_endring.csv — endring i tilstrømming-koeff
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PROCESSED = Path("data/processed")
_RESULTS = Path("data/results")

# Mønster for å identifisere arbeidsmarkedsvariabler (ikke-tilstrømming)
_LM_PREFIKS = ["stillingsrate", "shiftshare", "ledighetsrate"]

_MODELLNAVN = {
    77: "Referanse",
    78: "Stillingsrate",
    79: "Shiftshare",
    80: "Ledighetsrate",
    81: "Ledighetsrate ung",
}


def _er_lm_variabel(variabel: str) -> bool:
    """Sjekk om variabelen er en arbeidsmarkedsvariabel (ekskl. tilstrømming)."""
    return any(variabel.startswith(p) for p in _LM_PREFIKS)


def _er_tilstromming(variabel: str) -> bool:
    return variabel.startswith("tilstromming")


def _ekstraher_desil(variabel: str) -> int | None:
    """Hent ut desilverdien fra variabelnavn som 'stillingsrate_tilgang12_desil_5'."""
    parts = variabel.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _analyser_lm_koeffisienter(df: pd.DataFrame) -> pd.DataFrame:
    """Filtrer ut koeffisientene for de nye LM-variablene per modell og utfall."""
    mask = df["variable"].apply(_er_lm_variabel)
    df_lm = df[mask].copy()
    df_lm["desil"] = df_lm["variable"].apply(_ekstraher_desil)
    df_lm["modell"] = df_lm["result_id"].map(_MODELLNAVN)
    return df_lm.sort_values(["result_id", "response_variable", "desil"])


def _analyser_tilstromming_endring(df: pd.DataFrame) -> pd.DataFrame:
    """Sammenlign tilstrømming-koeffisienter mellom referanse og utvidede modeller."""
    mask = df["variable"].apply(_er_tilstromming)
    df_ts = df[mask].copy()
    df_ts["modell"] = df_ts["result_id"].map(_MODELLNAVN)
    df_ts["desil"] = df_ts["variable"].apply(_ekstraher_desil)

    # Beregn avvik fra referansemodellen (result_id 77)
    ref = df_ts[df_ts["result_id"] == 77][
        ["response_variable", "variable", "coefficient"]
    ].rename(columns={"coefficient": "koeff_referanse"})

    merged = df_ts.merge(ref, on=["response_variable", "variable"], how="left")
    merged["koeff_endring"] = merged["coefficient"] - merged["koeff_referanse"]
    merged["koeff_endring_pst"] = (
        merged["koeff_endring"] / merged["koeff_referanse"].abs() * 100
    ).round(1)

    return merged.sort_values(["response_variable", "variable", "result_id"])


def main() -> None:
    """Kjør koeffisientanalysen."""
    _RESULTS.mkdir(parents=True, exist_ok=True)

    print("Leser koeffisientdata...")
    df = pd.read_csv(_PROCESSED / "modellendringer_koeffisienter.csv")

    # 1. LM-koeffisienter
    print("\n1. Koeffisienter for nye arbeidsmarkedsvariabler:")
    df_lm = _analyser_lm_koeffisienter(df)
    dest = _RESULTS / "modellendringer_koeffisienter_lm.csv"
    df_lm.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_lm)} rader)")

    for rid in [78, 79, 80, 81]:
        sub = df_lm[
            (df_lm["result_id"] == rid) & (df_lm["response_variable"] == "jobb3")
        ]
        sub_sorted = sub.sort_values("desil")
        print(f"\n  {_MODELLNAVN[rid]} (jobb3):")
        for _, row in sub_sorted.iterrows():
            sig = (
                "***"
                if row["p_value"] < 0.001
                else (
                    "**"
                    if row["p_value"] < 0.01
                    else ("*" if row["p_value"] < 0.05 else "")
                )
            )
            print(
                f"    desil {row['desil']:>3}: koeff={row['coefficient']:>8.4f}  t={row['t_value']:>7.2f} {sig}"
            )

    # 2. Tilstrømming-endring
    print("\n2. Endring i tilstrømming-koeffisienter:")
    df_ts = _analyser_tilstromming_endring(df)
    dest = _RESULTS / "modellendringer_tilstromming_endring.csv"
    df_ts.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_ts)} rader)")

    # Oppsummering: gjennomsnittlig absolutt endring per modell
    for rid in [78, 79, 80, 81]:
        sub = df_ts[df_ts["result_id"] == rid]
        mean_abs_change = sub["koeff_endring"].abs().mean()
        mean_pst_change = sub["koeff_endring_pst"].abs().mean()
        print(
            f"  {_MODELLNAVN[rid]:20s}: gj.sn. |endring|={mean_abs_change:.4f} ({mean_pst_change:.1f}%)"
        )


if __name__ == "__main__":
    main()
