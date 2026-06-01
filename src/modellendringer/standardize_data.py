"""Standardiserer data for modellsammenlikning.

Leser koeffisient- og indikatordata fra data/raw/koeffisienter/,
mapper result_id til modellnavn, og normaliserer datoformat.

Produserer:
  data/processed/modellendringer_nasjonalt.csv
  data/processed/modellendringer_regionalt.csv
  data/processed/modellendringer_koeffisienter.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_RAW = Path("data/raw/koeffisienter")
_PROCESSED = Path("data/processed")

MODELLNAVN: dict[int, str] = {
    77: "Referanse",
    78: "Stillingsrate",
    79: "Shiftshare",
    80: "Ledighetsrate",
    81: "Ledighetsrate ung",
}

# Hvilken ekstra arbeidsmarkedsvariabel er lagt til i hver modell (utover tilstrømming)
EKSTRA_VARIABEL: dict[int, str] = {
    77: "(kun tilstrømming)",
    78: "stillingsrate_tilgang",
    79: "shiftshare",
    80: "ledighetsrate",
    81: "ledighetsrate_ung",
}


def _les_og_mapp(filnavn: str) -> pd.DataFrame:
    """Les CSV, legg til modellnavn, og normaliser datoformat."""
    df = pd.read_csv(_RAW / filnavn)
    df["result_id"] = df["result_id"].astype(int)
    df["modell"] = df["result_id"].map(MODELLNAVN)

    if "beholdningsmaaned" in df.columns:
        df["beholdningsmaaned"] = pd.to_datetime(df["beholdningsmaaned"], utc=True)
        df["aar"] = df["beholdningsmaaned"].dt.year
        df["maaned"] = df["beholdningsmaaned"].dt.month

    return df


def main() -> None:
    """Standardiser alle tre datasett."""
    _PROCESSED.mkdir(parents=True, exist_ok=True)

    # Nasjonalt
    print("Leser nasjonalt...")
    df_nasj = _les_og_mapp("indikator_data_nasjonalt.csv")
    dest = _PROCESSED / "modellendringer_nasjonalt.csv"
    df_nasj.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_nasj)} rader, {df_nasj['modell'].nunique()} modeller)")

    # Regionalt
    print("Leser regionalt...")
    df_reg = _les_og_mapp("indikator_data_region.csv")
    dest = _PROCESSED / "modellendringer_regionalt.csv"
    df_reg.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_reg)} rader, {df_reg['org_sted'].nunique()} regioner)")

    # Koeffisienter
    print("Leser koeffisienter...")
    df_koeff = _les_og_mapp("indikator_data_model.csv")
    df_koeff["ekstra_variabel"] = df_koeff["result_id"].map(EKSTRA_VARIABEL)
    dest = _PROCESSED / "modellendringer_koeffisienter.csv"
    df_koeff.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df_koeff)} rader)")

    # Oppsummering
    print("\nModelloversikt:")
    for rid, navn in MODELLNAVN.items():
        n_nasj = len(df_nasj[df_nasj["result_id"] == rid])
        n_reg = len(df_reg[df_reg["result_id"] == rid])
        n_koeff = len(df_koeff[df_koeff["result_id"] == rid])
        print(f"  {rid}: {navn:<20} nasj={n_nasj}, reg={n_reg}, koeff={n_koeff}")


if __name__ == "__main__":
    main()
