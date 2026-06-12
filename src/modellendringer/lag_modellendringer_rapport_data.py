"""Genererer figurer og tabeller for modellendringer-kapittelet.

Lagrer filer til:
  quarto/modellendringer/figurer/ – PNG-figurer
  quarto/modellendringer/tabeller/ – CSV-tabeller

Kjøres før quarto render:
  uv run python -m src.modellendringer.lag_modellendringer_rapport_data
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ── Stier ─────────────────────────────────────────────────────────────────────
_RESULTS = Path("data/results")
_FIG_DIR = Path("quarto/modellendringer/figurer")
_TBL_DIR = Path("quarto/modellendringer/tabeller")

# ── NAV-farger ────────────────────────────────────────────────────────────────
MODELL_FARGER = {
    "Referanse": "#0067C5",
    "Stillingsrate": "#C30000",
    "Shiftshare": "#06893A",
    "Ledighetsrate": "#FF9100",
    "Ledighetsrate ung": "#59514B",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.35,
        "figure.facecolor": "white",
    }
)

FIG_DPI = 150

_MODELLNAVN = {
    77: "Referanse",
    78: "Stillingsrate",
    79: "Shiftshare",
    80: "Ledighetsrate",
    81: "Ledighetsrate ung",
}

_UTFALL_LABELS = {
    "indikator_jobb3": "Indikator jobb 3 mnd",
    "indikator_jobb12": "Indikator jobb 12 mnd",
    "indikator_atid3": "Indikator arbeidstid 3 mnd",
    "indikator_atid12": "Indikator arbeidstid 12 mnd",
}

_UTFALL_YLABEL = {
    "indikator_jobb3": "Avvik fra forventet (andel)",
    "indikator_jobb12": "Avvik fra forventet (andel)",
    "indikator_atid3": "Avvik fra forventet (timer/uke)",
    "indikator_atid12": "Avvik fra forventet (timer/uke)",
}


# ── Figurer: Nasjonalt ────────────────────────────────────────────────────────


def _fig_nasjonal_tidsserie(df_nasj: pd.DataFrame) -> None:
    """Tidsserie av indikatorer for alle modeller (ett subplot per utfall)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    utfall_list = list(_UTFALL_LABELS.keys())

    for ax, utfall in zip(axes.flat, utfall_list):
        for rid, modell in _MODELLNAVN.items():
            sub = df_nasj[df_nasj["result_id"] == rid].sort_values("beholdningsmaaned")
            ax.plot(
                pd.to_datetime(sub["beholdningsmaaned"]),
                sub[utfall],
                label=modell,
                color=MODELL_FARGER[modell],
                linewidth=1.5,
                alpha=0.85,
            )
        ax.set_title(_UTFALL_LABELS[utfall], fontsize=11)
        ax.set_ylabel(_UTFALL_YLABEL[utfall], fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Nasjonal indikator over tid — alle modellvarianter", fontsize=13)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "nasjonal_tidsserie.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'nasjonal_tidsserie.png'}")


def _fig_nasjonal_delta(df_delta: pd.DataFrame) -> None:
    """Delta fra referansemodellen over tid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    utfall_list = list(_UTFALL_LABELS.keys())

    for ax, utfall in zip(axes.flat, utfall_list):
        sub = df_delta[df_delta["utfall"] == utfall]
        for rid in [78, 79, 80, 81]:
            msub = sub[sub["result_id"] == rid].sort_values("beholdningsmaaned")
            modell = _MODELLNAVN[rid]
            ax.plot(
                pd.to_datetime(msub["beholdningsmaaned"]),
                msub["delta"],
                label=modell,
                color=MODELL_FARGER[modell],
                linewidth=1.2,
                alpha=0.85,
            )
        ax.set_title(_UTFALL_LABELS[utfall], fontsize=11)
        ax.set_ylabel("Δ vs. referanse", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Avvik fra referansemodellen over tid", fontsize=13)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "nasjonal_delta.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'nasjonal_delta.png'}")


# ── Figurer: Volatilitet ──────────────────────────────────────────────────────


def _fig_volatilitet(df_vol: pd.DataFrame) -> None:
    """Barchart av volatilitet (std) per modell og utfall."""
    utfall_list = list(_UTFALL_LABELS.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, utfall in zip(axes, utfall_list):
        sub = df_vol[df_vol["utfall"] == utfall].sort_values("result_id")
        bars = ax.bar(
            sub["modell"],
            sub["std"],
            color=[MODELL_FARGER[m] for m in sub["modell"]],
            alpha=0.85,
        )
        ax.set_title(_UTFALL_LABELS[utfall], fontsize=10)
        ax.set_ylabel("Standardavvik", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

        # Legg til prosentendring vs referanse
        ref_std = sub[sub["result_id"] == 77]["std"].values[0]
        for bar, (_, row) in zip(bars, sub.iterrows()):
            if row["result_id"] != 77:
                pct = (row["std"] - ref_std) / ref_std * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{pct:+.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.suptitle("Volatilitet i nasjonal indikator per modell", fontsize=13)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "volatilitet.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'volatilitet.png'}")


# ── Figurer: Koeffisienter ────────────────────────────────────────────────────


def _fig_koeffisienter_desil(df_lm: pd.DataFrame) -> None:
    """Koeffisientplot for nye LM-variabler (desil) per modell, kun jobb3."""
    modeller = [78, 79, 80, 81]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)

    for ax, rid in zip(axes, modeller):
        sub = df_lm[
            (df_lm["result_id"] == rid) & (df_lm["response_variable"] == "jobb3")
        ]
        sub = sub.sort_values("desil")
        ax.errorbar(
            sub["desil"],
            sub["coefficient"],
            yerr=1.96 * sub["std_err"],
            fmt="o-",
            color=MODELL_FARGER[_MODELLNAVN[rid]],
            capsize=3,
            markersize=5,
        )
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_title(_MODELLNAVN[rid], fontsize=11)
        ax.set_xlabel("Desil", fontsize=9)

    axes[0].set_ylabel("Koeffisient (jobb3)", fontsize=9)
    fig.suptitle(
        "Koeffisienter for nye arbeidsmarkedsvariabler (jobb 3 mnd)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(
        _FIG_DIR / "koeffisienter_desil_jobb3.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'koeffisienter_desil_jobb3.png'}")


# ── Figurer: Stramhet-korrelasjon ─────────────────────────────────────────────


def _fig_stramhet_korrelasjon(df_korr: pd.DataFrame) -> None:
    """Gjennomsnittlig |r| med stramhetsindikator per modell og utfall."""
    utfall_list = list(_UTFALL_LABELS.keys())

    # Filtrer til stramhetsindikator
    sub = df_korr[df_korr["bedrifts_var"] == "stramhetsindikator"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, utfall in zip(axes, utfall_list):
        usub = sub[sub["indikator_var"] == utfall]
        mean_r = (
            usub.groupby(["result_id", "modell"])["r"]
            .apply(lambda x: x.abs().mean())
            .reset_index()
        )
        mean_r = mean_r.sort_values("result_id")
        # bars = ax.bar(
        #     mean_r["modell"],
        #     mean_r["r"],
        #     color=[MODELL_FARGER[m] for m in mean_r["modell"]],
        #     alpha=0.85,
        # )
        ax.set_title(_UTFALL_LABELS[utfall], fontsize=10)
        ax.set_ylabel("Gj.sn. |r|", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.suptitle("Korrelasjon med stramhetsindikator — per modell", fontsize=13)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "stramhet_korrelasjon.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'stramhet_korrelasjon.png'}")


def _fig_stramhet_pooled(df_pooled: pd.DataFrame) -> None:
    """Pooled demeaned korrelasjon per modell (stramhetsindikator)."""
    sub = df_pooled[df_pooled["bedrifts_var"] == "stramhetsindikator"]
    utfall_list = list(_UTFALL_LABELS.keys())

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, utfall in zip(axes, utfall_list):
        usub = sub[sub["indikator_var"] == utfall].sort_values("result_id")
        bars = ax.bar(
            usub["modell"],
            usub["r_demeaned"],
            color=[MODELL_FARGER[m] for m in usub["modell"]],
            alpha=0.85,
        )
        # Signifikansstjerner
        for bar, (_, row) in zip(bars, usub.iterrows()):
            sig = (
                "***"
                if row["p_verdi"] < 0.001
                else (
                    "**"
                    if row["p_verdi"] < 0.01
                    else ("*" if row["p_verdi"] < 0.05 else "")
                )
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                sig,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(_UTFALL_LABELS[utfall], fontsize=10)
        ax.set_ylabel("r (demeaned)", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        "Pooled korrelasjon med stramhetsindikator (år-faste effekter)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "stramhet_pooled.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'stramhet_pooled.png'}")


# ── Tabeller ──────────────────────────────────────────────────────────────────


def _tbl_volatilitet(df_vol: pd.DataFrame) -> None:
    """Tabell med volatilitetsstatistikk."""
    cols = ["utfall", "modell", "gjennomsnitt", "std", "range", "iqr"]
    df_vol[cols].to_csv(_TBL_DIR / "volatilitet.csv", index=False)
    print(f"  -> {_TBL_DIR / 'volatilitet.csv'}")


def _tbl_stramhet_sammendrag(df_samm: pd.DataFrame) -> None:
    """Tabell med stramhet-korrelasjonssammendrag."""
    df_samm.to_csv(_TBL_DIR / "stramhet_sammendrag.csv", index=False)
    print(f"  -> {_TBL_DIR / 'stramhet_sammendrag.csv'}")


def _tbl_stramhet_korrelasjoner(df_korr: pd.DataFrame) -> None:
    """Kompakt korrelasjontabell per modell (stramhetsindikator × indikatorvariabler)."""
    sub = df_korr[df_korr["bedrifts_var"] == "stramhetsindikator"]
    pivot = sub.pivot_table(
        index=["modell", "aar"],
        columns="indikator_var",
        values="r",
        aggfunc="first",
    ).round(3)
    pivot.to_csv(_TBL_DIR / "stramhet_korrelasjoner_per_aar.csv")
    print(f"  -> {_TBL_DIR / 'stramhet_korrelasjoner_per_aar.csv'}")


def _tbl_modelloversikt() -> None:
    """Oversiktstabell over modeller og deres variabler."""
    rows = [
        {
            "result_id": 77,
            "Modell": "Referanse",
            "Ekstra LM-variabel": "(kun tilstrømming)",
            "Beskrivelse": "Gjeldende produksjonsmodell",
        },
        {
            "result_id": 78,
            "Modell": "Stillingsrate",
            "Ekstra LM-variabel": "Stillingrate per BA-region (desiler)",
            "Beskrivelse": "Antall stillingsutlysninger / befolkning i bo- og arbeidsmarkedsregion",
        },
        {
            "result_id": 79,
            "Modell": "Shiftshare",
            "Ekstra LM-variabel": "Shiftshare per BA-region (desiler)",
            "Beskrivelse": "Instrumentvariabel: nasjonale sektortrender × lokal næringsstruktur",
        },
        {
            "result_id": 80,
            "Modell": "Ledighetsrate",
            "Ekstra LM-variabel": "Ledighetsrate per BA-region (desiler)",
            "Beskrivelse": "Arbeidsledighet som andel av arbeidsstyrken",
        },
        {
            "result_id": 81,
            "Modell": "Ledighetsrate ung",
            "Ekstra LM-variabel": "Ledighetsrate unge per BA-region (desiler)",
            "Beskrivelse": "Arbeidsledighet for personer under 30 som andel av arbeidsstyrken",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(_TBL_DIR / "modelloversikt.csv", index=False)
    print(f"  -> {_TBL_DIR / 'modelloversikt.csv'}")


# ── Hovedfunksjon ─────────────────────────────────────────────────────────────


def main() -> None:
    """Generer alle figurer og tabeller for modellendringer-kapittelet."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    _TBL_DIR.mkdir(parents=True, exist_ok=True)

    # Les data
    print("Leser data for figurer/tabeller...")
    df_nasj = pd.read_csv(_RESULTS / "modellendringer_nasjonalt_volatilitet.csv")
    df_delta = pd.read_csv(_RESULTS / "modellendringer_nasjonalt_delta.csv")
    df_nasj_raw = pd.read_csv("data/processed/modellendringer_nasjonalt.csv")
    df_korr = pd.read_csv(_RESULTS / "modellendringer_stramhet_korrelasjoner.csv")
    df_pooled = pd.read_csv(_RESULTS / "modellendringer_stramhet_pooled.csv")
    df_samm = pd.read_csv(_RESULTS / "modellendringer_stramhet_sammendrag.csv")
    df_lm = pd.read_csv(_RESULTS / "modellendringer_koeffisienter_lm.csv")

    # Figurer
    print("\nGenererer figurer...")
    _fig_nasjonal_tidsserie(df_nasj_raw)
    _fig_nasjonal_delta(df_delta)
    _fig_volatilitet(df_nasj)
    _fig_koeffisienter_desil(df_lm)
    _fig_stramhet_korrelasjon(df_korr)
    _fig_stramhet_pooled(df_pooled)

    # Tabeller
    print("\nGenererer tabeller...")
    _tbl_modelloversikt()
    _tbl_volatilitet(df_nasj)
    _tbl_stramhet_sammendrag(df_samm)
    _tbl_stramhet_korrelasjoner(df_korr)

    print("\nFerdig!")


if __name__ == "__main__":
    main()
