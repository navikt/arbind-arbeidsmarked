"""Analyserer SSB-stramhetsindikator per næring og sammenligner med bedriftsundersøkelsen.

To analytiske deler:

  A. Sammenligning: SSB-stramhetsindikator (v/(v+u), offisiell statistikk) vs
     bedriftsundersøkelsens stramhetsindikator (rapportert arbeidskraftmangel/
     sysselsatte) per næring, 2021–2026.
     Belyser konseptuell forskjell mellom annonserte ledige stillinger og
     rapportert mangel på arbeidskraft.

  B. Regresjonsanalyse: Næringsvise SSB-stramhetsindikator vs Nav-indikator
     (kvartalsvis, 2021–2025). Tester om næringsvis stramhet predikerer
     indikatoravvik — det vil si om næringssammensetningen skaper bias.

Produserer:
  quarto/stramhet/figurer/ssb_bedrift_scatter.png
  quarto/stramhet/figurer/ssb_bedrift_naering_tidsserie.png
  quarto/stramhet/figurer/ssb_naering_korrelasjon_heatmap.png
  quarto/stramhet/figurer/ssb_naering_regresjon_beta.png
  quarto/stramhet/tabeller/ssb_bedrift_sammenlikning.csv
  quarto/stramhet/tabeller/ssb_naering_regresjon.csv

Kjøres:
  uv run python -m src.stramhet.analyse_ssb_stramhet
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ── Stier ─────────────────────────────────────────────────────────────────────
_PROCESSED = Path("data/processed")
_FIG_DIR = Path("quarto/stramhet/figurer")
_TBL_DIR = Path("quarto/stramhet/tabeller")

# ── NAV-farger ─────────────────────────────────────────────────────────────────
NAV_BLÅ = "#0067C5"
NAV_RØD = "#C30000"
NAV_GRØNN = "#06893A"
NAV_ORANSJE = "#FF9100"
NAV_GRÅ = "#59514B"

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

# ── Felles næringsinndeling: 15 sammenliknbare grupper ─────────────────────────
# Nøkkel = felles gruppe-label brukt i begge datasett etter mapping
_FELLES_NAERINGER = [
    "Jordbruk, skogbruk og fiske",
    "Bergverksdrift og utvinning",
    "Industri",
    "Elektrisitet, vann og renovasjon",
    "Bygge- og anlegg",
    "Varehandel",
    "Transport og lagring",
    "Overnatting og servering",
    "Informasjon og kommunikasjon",
    "Finans og forsikring",
    "Eiendom og tjenesteyting",
    "Offentlig administrasjon",
    "Undervisning",
    "Helse- og sosialtjenester",
    "Kultur og personlige tjenester",
]

# ── Mapping: bedriftsundersøkelsen næringsnavn → felles gruppe ─────────────────
_BEDRIFT_MAP: dict[str, str] = {
    # Jordbruk
    "Jordbruk, skogbruk og fiske": "Jordbruk, skogbruk og fiske",
    # Bergverk
    "Bergverksdrift og utvinning": "Bergverksdrift og utvinning",
    # Industri (varierer mellom år)
    "Industrien totalt": "Industri",
    "Industrien": "Industri",
    "Industrien samlet": "Industri",
    # Elektrisitet
    "Elektrisitet, vann og renovasjon": "Elektrisitet, vann og renovasjon",
    # Bygge og anlegg
    "Bygge- og anleggsvirksomhet": "Bygge- og anlegg",
    # Varehandel
    "Varehandel, motorvognreparasjoner": "Varehandel",
    "Varehandel": "Varehandel",
    # Transport
    "Transport og lagring": "Transport og lagring",
    # Overnatting
    "Overnattings- og serveringsvirksomhet": "Overnatting og servering",
    # IKT
    "Informasjon og kommunikasjon": "Informasjon og kommunikasjon",
    # Finans
    "Finansierings- og forsikringsvirksomhet": "Finans og forsikring",
    "Finansiell tjenesteyting": "Finans og forsikring",
    # Eiendom og tjenesteyting (kombinert kategori)
    "Eiendomsdrift, forretningsmessig og faglig tjenesteyting": "Eiendom og tjenesteyting",
    "Eiendomsvirksomhet, faglig, vitenskapelig, teknisk og forretningsmessig tjenesteyting": (
        "Eiendom og tjenesteyting"
    ),
    # Offentlig administrasjon
    "Offentlig administrasjon og forsvar, og trygdeordninger underlagt offentlig forvaltning": (
        "Offentlig administrasjon"
    ),
    "Offentlig forvaltning": "Offentlig administrasjon",
    "Offentlig administrasjon og forsvar": "Offentlig administrasjon",
    # Undervisning
    "Undervisning": "Undervisning",
    # Helse
    "Helse- og sosialtjeneste": "Helse- og sosialtjenester",
    "Helse- og sosialtjenester": "Helse- og sosialtjenester",
    # Kultur/personlig
    "Personlig tjenesteyting": "Kultur og personlige tjenester",
    "Kultur, idrett, fritid og annen tjenesteyting": "Kultur og personlige tjenester",
}

# ── Mapping: SSB naering_label → felles gruppe (evt. aggregering) ─────────────
# SSB-kategorier 68, 69-75 og 77-82 slås sammen til "Eiendom og tjenesteyting"
# SSB-kategorier 90-93 og 94-96 slås sammen til "Kultur og personlige tjenester"
_SSB_MAP: dict[str, str] = {
    "Jordbruk, skogbruk og fiske": "Jordbruk, skogbruk og fiske",
    "Bergverksdrift og utvinning": "Bergverksdrift og utvinning",
    "Industri": "Industri",
    "Elektrisitet, vann og renovasjon": "Elektrisitet, vann og renovasjon",
    "Bygge- og anleggsvirksomhet": "Bygge- og anlegg",
    "Varehandel": "Varehandel",
    "Transport og lagring": "Transport og lagring",
    "Overnatting og servering": "Overnatting og servering",
    "Informasjon og kommunikasjon": "Informasjon og kommunikasjon",
    "Finans og forsikring": "Finans og forsikring",
    "Omsetning av fast eiendom": "Eiendom og tjenesteyting",
    "Faglig og teknisk tjenesteyting": "Eiendom og tjenesteyting",
    "Forretningsmessig tjenesteyting": "Eiendom og tjenesteyting",
    "Offentlig administrasjon": "Offentlig administrasjon",
    "Undervisning": "Undervisning",
    "Helse- og sosialtjenester": "Helse- og sosialtjenester",
    "Kultur og underholdning": "Kultur og personlige tjenester",
    "Annen tjenesteyting": "Kultur og personlige tjenester",
}


def _kvartal_til_aar_kvartal(kvartal: str) -> tuple[int, int]:
    """Konverter '2021K1' til (2021, 1)."""
    yr, kv = kvartal.split("K")
    return int(yr), int(kv)


# ─────────────────────────────────────────────────────────────────────────────
# Del A: Last og forbered sammenligning
# ─────────────────────────────────────────────────────────────────────────────


def _last_ssb_naering() -> pd.DataFrame:
    """Les SSB næring-stramhet, aggreger til felles næringer og til Q1 per år."""
    df = pd.read_csv(_PROCESSED / "ssb_stramhet_naering.csv")

    # Map til felles næringer (aggreger tallverdier for kategorier som slås sammen)
    df["felles"] = df["naering_label"].map(_SSB_MAP)
    df = df.dropna(subset=["felles"])

    # For sammenslåtte kategorier: aggreger stillinger og sysselsatte, reberegn stramhet
    df_raw = pd.read_csv(_PROCESSED / "ssb_stramhet_naering.csv").merge(
        df[["kvartal", "naering_label", "felles"]],
        on=["kvartal", "naering_label"],
        how="inner",
    )
    df_agg = (
        df_raw.groupby(["kvartal", "felles"])[["ledige_stillinger", "sysselsatte"]]
        .sum()
        .reset_index()
    )
    df_agg["stramhetsindikator_ssb"] = df_agg["ledige_stillinger"] / (
        df_agg["ledige_stillinger"] + df_agg["sysselsatte"]
    )

    # Parse år og kvartal
    df_agg[["aar", "kvartal_nr"]] = df_agg["kvartal"].apply(
        lambda x: pd.Series(_kvartal_til_aar_kvartal(x))
    )
    return df_agg


def _last_bedrift_naering() -> pd.DataFrame:
    """Les bedrift næring-stramhet, filtrer ikke-undernæringer og map til felles."""
    df = pd.read_csv(_PROCESSED / "mangel_per_naring.csv")
    df = df[~df["er_undernaring"]].copy()
    df["felles"] = df["naring"].map(_BEDRIFT_MAP)
    df = df.dropna(subset=["felles"])
    df = df[["aar", "felles", "stramhetsindikator", "mangel_antall"]].rename(
        columns={"stramhetsindikator": "stramhetsindikator_bedrift"}
    )
    # Bedrift rapporterer i prosent (1.5 = 1.5 %); konverter til desimalbrøk
    df["stramhetsindikator_bedrift"] = df["stramhetsindikator_bedrift"] / 100
    return df


def _lag_sammenlikning(
    df_ssb: pd.DataFrame,
    df_bedrift: pd.DataFrame,
    drop_years: list[int] | None = None,
) -> pd.DataFrame:
    """Slå sammen SSB Q1 med bedrift, begge på år × felles næring."""
    ssb_q1 = df_ssb[df_ssb["kvartal_nr"] == 1][
        ["aar", "felles", "stramhetsindikator_ssb"]
    ]
    df_b = df_bedrift.copy()
    if drop_years:
        ssb_q1 = ssb_q1[~ssb_q1["aar"].isin(drop_years)]
        df_b = df_b[~df_b["aar"].isin(drop_years)]
    return ssb_q1.merge(df_b, on=["aar", "felles"], how="inner")


# ─────────────────────────────────────────────────────────────────────────────
# Del A: Figurer og tabeller – sammenligning
# ─────────────────────────────────────────────────────────────────────────────

# Farger per næring for scatterplot
_NAERING_COLORS = {
    næring: color
    for næring, color in zip(
        _FELLES_NAERINGER,
        plt.colormaps["tab20"].colors,
    )
}


def fig_sammenligning_scatter(df: pd.DataFrame, suffix: str = "") -> None:
    """Scatter: SSB stramhetsindikator (x) vs bedrift stramhetsindikator (y)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for naering, grp in df.groupby("felles"):
        ax.scatter(
            grp["stramhetsindikator_ssb"],
            grp["stramhetsindikator_bedrift"],
            label=naering,
            color=_NAERING_COLORS.get(str(naering), NAV_GRÅ),
            s=60,
            alpha=0.8,
            zorder=3,
        )

    # Regresjonslinje
    x = df["stramhetsindikator_ssb"].values
    y = df["stramhetsindikator_bedrift"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 2:
        slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
        x_range = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(
            x_range,
            intercept + slope * x_range,
            color=NAV_RØD,
            lw=1.5,
            zorder=2,
            label=f"Reg.linje (r={r:.2f}, p={p:.3f})",
        )

    ax.set_xlabel(
        "SSB stramhetsindikator\n(stillinger / (stillinger + sysselsatte))", fontsize=10
    )
    ax.set_ylabel(
        "Bedrift stramhetsindikator\n(mangel på arbeidskraft / sysselsatte, omregnet fra %)",
        fontsize=10,
    )
    ax.set_title(
        "Sammenligning av stramhetsindikator per næring\n(Q1 per år, 2021–2026, begge mål på desimalskala)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.7)
    fig.tight_layout()
    path = _FIG_DIR / f"ssb_bedrift_scatter{suffix}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig_naering_tidsserie(df: pd.DataFrame, suffix: str = "") -> None:
    """Tidsserie per næring: SSB Q1 og bedrift stramhetsindikator side om side."""
    naeringer = sorted(df["felles"].unique())
    n = len(naeringer)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2), squeeze=False)
    fig.suptitle(
        "Stramhetsindikator per næring: SSB (blå) vs bedriftsundersøkelsen (rød)\n(begge på desimalskala; bedrift-verdier delt på 100)",
        fontsize=12,
        y=1.01,
    )

    for idx, naering in enumerate(naeringer):
        ax = axes[idx // ncols][idx % ncols]
        grp = df[df["felles"] == naering].sort_values("aar")
        ax.plot(
            grp["aar"],
            grp["stramhetsindikator_ssb"],
            color=NAV_BLÅ,
            marker="o",
            ms=5,
            label="SSB",
        )
        ax.plot(
            grp["aar"],
            grp["stramhetsindikator_bedrift"],
            color=NAV_RØD,
            marker="s",
            ms=5,
            label="Bedrift",
        )
        ax.set_title(naering, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_xlabel("")
        if idx == 0:
            ax.legend(fontsize=7)

    # Skjul tomme subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    path = _FIG_DIR / f"ssb_bedrift_naering_tidsserie{suffix}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Del B: Regresjonsanalyse – næringsvise SSB-stramhet vs Nav-indikator
# ─────────────────────────────────────────────────────────────────────────────

_INDIKATOR_LABELS = {
    "faktisk_jobb3": "Faktisk jobb 3 mnd",
    "faktisk_jobb12": "Faktisk jobb 12 mnd",
    "indikator_jobb3": "Indikatoravvik jobb 3 mnd",
    "indikator_jobb12": "Indikatoravvik jobb 12 mnd",
    "faktisk_atid3": "Faktisk atid 3 mnd",
    "faktisk_atid12": "Faktisk atid 12 mnd",
    "indikator_atid3": "Indikatoravvik atid 3 mnd",
    "indikator_atid12": "Indikatoravvik atid 12 mnd",
}


def _last_indikator_kvartal() -> pd.DataFrame:
    """Les national indikatordata og aggreger til kvartalsgjennomsnnitt."""
    df = pd.read_csv("data/raw/indikator_data_nasjonalt.csv")
    df["dato"] = pd.to_datetime(df["beholdningsmaaned"], utc=True).dt.tz_localize(None)
    df["aar"] = df["dato"].dt.year
    df["kvartal_nr"] = df["dato"].dt.month.map(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
    )
    df["kvartal"] = df["aar"].astype(str) + "K" + df["kvartal_nr"].astype(str)

    # Pivot utfall til bredt format
    df_pivot = df.pivot_table(
        index=["kvartal", "aar", "kvartal_nr"],
        columns="utfall",
        values=["faktisk", "indikator"],
        aggfunc="mean",
    ).reset_index()
    df_pivot.columns = [
        "_".join(c).strip("_") if c[1] else c[0] for c in df_pivot.columns
    ]
    # Rename: faktisk_jobb3 osv.
    renames = {}
    for col in df_pivot.columns:
        if "_jobb3" in col or "_jobb12" in col or "_atid3" in col or "_atid12" in col:
            renames[col] = col  # already correct
    return df_pivot


def _lag_regresjonsdata(
    df_ssb: pd.DataFrame,
    df_ind: pd.DataFrame,
    drop_years: list[int] | None = None,
) -> pd.DataFrame:
    """Slå sammen SSB kvartal-næring (wide) med indikator-kvartal."""
    df_ssb_f = df_ssb.copy()
    df_ind_f = df_ind.copy()
    if drop_years:
        df_ssb_f = df_ssb_f[~df_ssb_f["aar"].isin(drop_years)]
        df_ind_f = df_ind_f[~df_ind_f["aar"].isin(drop_years)]
    # Pivot SSB til bredt format: én kolonne per næring
    df_wide = (
        df_ssb_f[["kvartal", "felles", "stramhetsindikator_ssb"]]
        .pivot_table(index="kvartal", columns="felles", values="stramhetsindikator_ssb")
        .reset_index()
    )
    df_wide.columns.name = None
    # Fiks kolonnenavn til sikre strenger
    df_wide.columns = [
        c
        if c == "kvartal"
        else f"ssb_{c.lower().replace(' ', '_').replace('-', '_').replace(',', '')}"
        for c in df_wide.columns
    ]

    merged = df_ind_f.merge(df_wide, on="kvartal", how="inner")
    return merged


def _kjor_regresjoner(df: pd.DataFrame, ssb_cols: list[str]) -> pd.DataFrame:
    """Kjør bivariate OLS-regresjoner: hver næring vs hvert indikatorutfall."""
    utfall_cols = [
        c
        for c in df.columns
        if any(
            c.startswith(p)
            for p in [
                "faktisk_jobb",
                "faktisk_atid",
                "indikator_jobb",
                "indikator_atid",
            ]
        )
    ]
    records = []
    for utfall in utfall_cols:
        for ssb_col in ssb_cols:
            sub = df[[utfall, ssb_col]].dropna()
            if len(sub) < 5:
                continue
            x = sub[ssb_col].values
            y = sub[utfall].values
            slope, intercept, r, p, se = stats.linregress(x, y)
            records.append(
                {
                    "utfall": utfall,
                    "utfall_label": _INDIKATOR_LABELS.get(utfall, utfall),
                    "naering_col": ssb_col,
                    "naering_label": ssb_col.replace("ssb_", "")
                    .replace("_", " ")
                    .title(),
                    "beta": round(slope, 4),
                    "se": round(se, 4),
                    "r": round(r, 3),
                    "p": round(p, 4),
                    "r2": round(r**2, 3),
                    "n": len(sub),
                }
            )
    return pd.DataFrame(records)


def fig_korrelasjon_heatmap(df_reg: pd.DataFrame, suffix: str = "") -> None:
    """Heatmap av Pearson r mellom næringsvise SSB-stramhet og indikatorutfall."""
    # Filtrer til indikatoravvik (den interessante delen)
    sub = df_reg[df_reg["utfall"].str.startswith("indikator_")].copy()

    pivot = sub.pivot_table(index="naering_label", columns="utfall_label", values="r")

    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = max(abs(pivot.values[np.isfinite(pivot.values)]).max(), 0.1)
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Legg inn r-verdier i cellene
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(val) > vmax * 0.6 else "black",
                )

    ax.set_title(
        "Pearson r: næringsvise SSB-stramhetsindikator vs Nav-indikatoravvik\n(kvartalsvis, 2021–2025)",
        fontsize=11,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    path = _FIG_DIR / f"ssb_naering_korrelasjon_heatmap{suffix}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig_beta_naering(df_reg: pd.DataFrame, suffix: str = "") -> None:
    """Koeffisientplot: β per næring for hvert indikatorutfall (indikatoravvik)."""
    # Bruk indikatoravvikene som er mest interessante for bias-spørsmålet
    avvik_utfall = [u for u in df_reg["utfall"].unique() if u.startswith("indikator_")]
    utfall_labels = {
        u: _INDIKATOR_LABELS[u] for u in avvik_utfall if u in _INDIKATOR_LABELS
    }

    n_utfall = len(avvik_utfall)
    fig, axes = plt.subplots(1, n_utfall, figsize=(4 * n_utfall, 7), sharey=True)
    if n_utfall == 1:
        axes = [axes]

    for ax, utfall in zip(axes, avvik_utfall):
        sub = df_reg[df_reg["utfall"] == utfall].sort_values("r")
        colors = [NAV_RØD if p < 0.05 else NAV_GRÅ for p in sub["p"]]
        ax.barh(sub["naering_label"], sub["beta"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(utfall_labels.get(utfall, utfall), fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("β (OLS)", fontsize=8)
        ax.annotate(
            "Rød = p<0.05",
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            ha="right",
            fontsize=7,
            color=NAV_RØD,
        )

    axes[0].set_ylabel("")
    fig.suptitle(
        "Regresjonskoeffisienter β: SSB næringsstramhet → Nav-indikatoravvik",
        fontsize=11,
    )
    fig.tight_layout()
    path = _FIG_DIR / f"ssb_naering_regresjon_beta{suffix}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Del C: Panelregresjon med næring fixed effects
# ─────────────────────────────────────────────────────────────────────────────


def _lag_panel_data(
    df_ssb: pd.DataFrame,
    df_ind: pd.DataFrame,
    drop_years: list[int] | None = None,
) -> pd.DataFrame:
    """Lag langt format for panelregresjon: kvartal × næring.

    Outcome (indikatoravvik) varierer bare over kvartal (nasjonalt snitt).
    Prediktor (SSB-stramhet) varierer over kvartal × næring.
    """
    df_ssb_f = df_ssb.copy()
    df_ind_f = df_ind.copy()
    if drop_years:
        df_ssb_f = df_ssb_f[~df_ssb_f["aar"].isin(drop_years)]
        df_ind_f = df_ind_f[~df_ind_f["aar"].isin(drop_years)]
    ssb_long = df_ssb_f[["kvartal", "felles", "stramhetsindikator_ssb"]].copy()
    avvik_cols = [c for c in df_ind_f.columns if c.startswith("indikator_")]
    ind_sub = df_ind_f[["kvartal"] + avvik_cols].copy()
    return ssb_long.merge(ind_sub, on="kvartal", how="inner")


def _kjor_panel_regresjon(df: pd.DataFrame) -> pd.DataFrame:
    """Panelregresjon med næring FE (within-estimator) for hvert indikatoravvik.

    Modell:  y_t = α_j + β·stramhet_{j,t} + ε_{j,t}
    - α_j: næringsfaste effekter (within-transformasjon)
    - β: identifisert fra variasjon *innen* næring over tid
    - SE: HC1 (heteroskedastisitets-robust)

    NB (pseudo-replikasjon): indikatoravviket er en nasjonal serie som varierer
    kun over tid (kvartal) og er koblet på hver nærings-rad. Observasjoner innen
    samme kvartal deler dermed utfallsverdi, slik at HC1-SE kan undervurdere
    usikkerheten. Som robusthetssjekk rapporteres også SE/p klynget på kvartal
    (``se_klynge_kvartal``/``p_klynge_kvartal``) sammen med antall klynger
    (``n_kvartaler``); få klynger gjør den klyngede SE-en mindre pålitelig.

    Rapporterer: β, SE, t, p, adj. R², within-R² og mellom-R².
    """
    avvik_cols = [c for c in df.columns if c.startswith("indikator_")]
    records = []

    for utfall in avvik_cols:
        sub = (
            df[["felles", "kvartal", "stramhetsindikator_ssb", utfall]].dropna().copy()
        )

        # ── Within-transformasjon (demean per næring) ──────────────────────
        sub["y_dm"] = sub[utfall] - sub.groupby("felles")[utfall].transform("mean")
        sub["x_dm"] = sub["stramhetsindikator_ssb"] - sub.groupby("felles")[
            "stramhetsindikator_ssb"
        ].transform("mean")

        # Justert for tapte frihetsgrader fra næring-FE
        n_naeringer = sub["felles"].nunique()
        X_within = sm.add_constant(sub["x_dm"])
        within_res = sm.OLS(sub["y_dm"], X_within).fit(cov_type="HC1")

        # Robusthet mot pseudo-replikasjon: klynge-robust SE på kvartal, siden
        # utfallet er konstant innen hvert kvartal (nasjonal serie).
        n_kvartaler = sub["kvartal"].nunique()
        within_klynge = sm.OLS(sub["y_dm"], X_within).fit(
            cov_type="cluster", cov_kwds={"groups": sub["kvartal"]}
        )
        se_klynge = within_klynge.bse["x_dm"]
        p_klynge = within_klynge.pvalues["x_dm"]

        # Within-R²: R² fra det within-transformerte systemet (uten const)
        ss_res = (sub["y_dm"] - within_res.fittedvalues).var()
        ss_tot = sub["y_dm"].var()
        r2_within = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        # ── Mellom-næring-R² er ikke meningsfylt i dette designet ────────────
        # Outcome varierer bare over tid (nasjonalt snitt), ikke over næringer
        # for et gitt kvartal. Næring-gjennomsnittet av y er dermed (nesten)
        # identisk for alle næringer → between-R² er udefinert / null.

        # ── Full OLS med næring-dummies (for adj. R² og F-test) ───────────
        dummies = pd.get_dummies(sub["felles"], drop_first=True, dtype=float)
        X_full = pd.concat([sub[["stramhetsindikator_ssb"]], dummies], axis=1)
        X_full = sm.add_constant(X_full)
        full_res = sm.OLS(sub[utfall], X_full).fit(cov_type="HC1")

        beta = within_res.params["x_dm"]
        se = within_res.bse["x_dm"]
        tstat = within_res.tvalues["x_dm"]
        pval = within_res.pvalues["x_dm"]

        # Justert R² manuelt (within-estimator taper n_naeringer-1 DF)
        n = len(sub)
        k = 1  # antall regressorer (stramhet)
        adj_r2_within = 1 - (1 - r2_within) * (n - 1) / (n - k - n_naeringer)

        records.append(
            {
                "utfall": utfall,
                "utfall_label": _INDIKATOR_LABELS.get(utfall, utfall),
                "beta": round(beta, 4),
                "se": round(se, 4),
                "t": round(tstat, 2),
                "p": round(pval, 4),
                "se_klynge_kvartal": round(se_klynge, 4),
                "p_klynge_kvartal": round(p_klynge, 4),
                "n_kvartaler": n_kvartaler,
                "r2_within": round(r2_within, 3),
                "r2_within_adj": round(adj_r2_within, 3),
                "r2_full_adj": round(full_res.rsquared_adj, 3),
                "n": n,
                "n_naeringer": n_naeringer,
            }
        )

    return pd.DataFrame(records)


def fig_panel_koeffisienter(
    df_panel: pd.DataFrame, df_bivariat: pd.DataFrame, suffix: str = ""
) -> None:
    """Koeffisientplot: panel-β med 95 % KI vs fordeling av bivariate β per næring."""
    avvik_utfall = [
        u for u in df_panel["utfall"].unique() if u.startswith("indikator_")
    ]
    n = len(avvik_utfall)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, utfall in zip(axes, avvik_utfall):
        row = df_panel[df_panel["utfall"] == utfall].iloc[0]
        beta_p = row["beta"]
        se_p = row["se"]
        ci_lo = beta_p - 1.96 * se_p
        ci_hi = beta_p + 1.96 * se_p
        pval = row["p"]
        label_p = _INDIKATOR_LABELS.get(utfall, utfall)

        # Bivariate β-fordeling (per næring)
        biv = df_bivariat[df_bivariat["utfall"] == utfall]["beta"].values

        # Boks for bivariate (violinplot / boxplot)
        parts = ax.violinplot(biv, positions=[0], widths=0.6, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(NAV_GRÅ)
            pc.set_alpha(0.4)
        parts["cmedians"].set_color(NAV_GRÅ)
        parts["cmins"].set_color(NAV_GRÅ)
        parts["cmaxes"].set_color(NAV_GRÅ)
        parts["cbars"].set_color(NAV_GRÅ)

        # Panel-β med KI
        color_p = NAV_RØD if pval < 0.05 else NAV_BLÅ
        ax.errorbar(
            [1],
            [beta_p],
            yerr=[[beta_p - ci_lo], [ci_hi - beta_p]],
            fmt="o",
            color=color_p,
            ms=8,
            capsize=5,
            lw=2,
            label=f"Panel β={beta_p:.1f}\n(p={pval:.3f})",
            zorder=5,
        )

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["Bivariate\n(per næring)", "Panel\n(næring FE)"], fontsize=8
        )
        ax.set_title(label_p, fontsize=9)
        ax.set_ylabel("β (OLS)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Panel-β (næring FE, HC1) vs fordeling av bivariate β per næring\n"
        "Rød panel-β = signifikant (p<0.05)",
        fontsize=10,
    )
    fig.tight_layout()
    path = _FIG_DIR / f"ssb_panel_koeffisienter{suffix}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _run_analyse(
    df_ssb: pd.DataFrame,
    df_bedrift: pd.DataFrame,
    df_ind: pd.DataFrame,
    drop_years: list[int] | None,
    suffix: str,
    label: str,
) -> None:
    """Kjør alle tre analysedeler for ett sett av parametere."""
    # ── Del A ─────────────────────────────────────────────────────────────────
    print(f"\n=== Del A: Sammenligning SSB vs bedrift ({label}) ===")
    df_sammenlikning = _lag_sammenlikning(df_ssb, df_bedrift, drop_years)

    x = df_sammenlikning["stramhetsindikator_ssb"].values
    y = df_sammenlikning["stramhetsindikator_bedrift"].values
    mask = np.isfinite(x) & np.isfinite(y)
    r, p = stats.pearsonr(x[mask], y[mask])
    rs, ps = stats.spearmanr(x[mask], y[mask])
    print(
        f"  {len(df_sammenlikning)} rader, {df_sammenlikning['felles'].nunique()} næringer, "
        f"år {df_sammenlikning['aar'].min()}–{df_sammenlikning['aar'].max()}"
    )
    print(
        f"  Pearson r={r:.3f} (p={p:.4f}), Spearman ρ={rs:.3f} (p={ps:.4f}), n={mask.sum()}"
    )

    fig_sammenligning_scatter(df_sammenlikning, suffix)
    fig_naering_tidsserie(df_sammenlikning, suffix)

    dest = _TBL_DIR / f"ssb_bedrift_sammenlikning{suffix}.csv"
    df_sammenlikning[
        [
            "aar",
            "felles",
            "stramhetsindikator_ssb",
            "stramhetsindikator_bedrift",
            "mangel_antall",
        ]
    ].sort_values(["felles", "aar"]).to_csv(dest, index=False)
    print(f"  -> {dest}")

    # ── Del B ─────────────────────────────────────────────────────────────────
    print(f"\n=== Del B: Bivariate regresjoner ({label}) ===")
    df_reg_data = _lag_regresjonsdata(df_ssb, df_ind, drop_years)
    ssb_cols = [c for c in df_reg_data.columns if c.startswith("ssb_")]
    print(
        f"  {len(df_reg_data)} kvartaler, {len(ssb_cols)} næringer, "
        f"{df_reg_data['kvartal'].min()} – {df_reg_data['kvartal'].max()}"
    )

    df_reg = _kjor_regresjoner(df_reg_data, ssb_cols)
    fig_korrelasjon_heatmap(df_reg, suffix)
    fig_beta_naering(df_reg, suffix)

    dest = _TBL_DIR / f"ssb_naering_regresjon{suffix}.csv"
    df_reg.to_csv(dest, index=False)
    print(f"  -> {dest}")

    sig = df_reg[(df_reg["utfall"].str.startswith("indikator_")) & (df_reg["p"] < 0.05)]
    print(f"  Signifikante (p<0.05) næring×avvik-par: {len(sig)}")

    # ── Del C ─────────────────────────────────────────────────────────────────
    print(f"\n=== Del C: Panelregresjon med næring FE ({label}) ===")
    df_panel_data = _lag_panel_data(df_ssb, df_ind, drop_years)
    df_panel = _kjor_panel_regresjon(df_panel_data)

    df_biv_avvik = df_reg[df_reg["utfall"].str.startswith("indikator_")].copy()
    fig_panel_koeffisienter(df_panel, df_biv_avvik, suffix)

    dest = _TBL_DIR / f"ssb_panel_regresjon{suffix}.csv"
    df_panel.to_csv(dest, index=False)
    print(f"  -> {dest}")

    avvik_panel = df_panel[df_panel["utfall"].str.startswith("indikator_")]
    print(
        avvik_panel[
            [
                "utfall_label",
                "beta",
                "se",
                "t",
                "p",
                "r2_within",
                "r2_within_adj",
                "r2_full_adj",
                "n",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    """Kjør analysen for alle år og uten 2021 (sensitivitet)."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    _TBL_DIR.mkdir(parents=True, exist_ok=True)

    df_ssb = _last_ssb_naering()
    df_bedrift = _last_bedrift_naering()
    df_ind = _last_indikator_kvartal()

    _run_analyse(
        df_ssb, df_bedrift, df_ind, drop_years=None, suffix="", label="Alle år"
    )
    _run_analyse(
        df_ssb,
        df_bedrift,
        df_ind,
        drop_years=[2021],
        suffix="_excl2021",
        label="Uten 2021",
    )

    print("\nFerdig!")


if __name__ == "__main__":
    main()
