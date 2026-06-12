"""Analyserer sammenhengen mellom kompetansemismatch og nasjonale Nav-indikatorer.

Fire analytiske tilnærminger:
  A. Deskriptivt: yrkesvis stramhetsratio over tid
  B. Korrelasjon: mismatch-mål vs. årsgjennomsnitt av indikatorer
  C. Visuell tidsserie: månedlige indikatorer med mismatch som step-funksjon
  D. OLS-regresjon: mismatch som forklaringsvariabel med sesongkontroll

Produserer figurer og tabeller til quarto/mismatch/.

Kjøres:
  uv run python -m src.mismatch.analyse_mismatch
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.common.stats import adjust_pvalues, newey_west_se
from src.common.validation import require_columns, require_nonempty

# ── Stier ──────────────────────────────────────────────────────────────────────
_PROCESSED = Path("data/processed")
_FIG_DIR = Path("quarto/mismatch/figurer")
_TBL_DIR = Path("quarto/mismatch/tabeller")

# ── NAV-farger og stil ─────────────────────────────────────────────────────────
NAV_BLÅ = "#0067C5"
NAV_RØD = "#C30000"
NAV_GRØNN = "#06893A"
NAV_ORANSJE = "#FF9100"
NAV_GRÅ = "#59514B"
YEAR_COLORS = {
    2021: NAV_BLÅ,
    2022: NAV_RØD,
    2023: NAV_GRØNN,
    2024: NAV_ORANSJE,
    2025: NAV_GRÅ,
    2026: "#A13A2E",
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

# Utfallsvariabler vi analyserer
_UTFALL = ["jobb3", "jobb12", "atid3", "atid12"]
_UTFALL_LABELS = {
    "jobb3": "Jobb 3 mnd",
    "jobb12": "Jobb 12 mnd",
    "atid3": "Arbeidstid 3 mnd",
    "atid12": "Arbeidstid 12 mnd",
}
_MISMATCH_VARS = {
    "mismatch_indeks": "Jackman–Roper mismatch-indeks",
    "mismatch_indeks_cd_30": "Cobb–Douglas mismatch (α=0.3)",
    "mismatch_indeks_cd_50": "Cobb–Douglas mismatch (α=0.5)",
    "mismatch_indeks_cd_70": "Cobb–Douglas mismatch (α=0.7)",
    "stramhetsratio": "Aggregert stramhetsratio (mangel/ledige)",
    "mismatch_ledighet_andel": "Andel mismatch-ledighet",
}


# ═══════════════════════════════════════════════════════════════════════════════
# A. Deskriptiv: Yrkesvis stramhet over tid
# ═══════════════════════════════════════════════════════════════════════════════


def _figur_stramhet_per_yrke(df_yrke: pd.DataFrame) -> None:
    """Heatmap av stramhetsratio per yrke per år."""
    pivot = df_yrke.pivot_table(
        index="yrkesgruppe", columns="aar", values="stramhetsratio_yrke"
    )
    # Sorter yrker etter gjennomsnittlig stramhet (høyest øverst)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=3.5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Skriv verdier i cellene
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                color = "white" if val > 2.0 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    ax.set_title("Stramhetsratio per yrkesgruppe (mangel / ledige+tiltak)")
    fig.colorbar(im, ax=ax, label="Stramhetsratio (τ)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "stramhet_per_yrke.png", dpi=FIG_DPI)
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'stramhet_per_yrke.png'}")


def _figur_mismatch_tidsserie(df_aar: pd.DataFrame) -> None:
    """Tidsserie av mismatch-mål over 2021–2025."""
    n_vars = len(_MISMATCH_VARS)
    ncols = 3
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), sharey=False)
    axes_flat = axes.flat if n_vars > 1 else [axes]

    for ax, (col, label) in zip(axes_flat, _MISMATCH_VARS.items()):
        ax.plot(
            df_aar["aar"], df_aar[col], "o-", color=NAV_BLÅ, linewidth=2, markersize=8
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Undersøkelsesår")
        ax.set_xticks(df_aar["aar"])

    # Skjul ubrukte subplots
    for ax in list(axes_flat)[n_vars:]:
        ax.set_visible(False)

    fig.suptitle("Mismatch i arbeidsmarkedet 2021–2025", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "mismatch_indeks.png", dpi=FIG_DPI)
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'mismatch_indeks.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# B. Korrelasjon: Mismatch vs. årsgjennomsnitt indikatorer
# ═══════════════════════════════════════════════════════════════════════════════


def _korrelasjon_mismatch_indikatorer(
    df_aar: pd.DataFrame, df_nasjonal: pd.DataFrame
) -> pd.DataFrame:
    """Korreler mismatch-mål med årsgjennomsnitt av alle indikatorer."""
    # Beregn årssnitt per utfall
    df_nasjonal["beholdningsmaaned"] = pd.to_datetime(
        df_nasjonal["beholdningsmaaned"], utc=True
    )
    aarssnitt = (
        df_nasjonal.groupby(["undersokelsesaar", "utfall"])[["faktisk", "indikator"]]
        .mean()
        .reset_index()
    )

    records = []
    for mvar, mlabel in _MISMATCH_VARS.items():
        for utfall in _UTFALL:
            for ind_type, ind_label in [
                ("faktisk", "Faktisk"),
                ("indikator", "Indikator (avvik)"),
            ]:
                sub = aarssnitt[aarssnitt["utfall"] == utfall].merge(
                    df_aar[["aar", mvar]], left_on="undersokelsesaar", right_on="aar"
                )
                if len(sub) < 4:
                    continue
                r_pearson, p_pearson = stats.pearsonr(sub[mvar], sub[ind_type])
                r_spearman, p_spearman = stats.spearmanr(sub[mvar], sub[ind_type])
                n = len(sub)
                # Fisher-z 95 %-konfidensintervall for Pearson r (krever n > 3).
                if n > 3:
                    z = np.arctanh(r_pearson)
                    se_z = 1.0 / np.sqrt(n - 3)
                    ci_lav = round(float(np.tanh(z - 1.96 * se_z)), 3)
                    ci_hoy = round(float(np.tanh(z + 1.96 * se_z)), 3)
                else:
                    ci_lav, ci_hoy = np.nan, np.nan
                records.append(
                    {
                        "mismatch_var": mvar,
                        "mismatch_label": mlabel,
                        "utfall": utfall,
                        "utfall_label": _UTFALL_LABELS[utfall],
                        "indikator_type": ind_type,
                        "indikator_type_label": ind_label,
                        "r_pearson": round(r_pearson, 3),
                        "p_pearson": round(p_pearson, 4),
                        "r_pearson_ci_lav": ci_lav,
                        "r_pearson_ci_hoy": ci_hoy,
                        "r_spearman": round(r_spearman, 3),
                        "p_spearman": round(p_spearman, 4),
                        "n": n,
                        "liten_utvalg": n < 6,
                    }
                )

    resultat = pd.DataFrame(records)
    if not resultat.empty:
        # Korriger for multippel testing på tvers av korrelasjonsfamilien.
        resultat["p_pearson_fdr"] = np.round(
            adjust_pvalues(resultat["p_pearson"].to_numpy()), 4
        )
        resultat["p_spearman_fdr"] = np.round(
            adjust_pvalues(resultat["p_spearman"].to_numpy()), 4
        )
    return resultat


# ═══════════════════════════════════════════════════════════════════════════════
# C. Visuell tidsserie: Månedlige indikatorer + mismatch
# ═══════════════════════════════════════════════════════════════════════════════


def _figur_tidsserie_med_mismatch(
    df_nasjonal: pd.DataFrame,
    df_aar: pd.DataFrame,
    ind_type: str,
    ylabel: str,
    suffix: str,
) -> None:
    """Plott månedlige indikatorverdier med mismatch-indeks som step-bakgrunn."""
    df = df_nasjonal.copy()
    df["beholdningsmaaned"] = pd.to_datetime(df["beholdningsmaaned"], utc=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    for ax, utfall in zip(axes.flat, _UTFALL):
        sub = df[df["utfall"] == utfall].sort_values("beholdningsmaaned")
        if sub.empty:
            continue

        # Primær akse: indikatorverdi
        ax.plot(sub["beholdningsmaaned"], sub[ind_type], color=NAV_BLÅ, linewidth=1.2)
        ax.set_ylabel(f"{_UTFALL_LABELS[utfall]} ({ylabel})", color=NAV_BLÅ, fontsize=9)
        ax.tick_params(axis="y", labelcolor=NAV_BLÅ)

        # Sekundær akse: mismatch-indeks (step)
        ax2 = ax.twinx()
        ax2.step(
            df_aar["aar"].apply(lambda y: pd.Timestamp(y, _ref_month(y), 1, tz="UTC")),
            df_aar["mismatch_indeks"],
            where="post",
            color=NAV_RØD,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )
        ax2.set_ylabel("Mismatch-indeks", color=NAV_RØD, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=NAV_RØD)

        ax.set_title(_UTFALL_LABELS[utfall], fontsize=10)

    fig.suptitle(
        f"Månedlig {ylabel.lower()} og mismatch-indeks (stiplet)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(_FIG_DIR / f"tidsserie_{suffix}.png", dpi=FIG_DPI)
    plt.close(fig)
    print(f"  -> {_FIG_DIR / f'tidsserie_{suffix}.png'}")


def _ref_month(year: int) -> int:
    """Referansemåned for et undersøkelsesår."""
    return {2021: 2, 2022: 4, 2023: 4, 2024: 3, 2025: 3}.get(year, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# D. OLS-regresjon: Mismatch → indikator med sesongkontroll
# ═══════════════════════════════════════════════════════════════════════════════


def _regresjon_mismatch(df_nasjonal: pd.DataFrame) -> pd.DataFrame:
    """OLS-regresjon: indikator ~ mismatch_var + måned-dummyer."""
    df = df_nasjonal.copy()
    df["beholdningsmaaned"] = pd.to_datetime(df["beholdningsmaaned"], utc=True)
    df["maaned"] = df["beholdningsmaaned"].dt.month

    records = []
    for mvar, mlabel in _MISMATCH_VARS.items():
        for utfall in _UTFALL:
            for ind_type, ind_label in [
                ("faktisk", "Faktisk"),
                ("indikator", "Indikator (avvik)"),
            ]:
                sub = df[df["utfall"] == utfall].dropna(subset=[mvar, ind_type])
                if len(sub) < 15:
                    continue

                y = sub[ind_type].values
                x_mismatch = sub[mvar].values

                # Måned-dummyer (referanse = januar)
                month_dummies = pd.get_dummies(
                    sub["maaned"], prefix="m", drop_first=True
                ).values

                # Designmatrise: [konstant, mismatch, måned-dummyer]
                X = np.column_stack([np.ones(len(y)), x_mismatch, month_dummies])

                # OLS
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                y_hat = X @ beta
                resid = y - y_hat
                n, k = X.shape

                # Newey–West SE
                nw_se = newey_west_se(resid, X)

                # R²
                ss_res = np.sum(resid**2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

                # t-stat og p for mismatch-koeffisienten (index 1)
                t_stat = beta[1] / nw_se[1] if nw_se[1] > 0 else np.nan
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - k))

                records.append(
                    {
                        "mismatch_var": mvar,
                        "mismatch_label": mlabel,
                        "utfall": utfall,
                        "utfall_label": _UTFALL_LABELS[utfall],
                        "indikator_type": ind_type,
                        "indikator_type_label": ind_label,
                        "beta_mismatch": round(beta[1], 4),
                        "se_newey_west": round(nw_se[1], 4),
                        "t_stat": round(t_stat, 3),
                        "p_verdi": round(p_val, 4),
                        "r2_adj": round(r2_adj, 4),
                        "n": n,
                    }
                )

    resultat = pd.DataFrame(records)
    if not resultat.empty:
        # Korriger for multippel testing på tvers av familien av regresjoner
        # (mismatch-variabel × utfall × indikatortype).
        resultat["p_verdi_fdr"] = np.round(
            adjust_pvalues(resultat["p_verdi"].to_numpy()), 4
        )
    return resultat


# ═══════════════════════════════════════════════════════════════════════════════
# Scatterplott: Mismatch vs. årsgjennomsnitt
# ═══════════════════════════════════════════════════════════════════════════════


def _figur_scatter_mismatch(df_aar: pd.DataFrame, df_nasjonal: pd.DataFrame) -> None:
    """Scatter av mismatch_indeks vs. årssnitt for hver indikator (faktisk + avvik)."""
    df_nasjonal["beholdningsmaaned"] = pd.to_datetime(
        df_nasjonal["beholdningsmaaned"], utc=True
    )
    aarssnitt = (
        df_nasjonal.groupby(["undersokelsesaar", "utfall"])[["faktisk", "indikator"]]
        .mean()
        .reset_index()
    )

    for ind_type, ind_label, suffix in [
        ("faktisk", "Faktisk verdi", "faktisk"),
        ("indikator", "Indikator (avvik fra forventet)", "indikator"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for ax, utfall in zip(axes.flat, _UTFALL):
            sub = aarssnitt[aarssnitt["utfall"] == utfall].merge(
                df_aar[["aar", "mismatch_indeks"]],
                left_on="undersokelsesaar",
                right_on="aar",
            )
            for _, row in sub.iterrows():
                ax.scatter(
                    row["mismatch_indeks"],
                    row[ind_type],
                    color=YEAR_COLORS.get(int(row["aar"]), NAV_GRÅ),
                    s=100,
                    zorder=5,
                )
                ax.annotate(
                    str(int(row["aar"])),
                    (row["mismatch_indeks"], row[ind_type]),
                    textcoords="offset points",
                    xytext=(8, 4),
                    fontsize=9,
                )

            ax.set_xlabel("Mismatch-indeks")
            ax.set_ylabel(f"{_UTFALL_LABELS[utfall]}")
            ax.set_title(f"{_UTFALL_LABELS[utfall]} ({ind_label.lower()})")

        fig.suptitle(
            f"Mismatch-indeks vs. {ind_label.lower()} (årsgjennomsnitt)",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(_FIG_DIR / f"scatter_mismatch_{suffix}.png", dpi=FIG_DPI)
        plt.close(fig)
        print(f"  -> {_FIG_DIR / f'scatter_mismatch_{suffix}.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Kjør hele mismatch-analysen."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    _TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("Leser data...")
    df_yrke = require_nonempty(
        pd.read_csv(_PROCESSED / "mismatch_yrke.csv"), source="mismatch_yrke.csv"
    )
    df_aar = require_columns(
        pd.read_csv(_PROCESSED / "mismatch_aar.csv"),
        ["aar"],
        source="mismatch_aar.csv",
    )
    df_nasjonal = require_columns(
        pd.read_csv(_PROCESSED / "mismatch_nasjonal.csv"),
        ["beholdningsmaaned", "utfall"],
        source="mismatch_nasjonal.csv",
    )

    # ── A. Deskriptiv ──────────────────────────────────────────────────────
    print("\nA. Deskriptiv: yrkesvis stramhet")
    _figur_stramhet_per_yrke(df_yrke)
    _figur_mismatch_tidsserie(df_aar)

    # Lagre mismatch-mål som tabell
    df_aar.to_csv(_TBL_DIR / "mismatch_maal.csv", index=False)
    print(f"  -> {_TBL_DIR / 'mismatch_maal.csv'}")

    # ── B. Korrelasjon ─────────────────────────────────────────────────────
    print("\nB. Korrelasjon: mismatch vs. årsgjennomsnitt indikatorer")
    df_korr = _korrelasjon_mismatch_indikatorer(df_aar, df_nasjonal)
    df_korr.to_csv(_TBL_DIR / "korrelasjon.csv", index=False)
    print(f"  -> {_TBL_DIR / 'korrelasjon.csv'}")
    print("\n  Utvalgte korrelasjoner (mismatch_indeks, Pearson):")
    sub = df_korr[df_korr["mismatch_var"] == "mismatch_indeks"]
    for _, row in sub.iterrows():
        print(
            f"    {row['utfall_label']:<20} {row['indikator_type_label']:<25} "
            f"r={row['r_pearson']:>6.3f}  p={row['p_pearson']:.4f}  (n={row['n']})"
        )

    # ── C. Visuell tidsserie ───────────────────────────────────────────────
    print("\nC. Visuell tidsserie")
    _figur_tidsserie_med_mismatch(
        df_nasjonal, df_aar, "faktisk", "Faktisk verdi", "faktisk"
    )
    _figur_tidsserie_med_mismatch(
        df_nasjonal, df_aar, "indikator", "Avvik fra forventet", "indikator"
    )
    _figur_scatter_mismatch(df_aar, df_nasjonal)

    # ── D. Regresjon ───────────────────────────────────────────────────────
    print("\nD. OLS-regresjon med sesongkontroll (Newey–West SE)")
    df_reg = _regresjon_mismatch(df_nasjonal)
    df_reg.to_csv(_TBL_DIR / "regresjon.csv", index=False)
    print(f"  -> {_TBL_DIR / 'regresjon.csv'}")
    print("\n  Resultater (mismatch_indeks):")
    sub = df_reg[df_reg["mismatch_var"] == "mismatch_indeks"]
    for _, row in sub.iterrows():
        sig = (
            "***"
            if row["p_verdi"] < 0.001
            else "**"
            if row["p_verdi"] < 0.01
            else "*"
            if row["p_verdi"] < 0.05
            else ""
        )
        print(
            f"    {row['utfall_label']:<20} {row['indikator_type_label']:<25} "
            f"β={row['beta_mismatch']:>8.4f}  SE={row['se_newey_west']:>7.4f}  "
            f"t={row['t_stat']:>6.3f}  p={row['p_verdi']:.4f} {sig}  "
            f"R²adj={row['r2_adj']:.3f}  (n={row['n']})"
        )

    print("\nFerdig!")


if __name__ == "__main__":
    main()
