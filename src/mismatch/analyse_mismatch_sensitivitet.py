"""Sensitivitetsanalyse: leave-one-out og jackknife for mismatch-indikatorer.

Undersøker robustheten av korrelasjon og regresjon ved å ekskludere
ett undersøkelsesår om gangen (spesielt 2021 som koronarelatert uteligger).

Produserer:
  tabeller/loo_korrelasjon.csv     — korrelasjoner for hvert leave-one-out-utvalg
  tabeller/loo_regresjon.csv       — regresjoner for hvert leave-one-out-utvalg
  tabeller/jackknife_korrelasjon.csv — jackknife-estimater med SE og KI
  figurer/loo_korrelasjon.png      — visuelt sammendrag
  figurer/loo_regresjon.png        — visuelt sammendrag

Kjøres:
  uv run python -m src.mismatch.analyse_mismatch_sensitivitet
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.common.stats import adjust_pvalues, newey_west_se

# ── Stier ──────────────────────────────────────────────────────────────────────
_PROCESSED = Path("data/processed")
_FIG_DIR = Path("quarto/mismatch/figurer")
_TBL_DIR = Path("quarto/mismatch/tabeller")

# ── NAV-farger ─────────────────────────────────────────────────────────────────
NAV_BLÅ = "#0067C5"
NAV_RØD = "#C30000"
NAV_GRØNN = "#06893A"
NAV_ORANSJE = "#FF9100"
NAV_GRÅ = "#59514B"
YEAR_COLORS = {
    2021: NAV_RØD,
    2022: NAV_BLÅ,
    2023: NAV_GRØNN,
    2024: NAV_ORANSJE,
    2025: NAV_GRÅ,
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

_UTFALL = ["jobb3", "jobb12", "atid3", "atid12"]
_UTFALL_LABELS = {
    "jobb3": "Jobb 3 mnd",
    "jobb12": "Jobb 12 mnd",
    "atid3": "Arbeidstid 3 mnd",
    "atid12": "Arbeidstid 12 mnd",
}

# Fokuserer på Jackman–Roper for sensitivitetsanalysen
_MISMATCH_VAR = "mismatch_indeks"
_MISMATCH_LABEL = "Jackman–Roper mismatch-indeks"


# ═══════════════════════════════════════════════════════════════════════════════
# Hjelpefunksjoner
# ═══════════════════════════════════════════════════════════════════════════════


def _beregn_aarssnitt(df_nasjonal: pd.DataFrame) -> pd.DataFrame:
    """Beregn årsgjennomsnitt av indikatorer per undersøkelsesår."""
    df = df_nasjonal.copy()
    df["beholdningsmaaned"] = pd.to_datetime(df["beholdningsmaaned"], utc=True)
    return (
        df.groupby(["undersokelsesaar", "utfall"])[["faktisk", "indikator"]]
        .mean()
        .reset_index()
    )


def _fdr_per_gruppe(df: pd.DataFrame, p_kol: str) -> pd.Series:
    """FDR-juster ``p_kol`` innen hvert ``utvalg`` (leave-one-out-familie).

    Hvert leave-one-out-utvalg utgjør sin egen hypotesefamilie (utfall ×
    indikatortype), så korreksjonen gjøres gruppevis og rundes til 4 desimaler.
    """
    return (
        df.groupby("utvalg")[p_kol]
        .transform(lambda s: adjust_pvalues(s.to_numpy()))
        .round(4)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# A. Leave-one-out korrelasjon
# ═══════════════════════════════════════════════════════════════════════════════


def _loo_korrelasjon(df_aar: pd.DataFrame, aarssnitt: pd.DataFrame) -> pd.DataFrame:
    """Beregn korrelasjoner med ett år ekskludert om gangen."""
    alle_aar = sorted(df_aar["aar"].unique())
    # "Alle" = fullt utvalg, deretter ekskluder ett år
    utvalg = [("Alle", alle_aar)] + [
        (f"Uten {y}", [a for a in alle_aar if a != y]) for y in alle_aar
    ]

    records = []
    for utvalg_label, inkluderte_aar in utvalg:
        ekskludert = int(utvalg_label.split()[-1]) if utvalg_label != "Alle" else None
        df_sub = df_aar[df_aar["aar"].isin(inkluderte_aar)]

        for utfall in _UTFALL:
            for ind_type, ind_label in [
                ("faktisk", "Faktisk"),
                ("indikator", "Indikator (avvik)"),
            ]:
                sub = aarssnitt[
                    (aarssnitt["utfall"] == utfall)
                    & (aarssnitt["undersokelsesaar"].isin(inkluderte_aar))
                ].merge(
                    df_sub[["aar", _MISMATCH_VAR]],
                    left_on="undersokelsesaar",
                    right_on="aar",
                )
                if len(sub) < 3:
                    continue

                r_p, p_p = stats.pearsonr(sub[_MISMATCH_VAR], sub[ind_type])
                r_s, p_s = stats.spearmanr(sub[_MISMATCH_VAR], sub[ind_type])
                records.append(
                    {
                        "utvalg": utvalg_label,
                        "ekskludert_aar": ekskludert,
                        "er_2021_ekskludert": ekskludert == 2021,
                        "utfall": utfall,
                        "utfall_label": _UTFALL_LABELS[utfall],
                        "indikator_type": ind_type,
                        "indikator_type_label": ind_label,
                        "r_pearson": round(r_p, 3),
                        "p_pearson": round(p_p, 4),
                        "r_spearman": round(r_s, 3),
                        "p_spearman": round(p_s, 4),
                        "n": len(sub),
                        "liten_utvalg": len(sub) < 6,
                    }
                )

    resultat = pd.DataFrame(records)
    if not resultat.empty:
        # Korriger for multippel testing innen hvert utvalg (familien av
        # utfall × indikatortype for det aktuelle leave-one-out-utvalget).
        resultat["p_pearson_fdr"] = _fdr_per_gruppe(resultat, "p_pearson")
        resultat["p_spearman_fdr"] = _fdr_per_gruppe(resultat, "p_spearman")
    return resultat


# ═══════════════════════════════════════════════════════════════════════════════
# B. Leave-one-out regresjon (månedlig med sesongkontroll)
# ═══════════════════════════════════════════════════════════════════════════════


def _loo_regresjon(df_nasjonal: pd.DataFrame) -> pd.DataFrame:
    """OLS-regresjon med ett undersøkelsesår ekskludert om gangen."""
    df = df_nasjonal.copy()
    df["beholdningsmaaned"] = pd.to_datetime(df["beholdningsmaaned"], utc=True)
    df["maaned"] = df["beholdningsmaaned"].dt.month

    alle_aar = sorted(df["undersokelsesaar"].unique())
    utvalg = [("Alle", alle_aar)] + [
        (f"Uten {y}", [a for a in alle_aar if a != y]) for y in alle_aar
    ]

    records = []
    for utvalg_label, inkluderte_aar in utvalg:
        ekskludert = int(utvalg_label.split()[-1]) if utvalg_label != "Alle" else None
        sub_all = df[df["undersokelsesaar"].isin(inkluderte_aar)]

        for utfall in _UTFALL:
            for ind_type, ind_label in [
                ("faktisk", "Faktisk"),
                ("indikator", "Indikator (avvik)"),
            ]:
                sub = sub_all[sub_all["utfall"] == utfall].dropna(
                    subset=[_MISMATCH_VAR, ind_type]
                )
                if len(sub) < 10:
                    continue

                y = sub[ind_type].values
                x_m = sub[_MISMATCH_VAR].values
                month_dummies = pd.get_dummies(
                    sub["maaned"], prefix="m", drop_first=True
                ).values
                X = np.column_stack([np.ones(len(y)), x_m, month_dummies])

                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X @ beta
                n, k = X.shape
                nw_se = newey_west_se(resid, X)

                ss_res = np.sum(resid**2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
                t_stat = beta[1] / nw_se[1] if nw_se[1] > 0 else np.nan
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - k))

                records.append(
                    {
                        "utvalg": utvalg_label,
                        "ekskludert_aar": ekskludert,
                        "er_2021_ekskludert": ekskludert == 2021,
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
        resultat["p_verdi_fdr"] = _fdr_per_gruppe(resultat, "p_verdi")
    return resultat


# ═══════════════════════════════════════════════════════════════════════════════
# C. Jackknife-estimat av korrelasjon
# ═══════════════════════════════════════════════════════════════════════════════


def _jackknife_korrelasjon(
    df_aar: pd.DataFrame, aarssnitt: pd.DataFrame
) -> pd.DataFrame:
    """Jackknife-estimater med standardfeil og konfidensintervall."""
    alle_aar = sorted(df_aar["aar"].unique())
    n_years = len(alle_aar)

    records = []
    for utfall in _UTFALL:
        for ind_type, ind_label in [
            ("faktisk", "Faktisk"),
            ("indikator", "Indikator (avvik)"),
        ]:
            # Full-sample estimate
            sub_full = aarssnitt[aarssnitt["utfall"] == utfall].merge(
                df_aar[["aar", _MISMATCH_VAR]],
                left_on="undersokelsesaar",
                right_on="aar",
            )
            if len(sub_full) < 4:
                continue

            r_full, _ = stats.pearsonr(sub_full[_MISMATCH_VAR], sub_full[ind_type])

            # Leave-one-out estimates
            r_loo = []
            for drop_year in alle_aar:
                sub = sub_full[sub_full["aar"] != drop_year]
                if len(sub) < 3:
                    continue
                r_i, _ = stats.pearsonr(sub[_MISMATCH_VAR], sub[ind_type])
                r_loo.append(r_i)

            r_loo = np.array(r_loo)
            n = len(r_loo)

            # Jackknife pseudovalues
            pseudo = n_years * r_full - (n_years - 1) * r_loo

            # Jackknife estimate and SE
            r_jack = pseudo.mean()
            se_jack = np.sqrt(np.sum((pseudo - r_jack) ** 2) / (n * (n - 1)))

            # 95% CI
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci_lo = r_jack - t_crit * se_jack
            ci_hi = r_jack + t_crit * se_jack

            records.append(
                {
                    "utfall": utfall,
                    "utfall_label": _UTFALL_LABELS[utfall],
                    "indikator_type": ind_type,
                    "indikator_type_label": ind_label,
                    "r_full": round(r_full, 3),
                    "r_jackknife": round(r_jack, 3),
                    "se_jackknife": round(se_jack, 3),
                    "ki_nedre": round(ci_lo, 3),
                    "ki_ovre": round(ci_hi, 3),
                    "r_uten_2021": round(
                        r_loo[alle_aar.index(2021)] if 2021 in alle_aar else np.nan, 3
                    ),
                    "n_aar": n_years,
                }
            )

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# D. Figurer
# ═══════════════════════════════════════════════════════════════════════════════


def _figur_loo_korrelasjon(df_loo: pd.DataFrame) -> None:
    """Visuelt sammendrag av leave-one-out Pearson-korrelasjoner."""
    # Fokuser på faktisk-verdier for klarhet
    sub = df_loo[df_loo["indikator_type"] == "faktisk"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, utfall in zip(axes.flat, _UTFALL):
        data = sub[sub["utfall"] == utfall].copy()
        full = data[data["utvalg"] == "Alle"]
        loo = data[data["utvalg"] != "Alle"]

        # Fullt utvalg som horisontal linje
        if not full.empty:
            r_all = full["r_pearson"].values[0]
            ax.axhline(
                r_all,
                color=NAV_GRÅ,
                linestyle="--",
                alpha=0.5,
                label=f"Alle (r={r_all:.3f})",
            )

        # Leave-one-out punkter
        for _, row in loo.iterrows():
            yr = int(row["ekskludert_aar"])
            color = YEAR_COLORS.get(yr, NAV_GRÅ)
            marker = "D" if yr == 2021 else "o"
            size = 120 if yr == 2021 else 80
            ax.scatter(
                str(yr),
                row["r_pearson"],
                color=color,
                s=size,
                marker=marker,
                zorder=5,
                label=f"Uten {yr} (r={row['r_pearson']:.3f})",
            )

        ax.set_ylabel("Pearson r")
        ax.set_title(f"{_UTFALL_LABELS[utfall]} (faktisk)", fontsize=10)
        ax.legend(fontsize=7, loc="lower left")
        ax.set_xlabel("Ekskludert år")

    fig.suptitle(
        "Leave-one-out sensitivitet: korrelasjon (mismatch-indeks vs. faktisk)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "loo_korrelasjon.png", dpi=FIG_DPI)
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'loo_korrelasjon.png'}")


def _figur_loo_regresjon(df_loo: pd.DataFrame) -> None:
    """Visuelt sammendrag av leave-one-out regresjonskoeffisienter."""
    sub = df_loo[df_loo["indikator_type"] == "indikator"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, utfall in zip(axes.flat, _UTFALL):
        data = sub[sub["utfall"] == utfall].copy()
        full = data[data["utvalg"] == "Alle"]
        loo = data[data["utvalg"] != "Alle"]

        if not full.empty:
            b_all = full["beta_mismatch"].values[0]
            ax.axhline(
                b_all,
                color=NAV_GRÅ,
                linestyle="--",
                alpha=0.5,
                label=f"Alle (β={b_all:.2f})",
            )

        for _, row in loo.iterrows():
            yr = int(row["ekskludert_aar"])
            color = YEAR_COLORS.get(yr, NAV_GRÅ)
            marker = "D" if yr == 2021 else "o"

            # Feilstolpe med Newey–West SE
            ax.errorbar(
                str(yr),
                row["beta_mismatch"],
                yerr=1.96 * row["se_newey_west"],
                fmt=marker,
                color=color,
                markersize=8 if yr == 2021 else 6,
                capsize=3,
                elinewidth=1.5,
                zorder=5,
                label=f"Uten {yr} (β={row['beta_mismatch']:.2f})",
            )

        ax.set_ylabel("β (mismatch)")
        ax.set_title(f"{_UTFALL_LABELS[utfall]} (indikator)", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("Ekskludert år")
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle(
        "Leave-one-out sensitivitet: regresjonskoeffisient β (mismatch → indikator)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "loo_regresjon.png", dpi=FIG_DPI)
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'loo_regresjon.png'}")


def _figur_jackknife(df_jack: pd.DataFrame) -> None:
    """Jackknife-korrelasjoner med konfidensintervall."""
    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = 0
    y_labels = []
    for _, row in df_jack.iterrows():
        label = f"{row['utfall_label']} ({row['indikator_type_label'].lower()})"
        y_labels.append(label)

        # Full sample
        ax.scatter(
            row["r_full"], y_pos, color=NAV_GRÅ, s=60, marker="s", zorder=5, alpha=0.5
        )

        # Jackknife med KI
        ax.errorbar(
            row["r_jackknife"],
            y_pos,
            xerr=[
                [row["r_jackknife"] - row["ki_nedre"]],
                [row["ki_ovre"] - row["r_jackknife"]],
            ],
            fmt="o",
            color=NAV_BLÅ,
            markersize=8,
            capsize=4,
            elinewidth=2,
            zorder=6,
        )

        # Markér r uten 2021
        if pd.notna(row["r_uten_2021"]):
            ax.scatter(
                row["r_uten_2021"],
                y_pos,
                color=NAV_RØD,
                s=80,
                marker="D",
                zorder=7,
            )

        y_pos += 1

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Pearson r (korrelasjon)")
    ax.axvline(0, color="black", linewidth=0.5)

    # Legende
    ax.scatter([], [], color=NAV_GRÅ, s=60, marker="s", alpha=0.5, label="Fullt utvalg")
    ax.errorbar(
        [], [], fmt="o", color=NAV_BLÅ, markersize=8, label="Jackknife ± 95 % KI"
    )
    ax.scatter([], [], color=NAV_RØD, s=80, marker="D", label="Uten 2021")
    ax.legend(loc="lower right", fontsize=9)

    ax.set_title(
        "Jackknife-korrelasjon: mismatch-indeks vs. indikatorer",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "jackknife_korrelasjon.png", dpi=FIG_DPI)
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'jackknife_korrelasjon.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Kjør sensitivitetsanalysen."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    _TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("Leser data...")
    df_aar = pd.read_csv(_PROCESSED / "mismatch_aar.csv")
    df_nasjonal = pd.read_csv(_PROCESSED / "mismatch_nasjonal.csv")
    aarssnitt = _beregn_aarssnitt(df_nasjonal)

    # ── A. LOO-korrelasjon ─────────────────────────────────────────────────
    print("\nA. Leave-one-out korrelasjon")
    df_loo_korr = _loo_korrelasjon(df_aar, aarssnitt)
    df_loo_korr.to_csv(_TBL_DIR / "loo_korrelasjon.csv", index=False)
    print(f"  -> {_TBL_DIR / 'loo_korrelasjon.csv'}")

    # Vis resultat for faktisk, Pearson
    print("\n  Pearson r (faktisk) per utvalg:")
    sub = df_loo_korr[(df_loo_korr["indikator_type"] == "faktisk")]
    for utvalg in sub["utvalg"].unique():
        print(f"  {utvalg}:")
        for _, row in sub[sub["utvalg"] == utvalg].iterrows():
            print(
                f"    {row['utfall_label']:<20} r={row['r_pearson']:>6.3f}  p={row['p_pearson']:.4f}"
            )

    _figur_loo_korrelasjon(df_loo_korr)

    # ── B. LOO-regresjon ───────────────────────────────────────────────────
    print("\nB. Leave-one-out regresjon")
    df_loo_reg = _loo_regresjon(df_nasjonal)
    df_loo_reg.to_csv(_TBL_DIR / "loo_regresjon.csv", index=False)
    print(f"  -> {_TBL_DIR / 'loo_regresjon.csv'}")

    _figur_loo_regresjon(df_loo_reg)

    # ── C. Jackknife ───────────────────────────────────────────────────────
    print("\nC. Jackknife-estimater")
    df_jack = _jackknife_korrelasjon(df_aar, aarssnitt)
    df_jack.to_csv(_TBL_DIR / "jackknife_korrelasjon.csv", index=False)
    print(f"  -> {_TBL_DIR / 'jackknife_korrelasjon.csv'}")
    print(df_jack.to_string(index=False))

    _figur_jackknife(df_jack)

    print("\nFerdig!")


if __name__ == "__main__":
    main()
