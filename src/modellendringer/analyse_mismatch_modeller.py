"""Analyserer sammenhengen mellom kompetansemismatch og de utvidede modellvariantene.

Spørsmålet er om indikatoravvikene fra de utvidede modellene viser lavere
korrelasjon med mismatch-indeksene enn referansemodellen — analog til
stramhetstesten, men med mismatch som ekstern validitor.

Tilnærming:
  A. Korrelasjoner: Pearson/Spearman mellom mismatch-indeks og årsgjennomsnitt
     av indikatorer, per modellvariant
  B. OLS-regresjon: mismatch → indikator med sesongkontroll og Newey–West SE,
     per modellvariant

Kjøres:
  uv run python src/modellendringer/analyse_mismatch_modeller.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Stier ──────────────────────────────────────────────────────────────────────
_PROCESSED = Path("data/processed")
_RESULTS = Path("data/results")
_FIG_DIR = Path("quarto/modellendringer/figurer")
_TBL_DIR = Path("quarto/modellendringer/tabeller")

# ── NAV-farger ─────────────────────────────────────────────────────────────────
NAV_BLÅ = "#0067C5"
NAV_RØD = "#C30000"
NAV_GRØNN = "#06893A"
NAV_ORANSJE = "#FF9100"
NAV_GRÅ = "#59514B"

# Konsistent fargekart per modell (samme som resten av modellendringer-analysen)
MODELL_FARGER = {
    "Referanse": NAV_GRÅ,
    "Stillingsrate": NAV_BLÅ,
    "Shiftshare": NAV_GRØNN,
    "Ledighetsrate": NAV_RØD,
    "Ledighetsrate ung": NAV_ORANSJE,
}
MODELL_REKKEFØLGE = [
    "Referanse",
    "Stillingsrate",
    "Shiftshare",
    "Ledighetsrate",
    "Ledighetsrate ung",
]

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

# Referansemåneder for å tilknytte undersøkelsesår (identisk med standardiser_mismatch_data.py)
_REFERANSEMAANED: dict[int, int] = {2021: 2, 2022: 4, 2023: 4, 2024: 3, 2025: 3}

_UTFALL = ["jobb3", "jobb12", "atid3", "atid12"]
_UTFALL_LABELS = {
    "jobb3": "Jobb 3 mnd",
    "jobb12": "Jobb 12 mnd",
    "atid3": "Arbeidstid 3 mnd",
    "atid12": "Arbeidstid 12 mnd",
}
# Fokus på JR-indeksen (primær mismatch-mål i resten av rapporten)
_MISMATCH_VARS = {
    "mismatch_indeks": "Jackman–Roper mismatch-indeks",
    "mismatch_indeks_cd_50": "Cobb–Douglas mismatch (α=0.5)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Datapreprosessering
# ═══════════════════════════════════════════════════════════════════════════════


def _last_og_konverter_nasjonalt(path: Path) -> pd.DataFrame:
    """Les modellendringer-nasjonalt og konverter til langt format."""
    df = pd.read_csv(path)
    df["beholdningsmaaned"] = pd.to_datetime(df["beholdningsmaaned"], utc=True)

    records = []
    for utfall in _UTFALL:
        ind_col = f"indikator_{utfall}"
        fakt_col = f"faktisk_{utfall}"
        forv_col = f"forventet_{utfall}"
        if ind_col not in df.columns:
            continue
        sub = df[
            [
                "beholdningsmaaned",
                "modell",
                "result_id",
                "aar",
                "maaned",
                ind_col,
                fakt_col,
                forv_col,
                "antall_personer",
            ]
        ].copy()
        sub = sub.rename(
            columns={
                ind_col: "indikator",
                fakt_col: "faktisk",
                forv_col: "forventet",
            }
        )
        sub["utfall"] = utfall
        records.append(sub)

    return pd.concat(records, ignore_index=True)


def _tilknytt_undersokelsesaar(df: pd.DataFrame) -> pd.DataFrame:
    """Tilknytt undersøkelsesår på samme måte som standardiser_mismatch_data.py."""
    cutoffs = sorted(_REFERANSEMAANED.items())

    def _finn(row: pd.Series) -> int:
        y, m = int(row["aar"]), int(row["maaned"])
        tilhoerer = cutoffs[0][0]
        for survey_year, ref_month in cutoffs:
            if y > survey_year or (y == survey_year and m >= ref_month):
                tilhoerer = survey_year
        return tilhoerer

    df = df.copy()
    df["undersokelsesaar"] = df.apply(_finn, axis=1)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# A. Korrelasjoner per modell
# ═══════════════════════════════════════════════════════════════════════════════


def _korrelasjon_per_modell(
    df_lang: pd.DataFrame, df_aar: pd.DataFrame
) -> pd.DataFrame:
    """Korreler mismatch-mål med årsgjennomsnitt av indikatorer per modell."""
    aarssnitt = (
        df_lang.groupby(["modell", "undersokelsesaar", "utfall"])[
            ["faktisk", "indikator"]
        ]
        .mean()
        .reset_index()
    )

    records = []
    for modell in MODELL_REKKEFØLGE:
        for mvar, mlabel in _MISMATCH_VARS.items():
            for utfall in _UTFALL:
                for ind_type, ind_label in [
                    ("faktisk", "Faktisk"),
                    ("indikator", "Indikator (avvik)"),
                ]:
                    sub = aarssnitt[
                        (aarssnitt["modell"] == modell)
                        & (aarssnitt["utfall"] == utfall)
                    ].merge(
                        df_aar[["aar", mvar]],
                        left_on="undersokelsesaar",
                        right_on="aar",
                    )
                    sub = sub.dropna(subset=[mvar, ind_type])
                    if len(sub) < 4:
                        continue
                    r_p, p_p = stats.pearsonr(sub[mvar], sub[ind_type])
                    r_s, p_s = stats.spearmanr(sub[mvar], sub[ind_type])
                    records.append(
                        {
                            "modell": modell,
                            "mismatch_var": mvar,
                            "mismatch_label": mlabel,
                            "utfall": utfall,
                            "utfall_label": _UTFALL_LABELS[utfall],
                            "indikator_type": ind_type,
                            "indikator_type_label": ind_label,
                            "r_pearson": round(r_p, 3),
                            "p_pearson": round(p_p, 4),
                            "r_spearman": round(r_s, 3),
                            "p_spearman": round(p_s, 4),
                            "n": len(sub),
                        }
                    )

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# B. Regresjon per modell
# ═══════════════════════════════════════════════════════════════════════════════


def _newey_west_se(
    residuals: np.ndarray, X: np.ndarray, max_lag: int | None = None
) -> np.ndarray:
    """Beregn Newey–West standardfeil (kopiert fra analyse_mismatch.py)."""
    n, k = X.shape
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = residuals.reshape(-1, 1)
    S = np.zeros((k, k))
    for lag in range(max_lag + 1):
        w = 1.0 if lag == 0 else 1 - lag / (max_lag + 1)
        for t in range(lag, n):
            xt = X[t : t + 1].T
            xs = X[t - lag : t - lag + 1].T
            contrib = (e[t] * e[t - lag]) * (xt @ xs.T)
            if lag == 0:
                S += w * contrib
            else:
                S += w * (contrib + contrib.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(V))


def _regresjon_per_modell(df_lang: pd.DataFrame, df_aar: pd.DataFrame) -> pd.DataFrame:
    """OLS: indikator ~ mismatch + måned-dummyer, med Newey–West SE, per modell."""
    df = df_lang.copy()
    # Merge mismatch som step-funksjon via undersokelsesaar
    df = df.merge(
        df_aar[["aar"] + list(_MISMATCH_VARS.keys())],
        left_on="undersokelsesaar",
        right_on="aar",
        how="left",
    )

    records = []
    for modell in MODELL_REKKEFØLGE:
        for mvar, mlabel in _MISMATCH_VARS.items():
            for utfall in _UTFALL:
                for ind_type, ind_label in [
                    ("faktisk", "Faktisk"),
                    ("indikator", "Indikator (avvik)"),
                ]:
                    sub = (
                        df[(df["modell"] == modell) & (df["utfall"] == utfall)]
                        .dropna(subset=[mvar, ind_type])
                        .sort_values("beholdningsmaaned")
                    )
                    if len(sub) < 15:
                        continue

                    y = sub[ind_type].values
                    x_mm = sub[mvar].values
                    month_dummies = pd.get_dummies(
                        sub["maaned"], prefix="m", drop_first=True
                    ).values
                    X = np.column_stack([np.ones(len(y)), x_mm, month_dummies])

                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    y_hat = X @ beta
                    resid = y - y_hat
                    n, k = X.shape
                    nw_se = _newey_west_se(resid, X)

                    ss_res = np.sum(resid**2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

                    t_stat = beta[1] / nw_se[1] if nw_se[1] > 0 else np.nan
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - k))

                    records.append(
                        {
                            "modell": modell,
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

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Figurer
# ═══════════════════════════════════════════════════════════════════════════════


def _figur_mismatch_korrelasjon(df_korr: pd.DataFrame) -> None:
    """Stolpediagram: gjennomsnittlig |r| (JR-indeks, indikator-type) per modell."""
    sub = df_korr[
        (df_korr["mismatch_var"] == "mismatch_indeks")
        & (df_korr["indikator_type"] == "indikator")
    ]

    utfall_list = _UTFALL
    n_utfall = len(utfall_list)
    n_modeller = len(MODELL_REKKEFØLGE)
    x = np.arange(n_utfall)
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, modell in enumerate(MODELL_REKKEFØLGE):
        msub = sub[sub["modell"] == modell]
        vals = [
            msub[msub["utfall"] == u]["r_pearson"].abs().values[0]
            if not msub[msub["utfall"] == u].empty
            else np.nan
            for u in utfall_list
        ]
        ax.bar(
            x + (i - n_modeller / 2 + 0.5) * width,
            vals,
            width,
            label=modell,
            color=MODELL_FARGER[modell],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_UTFALL_LABELS[u] for u in utfall_list])
    ax.set_ylabel("|r| (Pearson)")
    ax.set_title(
        "Gjennomsnittlig |r|: Jackman–Roper mismatch-indeks vs. indikatoravvik"
    )
    ax.legend(title="Modell", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(_FIG_DIR / "mismatch_korrelasjon.png", dpi=FIG_DPI)
    plt.close(fig)
    print(f"  -> {_FIG_DIR / 'mismatch_korrelasjon.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sammendragstabell
# ═══════════════════════════════════════════════════════════════════════════════


def _lag_sammendrag(df_korr: pd.DataFrame) -> pd.DataFrame:
    """Tabell: |r|-reduksjon vs. referanse per modell og utfall (JR-indeks, indikator)."""
    sub = df_korr[
        (df_korr["mismatch_var"] == "mismatch_indeks")
        & (df_korr["indikator_type"] == "indikator")
    ]

    records = []
    for utfall in _UTFALL:
        usub = sub[sub["utfall"] == utfall]
        ref_r = usub[usub["modell"] == "Referanse"]["r_pearson"].abs().values
        if len(ref_r) == 0:
            continue
        ref_r = ref_r[0]
        for modell in MODELL_REKKEFØLGE:
            msub = usub[usub["modell"] == modell]["r_pearson"].abs().values
            if len(msub) == 0:
                continue
            records.append(
                {
                    "modell": modell,
                    "indikator_var": utfall,
                    "abs_r": round(msub[0], 3),
                    "r_reduksjon": round(ref_r - msub[0], 3),
                }
            )

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Hovedfunksjon: kjør alle analyser og lagre resultater for mismatch-analysen opp mot modellene."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    _TBL_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS.mkdir(parents=True, exist_ok=True)

    print("Leser data...")
    df_nasj = _last_og_konverter_nasjonalt(_PROCESSED / "modellendringer_nasjonalt.csv")
    df_nasj = _tilknytt_undersokelsesaar(df_nasj)
    df_aar = pd.read_csv(_PROCESSED / "mismatch_aar.csv")

    print(
        f"  Nasjonal data: {len(df_nasj)} rader, {df_nasj['modell'].nunique()} modeller"
    )
    print(
        f"  Mismatch-data: {len(df_aar)} år ({df_aar['aar'].min()}–{df_aar['aar'].max()})"
    )

    # ── A. Korrelasjoner ───────────────────────────────────────────────────
    print("\nA. Korrelasjoner per modell")
    df_korr = _korrelasjon_per_modell(df_nasj, df_aar)
    df_korr.to_csv(_RESULTS / "modellendringer_mismatch_korrelasjoner.csv", index=False)
    print(f"  -> {_RESULTS / 'modellendringer_mismatch_korrelasjoner.csv'}")

    print("\n  Utvalgte korrelasjoner (JR-indeks, indikator-type):")
    sub = df_korr[
        (df_korr["mismatch_var"] == "mismatch_indeks")
        & (df_korr["indikator_type"] == "indikator")
    ]
    for modell in MODELL_REKKEFØLGE:
        for utfall in _UTFALL:
            row = sub[(sub["modell"] == modell) & (sub["utfall"] == utfall)]
            if row.empty:
                continue
            r = row["r_pearson"].values[0]
            p = row["p_pearson"].values[0]
            print(
                f"    {modell:<20} {_UTFALL_LABELS[utfall]:<20} r={r:>6.3f}  p={p:.4f}"
            )

    # ── B. Regresjon ───────────────────────────────────────────────────────
    print("\nB. Regresjon per modell (Newey–West SE)")
    df_reg = _regresjon_per_modell(df_nasj, df_aar)
    df_reg.to_csv(_RESULTS / "modellendringer_mismatch_regresjon.csv", index=False)
    print(f"  -> {_RESULTS / 'modellendringer_mismatch_regresjon.csv'}")

    # ── Figurer ────────────────────────────────────────────────────────────
    print("\nFigurer")
    _figur_mismatch_korrelasjon(df_korr)

    # ── Sammendragstabell ──────────────────────────────────────────────────
    df_samm = _lag_sammendrag(df_korr)
    df_samm.to_csv(_TBL_DIR / "mismatch_sammendrag.csv", index=False)
    print(f"  -> {_TBL_DIR / 'mismatch_sammendrag.csv'}")

    print("\nFerdig!")


if __name__ == "__main__":
    main()
