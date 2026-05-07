"""Genererer alle figurer og tabeller til rapportene (jobb og atid).

Lagrer filer til:
  quarto/jobb/figurer/   – PNG-figurer (jobbindikator)
  quarto/jobb/tabeller/  – CSV-tabeller (jobbindikator)
  quarto/atid/figurer/   – PNG-figurer (atid-indikator)
  quarto/atid/tabeller/  – CSV-tabeller (atid-indikator)

Kjøres før quarto render:
  uv run python src/lag_rapport_data.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ── Stier ─────────────────────────────────────────────────────────────────────
_RESULTS = Path("data/results")
_PROCESSED = Path("data/processed")

# ── NAV-farger og stil ─────────────────────────────────────────────────────────
NAV_BLÅ = "#0067C5"
NAV_GRØNN = "#06893A"
NAV_GRÅ = "#59514B"
YEAR_COLORS = {
    2021: "#0067C5",
    2022: "#C30000",
    2023: "#06893A",
    2024: "#FF9100",
    2025: "#59514B",
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


# ── Moduskonfigurasjon ─────────────────────────────────────────────────────────


@dataclass
class ModeConfig:
    """Konfigurasjon for en rapport-modus (jobb eller atid)."""

    mode: str
    faktisk3: str
    faktisk12: str
    indikator3: str
    indikator12: str
    faktisk3_label: str
    faktisk12_label: str
    indikator3_label: str
    indikator12_label: str
    ylabel3: str
    yunit: str
    fig_dir: Path
    tbl_dir: Path

    def setup_dirs(self) -> None:
        """Opprett output-mapper."""
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tbl_dir.mkdir(parents=True, exist_ok=True)


JOBB = ModeConfig(
    mode="jobb",
    faktisk3="faktisk_jobb3",
    faktisk12="faktisk_jobb12",
    indikator3="indikator_jobb3",
    indikator12="indikator_jobb12",
    faktisk3_label="Faktisk jobb 3 mnd (%)",
    faktisk12_label="Faktisk jobb 12 mnd (%)",
    indikator3_label="Indikator jobb 3 mnd (avvik fra forventet, pp)",
    indikator12_label="Indikator jobb 12 mnd (avvik fra forventet, pp)",
    ylabel3="Faktisk andel i jobb etter 3 mnd (%)",
    yunit="%",
    fig_dir=Path("quarto/jobb/figurer"),
    tbl_dir=Path("quarto/jobb/tabeller"),
)

ATID = ModeConfig(
    mode="atid",
    faktisk3="faktisk_atid3",
    faktisk12="faktisk_atid12",
    indikator3="indikator_atid3",
    indikator12="indikator_atid12",
    faktisk3_label="Faktisk atid 3 mnd (t/uke)",
    faktisk12_label="Faktisk atid 12 mnd (t/uke)",
    indikator3_label="Indikator atid 3 mnd (avvik fra forventet, t/uke)",
    indikator12_label="Indikator atid 12 mnd (avvik fra forventet, t/uke)",
    ylabel3="Faktisk gj.sn. ukentlige arbeidstimer etter 3 mnd (t/uke)",
    yunit="t/uke",
    fig_dir=Path("quarto/atid/figurer"),
    tbl_dir=Path("quarto/atid/tabeller"),
)


# ── Hjelpefunksjoner ───────────────────────────────────────────────────────────


def _save(fig: plt.Figure, name: str, cfg: ModeConfig) -> None:
    """Lagre figur til mode-spesifikt figur-katalog."""
    path = cfg.fig_dir / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Lagret {path}")


def _last_data() -> tuple[pd.DataFrame, ...]:
    """Last inn alle resultat- og prosesserte datafiler."""
    df = pd.read_csv(_RESULTS / "analyse_ar_faste_effekter.csv")
    ts = pd.read_csv(_RESULTS / "analyse_nasjonal_tidsserie.csv")
    kor = pd.read_csv(_RESULTS / "analyse_korrelasjoner.csv")
    baro = pd.read_csv(_PROCESSED / "sysselsettingsbarometer.csv")
    yrke = pd.read_csv(_PROCESSED / "mangel_per_yrkesgruppe.csv")
    return df, ts, kor, baro, yrke


# ══════════════════════════════════════════════════════════════════════════════
# FIGURER
# ══════════════════════════════════════════════════════════════════════════════


def fig_nasjonal(ts: pd.DataFrame, baro: pd.DataFrame, cfg: ModeConfig) -> None:
    """Nasjonal stramhet og sysselsettingsbarometer (felles for alle modi)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1, ax2 = axes

    ax1b = ax1.twinx()
    ax1.bar(
        ts["aar"],
        ts["stramhetsindikator"],
        color=NAV_BLÅ,
        alpha=0.75,
        width=0.4,
        label="Stramhetsindikator (venstre akse)",
    )
    ax1b.plot(
        ts["aar"],
        ts["andel_alvorlige_rekrutteringsproblemer_pst"],
        color=NAV_GRÅ,
        marker="o",
        linewidth=2,
        label="Andel rekrutteringsproblemer % (høyre)",
    )
    ax1.set_xlabel("År")
    ax1.set_ylabel("Stramhetsindikator (snitt regioner)")
    ax1b.set_ylabel("Andel virksomheter m/ alvorlige\nrekrutteringsproblemer (%)")
    ax1.set_xticks(ts["aar"])
    ax1.spines["top"].set_visible(False)
    ax1b.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax1.set_title("Arbeidsmarkedsstramhet", fontsize=10, fontweight="bold")

    baro_vis = baro[baro["aar"] >= 2015].copy()
    ax2.fill_between(
        baro_vis["aar"],
        baro_vis["okning_pst"],
        alpha=0.35,
        color=NAV_GRØNN,
        label="Forventer økning (%)",
    )
    ax2.fill_between(
        baro_vis["aar"],
        -baro_vis["nedgang_pst"],
        alpha=0.35,
        color="#C30000",
        label="Forventer nedgang (%, neg. skala)",
    )
    ax2.plot(
        baro_vis["aar"],
        baro_vis["okning_pst"] - baro_vis["nedgang_pst"],
        color="black",
        linewidth=2,
        label="Nettoandel",
    )
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax2.set_xlabel("År")
    ax2.set_ylabel("Prosent")
    ax2.set_title("Sysselsettingsbarometer (nasjonalt)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "fig_nasjonal", cfg)


def fig_ind_tid(ts: pd.DataFrame, cfg: ModeConfig) -> None:
    """Nasjonal faktisk og indikator over tid for den aktuelle modus."""
    if cfg.faktisk3 not in ts.columns:
        print(
            f"  Hopper over fig_ind_tid – {cfg.faktisk3} mangler i nasjonal tidsserie"
        )
        return
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        ts["aar"] - 0.2,
        ts[cfg.faktisk3],
        width=0.38,
        color=NAV_BLÅ,
        alpha=0.8,
        label=cfg.faktisk3_label,
    )
    ax2 = ax.twinx()
    ax2.plot(
        ts["aar"],
        ts[cfg.indikator3],
        color=NAV_GRÅ,
        marker="o",
        linewidth=2.5,
        label=cfg.indikator3_label,
    )
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)

    ax.set_ylabel(cfg.ylabel3)
    ax2.set_ylabel(f"Avvik fra forventet ({cfg.yunit})")
    ax.set_xticks(ts["aar"])
    ax.set_xlabel("År")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    short = "jobbsannsynlighet" if cfg.mode == "jobb" else "arbeidstimer (3 mnd)"
    ax.set_title(
        f"Indikator og faktisk {short} – nasjonalt gjennomsnitt",
        fontsize=10,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    plt.tight_layout()
    _save(fig, "fig_ind_tid", cfg)


def fig_yrke(yrke: pd.DataFrame, years: list[int], cfg: ModeConfig) -> None:
    """Mangel per yrkesgruppe 2021–2025 (felles for alle modi)."""
    yrke_v = yrke[yrke["yrkesgruppe"] != "Ingen yrkesbakgrunn eller uoppgitt"].copy()
    order = (
        yrke_v[yrke_v["aar"] == max(years)]
        .sort_values("mangel_antall", ascending=False)["yrkesgruppe"]
        .tolist()
    )
    pivot = yrke_v.pivot_table(
        index="yrkesgruppe", columns="aar", values="mangel_antall", aggfunc="sum"
    )
    pivot = pivot.loc[[y for y in order if y in pivot.index]]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(pivot))
    w = 0.16
    offsets = np.linspace(-(len(years) - 1) / 2, (len(years) - 1) / 2, len(years)) * w
    for i, yr in enumerate(years):
        if yr not in pivot.columns:
            continue
        vals = pivot[yr].fillna(0)
        ax.bar(
            x + offsets[i],
            vals / 1000,
            width=w,
            color=YEAR_COLORS[yr],
            label=str(yr),
            alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Estimert mangel (antall tusen)")
    ax.set_title(
        "Mangel på arbeidskraft etter yrkesgruppe", fontsize=11, fontweight="bold"
    )
    ax.legend(title="År", ncol=5, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}k"))
    plt.tight_layout()
    _save(fig, "fig_yrke", cfg)


def fig_regional(df: pd.DataFrame, years: list[int], cfg: ModeConfig) -> None:
    """Stramhetsindikator per Nav-region og år (felles for alle modi)."""
    pivot = df.pivot_table(
        index="nav_region", columns="aar", values="stramhetsindikator"
    )
    order = pivot.mean(axis=1).sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    w = 0.16
    offsets = np.linspace(-(len(years) - 1) / 2, (len(years) - 1) / 2, len(years)) * w
    for i, yr in enumerate(years):
        if yr not in pivot.columns:
            continue
        vals = [pivot.loc[r, yr] if r in pivot.index else np.nan for r in order]
        ax.bar(
            x + offsets[i],
            vals,
            width=w,
            color=YEAR_COLORS[yr],
            label=str(yr),
            alpha=0.88,
        )

    labels = [r.replace("Nav ", "") for r in order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Stramhetsindikator")
    ax.set_title(
        "Stramhetsindikator etter Nav-region og år", fontsize=11, fontweight="bold"
    )
    ax.legend(title="År", ncol=5, fontsize=9)
    plt.tight_layout()
    _save(fig, "fig_regional", cfg)


def fig_scatter_all(df: pd.DataFrame, years: list[int], cfg: ModeConfig) -> None:
    """Scatter: stramhet vs utfall – alle observasjoner."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (yvar, ylabel, tittel) in zip(
        axes,
        [
            (cfg.faktisk3, cfg.ylabel3, f"vs. faktisk {cfg.mode} 3 mnd"),
            (
                cfg.indikator3,
                f"Avvik fra forventet ({cfg.yunit})",
                f"vs. indikator {cfg.mode} 3 mnd",
            ),
        ],
    ):
        for yr in years:
            sub = df[
                (df["aar"] == yr) & df["stramhetsindikator"].notna() & df[yvar].notna()
            ]
            ax.scatter(
                sub["stramhetsindikator"],
                sub[yvar],
                color=YEAR_COLORS[yr],
                s=60,
                alpha=0.85,
                label=str(yr),
                zorder=3,
            )
        pair = df[["stramhetsindikator", yvar]].dropna()
        slope, intercept, r, *_ = stats.linregress(
            pair["stramhetsindikator"], pair[yvar]
        )
        xs = np.linspace(
            pair["stramhetsindikator"].min(), pair["stramhetsindikator"].max(), 100
        )
        ax.plot(
            xs,
            slope * xs + intercept,
            color="black",
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
            label=f"Trend (r={r:.2f})",
        )
        ax.set_xlabel("Stramhetsindikator")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=3)
        ax.set_title(tittel, fontsize=10, fontweight="bold")

    plt.suptitle(
        f"Stramhetsindikator og {cfg.mode}-resultater (alle år og regioner)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save(fig, "fig_scatter_all", cfg)


def fig_scatter_years(df: pd.DataFrame, cfg: ModeConfig) -> None:
    """Scatter per år med regresjonslinje og regionetiketter."""
    plot_years = [2021, 2022, 2023, 2024]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)

    for ax, yr in zip(axes, plot_years):
        sub = df[df["aar"] == yr][
            ["stramhetsindikator", cfg.faktisk3, "nav_region"]
        ].dropna()
        ax.scatter(
            sub["stramhetsindikator"],
            sub[cfg.faktisk3],
            color=YEAR_COLORS[yr],
            s=65,
            alpha=0.9,
            zorder=3,
        )
        for _, row in sub.iterrows():
            short = row["nav_region"].replace("Nav ", "").replace(" og ", "/")
            ax.annotate(
                short,
                (row["stramhetsindikator"], row[cfg.faktisk3]),
                fontsize=6.5,
                alpha=0.75,
                xytext=(3, 2),
                textcoords="offset points",
            )
        if len(sub) > 2:
            slope, intercept, r, p, _ = stats.linregress(
                sub["stramhetsindikator"], sub[cfg.faktisk3]
            )
            xs = np.linspace(
                sub["stramhetsindikator"].min(), sub["stramhetsindikator"].max(), 100
            )
            ax.plot(
                xs,
                slope * xs + intercept,
                color="black",
                linewidth=1.5,
                linestyle="--",
                alpha=0.55,
            )
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax.set_title(f"{yr}\nr = {r:.2f}{sig}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Stramhetsindikator")
        if yr == 2021:
            ax.set_ylabel(cfg.ylabel3)

    plt.suptitle(
        f"Krysseksjonell sammenheng per år – {cfg.faktisk3} (* p<0,05  ** p<0,01)",
        fontsize=10,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, "fig_scatter_years", cfg)


def fig_kor_heat(kor: pd.DataFrame, cfg: ModeConfig) -> None:
    """Heatmap av Pearson r per år for den aktuelle modus."""
    col_order = [cfg.faktisk3, cfg.faktisk12, cfg.indikator3, cfg.indikator12]
    heat = kor[kor["bedrifts_var"] == "stramhetsindikator"].pivot(
        index="aar", columns="indikator_var", values="r"
    )
    heat = heat[[c for c in col_order if c in heat.columns]]
    if heat.empty:
        print(f"  Hopper over fig_kor_heat – ingen data for {cfg.mode}")
        return

    col_labels = [
        cfg.faktisk3_label,
        cfg.faktisk12_label,
        cfg.indikator3_label,
        cfg.indikator12_label,
    ]
    col_labels = col_labels[: heat.shape[1]]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(heat.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index.astype(int))

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if abs(val) < 0.7 else "white",
                    fontweight="bold",
                )

    ax.set_title(
        f"Korrelasjon: Stramhetsindikator vs. {cfg.mode}-utfall per år",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "fig_kor_heat", cfg)


def fig_demeaned(df: pd.DataFrame, years: list[int], cfg: ModeConfig) -> None:
    """Demeaned scatter – år-faste effekter fjernet."""
    dm3_f = f"{cfg.faktisk3}_dm"
    dm3_i = f"{cfg.indikator3}_dm"
    dm_x = "stramhetsindikator_dm"

    if dm3_f not in df.columns:
        print(f"  Hopper over fig_demeaned – {dm3_f} mangler")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (xv, yv, tit) in zip(
        axes,
        [
            (dm_x, dm3_f, f"Stramhetsindikator → Faktisk {cfg.mode} 3 mnd"),
            (dm_x, dm3_i, f"Stramhetsindikator → Indikator {cfg.mode} 3 mnd"),
        ],
    ):
        for yr in years:
            sub = df[(df["aar"] == yr) & df[xv].notna() & df[yv].notna()]
            ax.scatter(
                sub[xv],
                sub[yv],
                color=YEAR_COLORS[yr],
                s=55,
                alpha=0.85,
                label=str(yr),
                zorder=3,
            )
        pair = df[[xv, yv]].dropna()
        r, p = stats.pearsonr(pair[xv], pair[yv])
        slope, intercept = np.polyfit(pair[xv], pair[yv], 1)
        xs = np.linspace(pair[xv].min(), pair[xv].max(), 100)
        ax.plot(xs, slope * xs + intercept, "k--", linewidth=1.5, alpha=0.55)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        ax.set_title(f"{tit}\nr = {r:.2f}{sig}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Demeaned stramhetsindikator")
        ax.set_ylabel(f"Demeaned utfallsvariabel ({cfg.yunit})")

    axes[0].legend(fontsize=8, ncol=3, title="År")
    plt.suptitle(
        "Samvariasjon etter fjerning av år-faste effekter",
        fontsize=10,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, "fig_demeaned", cfg)


# ══════════════════════════════════════════════════════════════════════════════
# TABELLER
# ══════════════════════════════════════════════════════════════════════════════


def tbl_oversikt(df: pd.DataFrame, years: list[int], cfg: ModeConfig) -> None:
    """Antall regioner og komplette observasjoner per år."""
    rows = []
    for y in years:
        sub = df[df["aar"] == y]
        rows.append(
            {
                "År": y,
                "Regioner": sub["nav_region"].nunique(),
                f"{cfg.faktisk3} (n)": int(sub[cfg.faktisk3].notna().sum()),
                f"{cfg.faktisk12} (n)": int(sub[cfg.faktisk12].notna().sum()),
            }
        )
    pd.DataFrame(rows).to_csv(cfg.tbl_dir / "tbl_oversikt.csv", index=False)
    print(f"  Lagret {cfg.tbl_dir / 'tbl_oversikt.csv'}")


def tbl_geomap(cfg: ModeConfig) -> None:
    """Geografisk mapping fra undersøkelsesenheter til Nav-regioner."""
    geo = pd.DataFrame(
        {
            "Nav-region": [
                "Nav Øst-Viken",
                "Nav Vest-Viken",
                "Nav Oslo",
                "Nav Innlandet",
                "Nav Vestfold og Telemark",
                "Nav Agder",
                "Nav Rogaland",
                "Nav Vestland",
                "Nav Møre og Romsdal",
                "Nav Trøndelag",
                "Nav Nordland",
                "Nav Troms og Finnmark",
            ],
            "2021–2023 (region)": [
                "Øst-Viken",
                "Vest-Viken",
                "Oslo",
                "Innlandet",
                "Vestfold og Telemark",
                "Agder",
                "Rogaland",
                "Vestland",
                "Møre og Romsdal",
                "Trøndelag",
                "Nordland",
                "Troms og Finnmark",
            ],
            "2024–2025 (fylker aggregert)": [
                "Østfold + Akershus",
                "Buskerud",
                "Oslo",
                "Innlandet",
                "Vestfold + Telemark",
                "Agder",
                "Rogaland",
                "Vestland",
                "Møre og Romsdal",
                "Trøndelag",
                "Nordland",
                "Troms + Finnmark",
            ],
        }
    )
    geo.to_csv(cfg.tbl_dir / "tbl_geomap.csv", index=False)
    print(f"  Lagret {cfg.tbl_dir / 'tbl_geomap.csv'}")


def tbl_kor_per_aar(kor: pd.DataFrame, cfg: ModeConfig) -> None:
    """Krysseksjonell Pearson r per år med signifikansstjerner."""
    col_map = {
        cfg.faktisk3: cfg.faktisk3_label,
        cfg.faktisk12: cfg.faktisk12_label,
        cfg.indikator3: cfg.indikator3_label,
        cfg.indikator12: cfg.indikator12_label,
    }
    stram = kor[kor["bedrifts_var"] == "stramhetsindikator"].copy()
    stram = stram[stram["indikator_var"].isin(col_map)]
    stram["r_vis"] = stram.apply(
        lambda row: f"{row['r']:.3f}{'*' if row['signifikant_5pst'] else ''}",
        axis=1,
    )
    stram["indikator_label"] = stram["indikator_var"].map(col_map)
    pivot = stram.pivot(index="aar", columns="indikator_label", values="r_vis")
    col_order = [col_map[k] for k in col_map if col_map[k] in pivot.columns]
    pivot = pivot[col_order].reset_index().rename(columns={"aar": "År"})
    pivot.to_csv(cfg.tbl_dir / "tbl_kor_per_aar.csv", index=False)
    print(f"  Lagret {cfg.tbl_dir / 'tbl_kor_per_aar.csv'}")


def tbl_kor_pooled(df: pd.DataFrame, cfg: ModeConfig) -> None:
    """Pooled r med år-faste effekter for den aktuelle modus."""
    from scipy.stats import pearsonr  # noqa: PLC0415

    bedrifts_vars = {
        "stramhetsindikator": "Stramhetsindikator",
        "andel_alvorlige_rekrutteringsproblemer_pst": "Andel rekrutteringsproblemer (%)",
    }
    ind_vars = {
        cfg.faktisk3: cfg.faktisk3_label,
        cfg.faktisk12: cfg.faktisk12_label,
        cfg.indikator3: cfg.indikator3_label,
        cfg.indikator12: cfg.indikator12_label,
    }

    rows = []
    for bv, bv_label in bedrifts_vars.items():
        for iv, iv_label in ind_vars.items():
            dm_b, dm_i = f"{bv}_dm", f"{iv}_dm"
            if dm_b not in df.columns or dm_i not in df.columns:
                continue
            pair = df[[dm_b, dm_i]].dropna()
            if len(pair) < 4:
                continue
            r, p = pearsonr(pair[dm_b], pair[dm_i])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            rows.append(
                {
                    "Forklaringsvariabel": bv_label,
                    "Utfallsvariabel": iv_label,
                    "r": round(r, 3),
                    "p-verdi": round(p, 4),
                    "Sig.": sig,
                    "n": len(pair),
                }
            )

    pd.DataFrame(rows).to_csv(cfg.tbl_dir / "tbl_kor_pooled.csv", index=False)
    print(f"  Lagret {cfg.tbl_dir / 'tbl_kor_pooled.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def _generer_for_modus(
    df: pd.DataFrame,
    ts: pd.DataFrame,
    kor: pd.DataFrame,
    baro: pd.DataFrame,
    yrke: pd.DataFrame,
    years: list[int],
    cfg: ModeConfig,
) -> None:
    """Generer alle figurer og tabeller for én rapport-modus."""
    cfg.setup_dirs()
    print(f"\n=== Modus: {cfg.mode.upper()} ===")

    print("  Figurer...")
    fig_nasjonal(ts, baro, cfg)
    fig_ind_tid(ts, cfg)
    fig_yrke(yrke, years, cfg)
    fig_regional(df, years, cfg)
    fig_scatter_all(df, years, cfg)
    fig_scatter_years(df, cfg)
    fig_kor_heat(kor, cfg)
    fig_demeaned(df, years, cfg)

    print("  Tabeller...")
    tbl_oversikt(df, years, cfg)
    tbl_geomap(cfg)
    tbl_kor_per_aar(kor, cfg)
    tbl_kor_pooled(df, cfg)


def main() -> None:
    """Generer alle figurer og tabeller for begge rapporter (jobb og atid)."""
    print("Leser data...")
    df, ts, kor, baro, yrke = _last_data()
    years = sorted(df["aar"].unique())
    print(f"  {len(df)} rader, år {min(years)}–{max(years)}")

    _generer_for_modus(df, ts, kor, baro, yrke, years, JOBB)
    _generer_for_modus(df, ts, kor, baro, yrke, years, ATID)

    print("\nFerdig. Kjør nå:")
    print("  uv run --group quarto quarto render quarto/")


if __name__ == "__main__":
    main()
