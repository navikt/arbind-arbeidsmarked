"""Create descriptive Quarto artifacts for new labour-market source panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_PROCESSED = Path("data/processed")
_OUT = Path("quarto/datagrunnlag")


def _save_labour_market_artifacts() -> None:
    df = pd.read_csv(_PROCESSED / "arbeidsmarkedsmaal_kommune.csv")
    measures = [
        "sysselsettingsrate",
        "vekst_sysselsetting_bosted",
        "vekst_sysselsetting_arbeidssted",
        "arbeidsplassdekning",
        "andel_privat_sysselsetting",
    ]
    coverage = (
        df.groupby("aar", as_index=False)
        .agg(
            kommuner=("kommunenummer", "nunique"),
            nav_regioner=("nav_region", "nunique"),
        )
        .rename(
            columns={
                "aar": "År",
                "kommuner": "Kommuner",
                "nav_regioner": "Nav-regioner",
            }
        )
    )
    coverage.to_csv(_OUT / "tabeller/arbeidsmarkedsmaal_dekning.csv", index=False)
    trends = (
        df.groupby("aar")[measures].median().reset_index().rename(columns={"aar": "År"})
    )
    trends.to_csv(_OUT / "tabeller/arbeidsmarkedsmaal_median.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df.groupby("aar")["sysselsettingsrate"].median(), marker="o")
    axes[0].set(title="Median sysselsettingsrate", xlabel="År", ylabel="Prosent")
    axes[1].plot(df.groupby("aar")["arbeidsplassdekning"].median(), marker="o")
    axes[1].set(title="Median arbeidsplassdekning", xlabel="År", ylabel="Forholdstall")
    fig.tight_layout()
    fig.savefig(_OUT / "figurer/arbeidsmarkedsmaal_trend.png", dpi=160)
    plt.close(fig)

    labels = {
        "sysselsettingsrate": "Sysselsettingsrate (%)",
        "vekst_sysselsetting_bosted": "Vekst, bosted",
        "vekst_sysselsetting_arbeidssted": "Vekst, arbeidssted",
        "arbeidsplassdekning": "Arbeidsplassdekning",
        "andel_privat_sysselsetting": "Privat sysselsettingsandel",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, measure in zip(axes.flat, measures):
        grouped = df.groupby("aar")[measure].median()
        axis.plot(grouped.index, grouped, marker="o")
        axis.set(title=labels[measure], xlabel="År")
        axis.tick_params(axis="x", rotation=45)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Median for kommunene", fontsize=13)
    fig.tight_layout()
    fig.savefig(_OUT / "figurer/arbeidsmarkedsmaal_alle_trender.png", dpi=160)
    plt.close(fig)

    latest = df[df["aar"] == df["aar"].max()]
    regional = latest.groupby("nav_region")[measures].median()
    standardized = (regional - regional.mean()) / regional.std(ddof=0)
    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(standardized, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axis.set(
        xticks=range(len(measures)),
        xticklabels=[labels[measure] for measure in measures],
        yticks=range(len(regional)),
        yticklabels=regional.index,
        title=f"Regional variasjon i {int(latest['aar'].max())} (median per kommune, z-skår)",
    )
    axis.tick_params(axis="x", rotation=35, labelsize=8)
    fig.colorbar(image, ax=axis, label="Standardavvik fra gjennomsnittet")
    fig.tight_layout()
    fig.savefig(_OUT / "figurer/arbeidsmarkedsmaal_regioner.png", dpi=160)
    plt.close(fig)


def _save_bankruptcy_artifacts() -> None:
    df = pd.read_csv(_PROCESSED / "konkurser_nav_region_maaned.csv")
    coverage = (
        df.groupby(df["maaned"].str[:4], as_index=False)
        .agg(
            nav_regioner=("nav_region", "nunique"),
            konkurser=("foretakskonkurser", "sum"),
        )
        .rename(
            columns={
                "maaned": "År",
                "nav_regioner": "Nav-regioner",
                "konkurser": "Foretakskonkurser",
            }
        )
    )
    coverage.to_csv(_OUT / "tabeller/konkurser_dekning.csv", index=False)
    national = df.groupby("maaned", as_index=False)["foretakskonkurser"].sum()
    national.to_csv(_OUT / "tabeller/konkurser_nasjonalt_maaned.csv", index=False)

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(
        pd.to_datetime(national["maaned"].str.replace("M", "-")),
        national["foretakskonkurser"],
    )
    axis.set(title="Opna føretakskonkursar", xlabel="", ylabel="Tal konkursar")
    fig.tight_layout()
    fig.savefig(_OUT / "figurer/konkurser_maaned.png", dpi=160)
    plt.close(fig)

    regional = df.pivot(
        index="maaned", columns="nav_region", values="foretakskonkurser"
    )
    regional.index = pd.to_datetime(regional.index.str.replace("M", "-"))
    fig, axis = plt.subplots(figsize=(11, 5))
    for region in regional:
        axis.plot(regional.index, regional[region], label=region.removeprefix("Nav "))
    axis.set(
        title="Opna føretakskonkursar per Nav-region", xlabel="", ylabel="Tal konkursar"
    )
    axis.legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(_OUT / "figurer/konkurser_regioner_maaned.png", dpi=160)
    plt.close(fig)

    annual = df.assign(aar=df["maaned"].str[:4]).pivot_table(
        index="nav_region", columns="aar", values="foretakskonkurser", aggfunc="sum"
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(annual, aspect="auto", cmap="YlOrRd")
    axis.set(
        xticks=range(len(annual.columns)),
        xticklabels=annual.columns,
        yticks=range(len(annual)),
        yticklabels=annual.index,
        title="Årssum av opna føretakskonkursar",
    )
    fig.colorbar(image, ax=axis, label="Tal konkursar")
    fig.tight_layout()
    fig.savefig(_OUT / "figurer/konkurser_regioner_aar.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Write all descriptive artifacts; these files are not analysis inputs."""
    (_OUT / "figurer").mkdir(parents=True, exist_ok=True)
    (_OUT / "tabeller").mkdir(parents=True, exist_ok=True)
    _save_labour_market_artifacts()
    _save_bankruptcy_artifacts()


if __name__ == "__main__":
    main()
