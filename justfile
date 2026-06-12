# https://just.systems

# Hvis ingen kommando vis alle tilgjengelige oppskrifter
default:
    @just --list

# Klargjør prosjektet ved å installere `prek` og oppdatere avhengigheter fra malen
prepare:
    uv run --only-dev prek install
    uv lock --upgrade

# Fiks feil og formater kode med ruff
fix:
    uv run --only-dev ruff check --fix .
    uv run --only-dev ruff format .

# Sjekk at alt koden ser bra ut og er klar for å legges til i git
lint:
    uv run --only-dev prek run --all-files --color always

# Kjør testene
test:
    uv run --group dev pytest

# ─── Pipeline-faser ────────────────────────────────────────────────────────────

# Fase 1: Hent rådata fra Nav BigQuery og SSB → data/raw/
fetch:
    uv run python -m src.datagrunnlag.fetch_bedrifts_data
    uv run python -m src.datagrunnlag.fetch_indikator_data
    uv run python -m src.datagrunnlag.fetch_indikator_data --level nasjonalt
    uv run python -m src.datagrunnlag.fetch_coefficients_data
    uv run python -m src.datagrunnlag.fetch_ssb_data

# Fase 2: Standardiser rådata til felles geografi/tidsperiode → data/processed/
standardise:
    uv run python -m src.datagrunnlag.standardiser_data
    uv run python -m src.datagrunnlag.standardiser_mismatch_data
    uv run python -m src.datagrunnlag.standardiser_ssb_mismatch_data
    uv run python -m src.modellendringer.standardize_data

# Fase 3a: Stramhetsanalyser → quarto/stramhet/
stramhet:
    uv run python -m src.stramhet.analyse_arbeidsmarked
    uv run python -m src.stramhet.analyse_ssb_stramhet
    uv run python -m src.stramhet.lag_rapport_data

# Fase 3b: Mismatch-analyser → quarto/mismatch/
mismatch:
    uv run python -m src.mismatch.analyse_mismatch
    uv run python -m src.mismatch.analyse_mismatch_sensitivitet

# Fase 3c: Modellendringer → quarto/modellendringer/
modellendringer:
    uv run python -m src.modellendringer.analyse_mismatch_modeller
    uv run python -m src.modellendringer.analyse_stramhet_modeller
    uv run python -m src.modellendringer.analyse_koeffisienter
    uv run python -m src.modellendringer.analyse_nasjonalt
    uv run python -m src.modellendringer.analyse_regionalt
    uv run python -m src.modellendringer.lag_modellendringer_rapport_data

# Alle analyser (fase 3a + 3b + 3c)
analyse: stramhet mismatch modellendringer

# Hele pipelinen: hent → standardiser → analyser → bygg rapport
pipeline: fetch standardise analyse render

# ─── Rapport ───────────────────────────────────────────────────────────────────

# Lag et preview med Quarto
preview:
    uv run --group quarto quarto preview .

# Bygg Quarto-prosjektet
render:
    uv run --group quarto quarto render .

# Bygg prosjektet i Docker
[arg('image', pattern='chainguard_python.Dockerfile|Dockerfile')]
build image='Dockerfile':
    docker build -f {{image}} .

# Sjekk etter sårbarheter i Python-avhengigheter
audit:
    uv audit

# Oppdater Python og pre-commit avhengigheter
update:
    uv lock --upgrade
    uv run prek auto-update
