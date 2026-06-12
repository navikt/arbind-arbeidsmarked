# Grunnlag for analyse av Arbeidsindikator og arbeidsmarkedet

Evaluerer om Navs arbeidsindikatorer i tilstrekkelig grad kontrollerer for
arbeidsmarkedet. Analysekoden ligger under `src/`, rapporten (Quarto-bok) under
`quarto/`, og data under `data/` (`raw/` → `processed/` → `results/`).

## Oppsett

```sh
just prepare        # installerer pre-commit-hooks og låser avhengigheter
```

Avhengigheter håndteres med [uv](https://docs.astral.sh/uv/). Skript kjøres som
moduler fra repo-roten med `uv run python -m <modulsti>` (se under). Dette gjør at
delte hjelpemoduler i `src/common/` kan importeres på tvers av analysene.

## Kjørerekkefølge

Pipelinen kjøres fra repo-roten i fire faser. Hvert skript dokumenterer sine
inn- og utdata i toppdocstringen.

1. **Hent rådata** → `data/raw/`
   ```sh
   uv run python -m src.datagrunnlag.fetch_bedrifts_data
   uv run python -m src.datagrunnlag.fetch_indikator_data
   uv run python -m src.datagrunnlag.fetch_coefficients_data
   uv run python -m src.datagrunnlag.fetch_ssb_data
   ```

2. **Standardiser** rådata til felles geografi/tidsperiode → `data/processed/`
   ```sh
   uv run python -m src.datagrunnlag.standardiser_data
   uv run python -m src.datagrunnlag.standardiser_mismatch_data
   uv run python -m src.datagrunnlag.standardiser_ssb_mismatch_data
   ```

3. **Analyser** og produser figurer/tabeller → `data/results/`, `quarto/**/figurer`, `quarto/**/tabeller`
   ```sh
   # Stramhet
   uv run python -m src.stramhet.analyse_arbeidsmarked
   uv run python -m src.stramhet.analyse_ssb_stramhet
   uv run python -m src.stramhet.lag_rapport_data
   # Mismatch
   uv run python -m src.mismatch.analyse_mismatch
   uv run python -m src.mismatch.analyse_mismatch_sensitivitet
   # Modellendringer
   uv run python -m src.modellendringer.analyse_mismatch_modeller
   uv run python -m src.modellendringer.analyse_stramhet_modeller
   uv run python -m src.modellendringer.lag_modellendringer_rapport_data
   ```

4. **Bygg rapporten** (Quarto-bok → `quarto/_book/`)
   ```sh
   just render         # eller: just preview
   ```

> Sentraliserte antakelser (referansemåneder, region-/fylkemappinger) ligger i
> `src/common/config.py`. Delte statistiske hjelpefunksjoner (Newey–West SE,
> korreksjon for multippel testing) ligger i `src/common/stats.py`.

## Utvikling

```sh
just lint           # ruff + mypy via pre-commit
just fix            # auto-fiks og formatering med ruff
just test           # kjør testene
```

Pipeline-snarveier:

```sh
just fetch          # hent rådata (BigQuery + SSB)
just standardise    # standardiser rådata
just stramhet       # stramhetsanalyser
just mismatch       # mismatch-analyser
just modellendringer # modellendringer
just analyse        # alle analyser (fase 3)
just pipeline       # hele pipelinen (fetch → standardise → analyse → render)
```
