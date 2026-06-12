"""Sentraliserte antakelser og mappinger brukt på tvers av datagrunnlag og analyser.

Tidligere var disse tabellene innebygd i flere skript, slik at en analytiker måtte
inspisere flere filer for å bekrefte at antakelsene var konsistente
(jf. CRITIQUE_METHOD_AND_CODE.md). Alle delte antakelser samles her.
"""

from __future__ import annotations

# ── Tidsperiode ─────────────────────────────────────────────────────────────────
# Bedriftsundersøkelsen gjennomføres om våren med følgende referansemåned per år.
# Indikatordata hentes for tilsvarende referansemåned samme år.
#   2021: februar, 2022: april, 2023: april, 2024: mars, 2025: mars
REFERANSEMAANED: dict[int, int] = {
    2021: 2,  # februar
    2022: 4,  # april
    2023: 4,  # april
    2024: 3,  # mars
    2025: 3,  # mars
}

# ── Geografi ────────────────────────────────────────────────────────────────────
# Ordnet liste over Nav-regioner (brukt til konsistent sortering i rapporter).
NAV_REGIONER: list[str] = [
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
]

# Direkte navnmapping: bedriftsundersøkelse-regioner (2021–2023) → Nav-region.
REGION_TIL_NAV: dict[str, str] = {
    "Øst-Viken": "Nav Øst-Viken",
    "Vest-Viken": "Nav Vest-Viken",
    "Oslo": "Nav Oslo",
    "Innlandet": "Nav Innlandet",
    "Vestfold og Telemark": "Nav Vestfold og Telemark",
    "Agder": "Nav Agder",
    "Rogaland": "Nav Rogaland",
    "Vestland": "Nav Vestland",
    "Møre og Romsdal": "Nav Møre og Romsdal",
    "Trøndelag": "Nav Trøndelag",
    "Nordland": "Nav Nordland",
    "Troms og Finnmark": "Nav Troms og Finnmark",
}

# Fylke → Nav-region (2024–2025 bruker individuelle fylker).
FYLKE_TIL_NAV: dict[str, str] = {
    "Østfold": "Nav Øst-Viken",
    "Akershus": "Nav Øst-Viken",
    "Oslo": "Nav Oslo",
    "Innlandet": "Nav Innlandet",
    "Buskerud": "Nav Vest-Viken",
    "Vestfold": "Nav Vestfold og Telemark",
    "Telemark": "Nav Vestfold og Telemark",
    "Agder": "Nav Agder",
    "Rogaland": "Nav Rogaland",
    "Vestland": "Nav Vestland",
    "Møre og Romsdal": "Nav Møre og Romsdal",
    "Trøndelag": "Nav Trøndelag",
    "Nordland": "Nav Nordland",
    "Troms": "Nav Troms og Finnmark",
    "Finnmark": "Nav Troms og Finnmark",
}
