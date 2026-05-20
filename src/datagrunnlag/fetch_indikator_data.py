"""Fetch all regional Nav indicator data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery

_TABLE_URI = "arbeidsindikator-prod-51bc.arbeidsindikator.agg_indikator_siste_pub"
_TARGET_UTFALL = ["atid3", "jobb3", "atid12", "jobb12"]

QUERIES = {
    "region": f"""
        SELECT
            BEHOLDNINGSMAANED AS beholdningsmaaned,
            org_sted AS org_sted,
            UTFALL AS utfall,
            INDIKATOR AS indikator,
            forventet AS forventet,
            faktisk AS faktisk,
            ANTALL_PERSONER AS antall_personer,
            NEDBRYTNING AS nedbrytning
        FROM `{_TABLE_URI}`
        WHERE ORG_NIVAA = 2
          AND NEDBRYTNING = 'Alle'
          AND org_sted != 'Nasjonal oppfølgingsenhet'
          AND UTFALL IN UNNEST(@utfall)
        ORDER BY BEHOLDNINGSMAANED
    """,
    "nasjonalt": f"""
        SELECT
            BEHOLDNINGSMAANED AS beholdningsmaaned,
            org_sted AS org_sted,
            UTFALL AS utfall,
            INDIKATOR AS indikator,
            forventet AS forventet,
            faktisk AS faktisk,
            ANTALL_PERSONER AS antall_personer,
            NEDBRYTNING AS nedbrytning
        FROM `{_TABLE_URI}`
        WHERE ORG_NIVAA = 1
          AND NEDBRYTNING = 'Alle'
          AND UTFALL IN UNNEST(@utfall)
        ORDER BY BEHOLDNINGSMAANED
    """,
}


def _query_nedbrytning(query: str) -> list[dict[str, Any]]:
    """Fetch all indicator columns for one nedbrytning in a single query."""
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("utfall", "STRING", _TARGET_UTFALL),
        ]
    )
    query_job = client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch indicator data")
    parser.add_argument(
        "--level",
        choices=["region", "nasjonalt"],
        default="region",
        help="Data level to fetch (default: region)",
    )
    args = parser.parse_args()

    print(f"Fetching data for level = '{args.level}'...")
    rows = _query_nedbrytning(QUERIES[args.level])
    df = pd.DataFrame(rows)
    target_path = Path(f"data/raw/indikator_data_{args.level}.csv")
    print(f"Saving data to {target_path}...")
    df.to_csv(target_path, index=False)
    print("Done.")
