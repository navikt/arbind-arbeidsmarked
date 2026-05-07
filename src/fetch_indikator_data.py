"""Fetch all regional Nav indicator data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery

_TABLE_URI = "arbeidsindikator-prod-51bc.arbeidsindikator.agg_indikator_siste_pub"
_TARGET_UTFALL = ["atid3", "jobb3", "atid12", "jobb12"]


def _query_nedbrytning(nedbrytning: str) -> list[dict[str, Any]]:
    """Fetch all indicator columns for one nedbrytning in a single query."""
    query = f"""
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
          AND NEDBRYTNING = @nedbrytning
          AND org_sted != 'Nasjonal oppfølgingsenhet'
          AND UTFALL IN UNNEST(@utfall)
        ORDER BY BEHOLDNINGSMAANED
    """
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("nedbrytning", "STRING", nedbrytning),
            bigquery.ArrayQueryParameter("utfall", "STRING", _TARGET_UTFALL),
        ]
    )
    query_job = client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]


if __name__ == "__main__":
    # Run the query for nedrbytning = 'Alle' and save the result as a csv file.
    nedbrytning = "Alle"
    print(f"Fetching data for nedbrytning = '{nedbrytning}'...")
    rows = _query_nedbrytning(nedbrytning)
    df = pd.DataFrame(rows)
    target_path = Path("data/raw/indikator_data.csv")
    print(f"Saving data to {target_path}...")
    df.to_csv(target_path, index=False)
    print("Done.")
