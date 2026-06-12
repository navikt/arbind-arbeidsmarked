"""Delt hjelper for BigQuery-spørringer i datagrunnlaget.

Samler det gjentatte mønsteret «opprett klient → kjør parametrisert spørring →
returner rader» som tidligere var duplisert i flere fetch-skript
(jf. CRITIQUE_METHOD_AND_CODE.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from google.cloud import bigquery


def query_rows(
    sql: str, parameters: Sequence[bigquery.ScalarQueryParameter | Any] | None = None
) -> list[dict[str, Any]]:
    """Kjør en parametrisert BigQuery-spørring og returner radene som dicts.

    Args:
        sql: SQL-spørringen som skal kjøres.
        parameters: Eventuelle query-parametre (f.eks.
            ``bigquery.ArrayQueryParameter``). ``None`` betyr ingen parametre.

    Returns:
        Én dict per rad i resultatet.
    """
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(query_parameters=list(parameters or []))
    query_job = client.query(sql, job_config=job_config)
    return [dict(row) for row in query_job.result()]
