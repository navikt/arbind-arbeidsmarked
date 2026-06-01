"""Fetch coefficients and results for the indicator models from different types of modelsfrom BigQuery and save them as CSV files."""

from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery

_TABLE_URI = "arbeidsindikator-prod-51bc.arbeidsindikator.model_metrics_hist"
_RESULTS_URI = "arbeidsindikator-prod-51bc.arbeidsindikator.AGG_INDIKATOR_HIST"

_MODELS = {
    "referanse": 77,
    "stillingsrate": 78,
    "shiftshare": 79,
    "ledighetsrate": 80,
    "ledighetsrate_ung": 81,
}

_UTFALL = ["atid3", "jobb3", "atid12", "jobb12"]

QUERIES = {
    "model": f"""
        SELECT
            result_id,
            response_variable,
            variable,
            coefficient,
            p_value,
            std_err,
            t_value,
            p_value
        FROM `{_TABLE_URI}`
        where result_id IN UNNEST(@result_id)
    """,
    "region": f"""
        SELECT
            result_id,
            AARMND_DATO AS beholdningsmaaned,
            org_sted AS org_sted,
            indi_atid3_avg AS indikator_atid3,
            yhat_atid3_avg AS forventet_atid3,
            atid3_avg AS faktisk_atid3,
            indi_atid12_avg AS indikator_atid12,
            yhat_atid12_avg AS forventet_atid12,
            atid12_avg AS faktisk_atid12,
            indi_jobb3_avg AS indikator_jobb3,
            yhat_jobb3_avg AS forventet_jobb3,
            jobb3_avg AS faktisk_jobb3,
            indi_jobb12_avg AS indikator_jobb12,
            yhat_jobb12_avg AS forventet_jobb12,
            jobb12_avg AS faktisk_jobb12,
            ANTALL_PERSONER AS antall_personer
        FROM `{_RESULTS_URI}`
        WHERE ORG_NIVAA = 2
          AND nedbrytning_navn like '%organisasjon%'
          AND org_sted != 'Nasjonal oppfølgingsenhet'
          AND RESULT_ID IN UNNEST(@result_id)
        ORDER BY beholdningsmaaned
    """,
    "nasjonalt": f"""
        SELECT
            result_id,
            AARMND_DATO AS beholdningsmaaned,
            org_sted AS org_sted,
            indi_atid3_avg AS indikator_atid3,
            yhat_atid3_avg AS forventet_atid3,
            atid3_avg AS faktisk_atid3,
            indi_atid12_avg AS indikator_atid12,
            yhat_atid12_avg AS forventet_atid12,
            atid12_avg AS faktisk_atid12,
            indi_jobb3_avg AS indikator_jobb3,
            yhat_jobb3_avg AS forventet_jobb3,
            jobb3_avg AS faktisk_jobb3,
            indi_jobb12_avg AS indikator_jobb12,
            yhat_jobb12_avg AS forventet_jobb12,
            jobb12_avg AS faktisk_jobb12,
            ANTALL_PERSONER AS antall_personer
        FROM `{_RESULTS_URI}`
        WHERE ORG_NIVAA = 1
          AND nedbrytning_navn like '%organisasjon%'
          AND RESULT_ID IN UNNEST(@result_id)
        ORDER BY beholdningsmaaned
    """,
}


def _query_coefficients(query: str) -> list[dict[str, Any]]:
    """Fetch all coefficients for the indicator models in a single query."""
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("result_id", "FLOAT64", _MODELS.values()),
        ]
    )
    query_job = client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]


def _query_results(query: str) -> list[dict[str, Any]]:
    """Fetch all results for the indicator models in a single query."""
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("result_id", "FLOAT64", _MODELS.values()),
        ]
    )
    query_job = client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]


if __name__ == "__main__":
    print("Fetching coefficients for arb.markedsmodeller...")
    rows = _query_coefficients(QUERIES["model"])
    df = pd.DataFrame(rows)
    target_path = Path("data/raw/koeffisienter/indikator_data_model.csv")
    print(f"Saving data to {target_path}...")
    df.to_csv(target_path, index=False)

    print("Fetching results for arb.markedsmodeller...")
    for level in ["region", "nasjonalt"]:
        print(f"Fetching data for level = '{level}'...")
        rows = _query_results(QUERIES[level])
        df = pd.DataFrame(rows)
        target_path = Path(f"data/raw/koeffisienter/indikator_data_{level}.csv")
        print(f"Saving data to {target_path}...")
        df.to_csv(target_path, index=False)
    print("Done.")
