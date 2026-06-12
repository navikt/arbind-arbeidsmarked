"""Tester for valideringshjelpere i src/common/validation.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.validation import (
    DataValidationError,
    require_columns,
    require_no_nulls,
    require_nonempty,
)


def test_require_columns_passes_and_returns_df() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert require_columns(df, ["a", "b"], source="t.csv") is df


def test_require_columns_raises_on_missing() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="mangler forventede kolonner"):
        require_columns(df, ["a", "b"], source="t.csv")


def test_require_nonempty_raises_on_empty() -> None:
    with pytest.raises(DataValidationError, match="tomt"):
        require_nonempty(pd.DataFrame({"a": []}), source="t.csv")


def test_require_no_nulls_raises_on_null_key() -> None:
    df = pd.DataFrame({"k": [1, None], "v": [3, 4]})
    with pytest.raises(DataValidationError, match="manglende verdier"):
        require_no_nulls(df, ["k"], source="t.csv")


def test_require_no_nulls_passes_when_clean() -> None:
    df = pd.DataFrame({"k": [1, 2]})
    assert require_no_nulls(df, ["k"], source="t.csv") is df
