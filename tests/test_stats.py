"""Tester for delte statistiske hjelpefunksjoner i src/common/stats.py."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from src.common.stats import adjust_pvalues, newey_west_se


def _ols_fit(seed: int = 0, n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y = X @ np.array([1.0, 2.0]) + rng.normal(size=n)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return resid, X, y


def test_newey_west_zero_lag_equals_hc0_sandwich() -> None:
    """Med max_lag=0 skal Newey–West tilsvare White (HC0) sandwich-SE."""
    resid, X, _ = _ols_fit()
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = X.T @ np.diag(resid**2) @ X
    hc0 = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))

    nw = newey_west_se(resid, X, max_lag=0)
    np.testing.assert_allclose(nw, hc0, rtol=1e-10, atol=1e-12)


def test_newey_west_matches_statsmodels_hac() -> None:
    """Newey–West-SE skal være nær statsmodels HAC for samme antall lags."""
    resid, X, y = _ols_fit()
    n = X.shape[0]
    max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    nw = newey_west_se(resid, X)
    res = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": max_lag, "use_correction": False}
    )
    np.testing.assert_allclose(nw, res.bse, rtol=1e-6, atol=1e-8)


def test_newey_west_returns_one_se_per_column() -> None:
    resid, X, _ = _ols_fit()
    se = newey_west_se(resid, X)
    assert se.shape == (X.shape[1],)
    assert np.all(se > 0)


def test_adjust_pvalues_matches_multipletests() -> None:
    p = np.array([0.001, 0.04, 0.2, 0.5])
    expected = multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(adjust_pvalues(p), expected)


def test_adjust_pvalues_preserves_nan_and_excludes_from_correction() -> None:
    p = np.array([0.01, np.nan, 0.04])
    adjusted = adjust_pvalues(p)
    assert np.isnan(adjusted[1])

    # Korreksjonen skal kun bruke de endelige verdiene.
    finite_expected = multipletests(np.array([0.01, 0.04]), method="fdr_bh")[1]
    np.testing.assert_allclose(adjusted[[0, 2]], finite_expected)


def test_adjust_pvalues_all_nan() -> None:
    adjusted = adjust_pvalues(np.array([np.nan, np.nan]))
    assert np.all(np.isnan(adjusted))


def test_adjust_pvalues_holm_method() -> None:
    p = np.array([0.01, 0.02, 0.03])
    expected = multipletests(p, method="holm")[1]
    np.testing.assert_allclose(adjust_pvalues(p, method="holm"), expected)
