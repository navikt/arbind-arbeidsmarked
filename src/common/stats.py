"""Delte statistiske hjelpefunksjoner for analysene.

Samler implementasjoner som tidligere var duplisert på tvers av analyse-skript,
slik at rettelser og endringer gjøres ett sted (jf. CRITIQUE_METHOD_AND_CODE.md).

Inneholder:
  - ``newey_west_se``: HAC (Newey–West) standardfeil for OLS-koeffisienter.
  - ``adjust_pvalues``: korreksjon for multippel testing (BH-FDR / Holm).
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests


def newey_west_se(
    residuals: np.ndarray, X: np.ndarray, max_lag: int | None = None
) -> np.ndarray:
    """Beregn Newey–West (HAC) standardfeil for OLS-koeffisienter.

    Args:
        residuals: Residualer fra OLS-regresjonen, lengde ``n``.
        X: Designmatrise med form ``(n, k)`` inkludert konstantledd.
        max_lag: Maksimalt lag for autokorrelasjon. Standard bruker
            Newey–West-regelen ``floor(4 * (n / 100) ** (2 / 9))``.

    Returns:
        Vektor med standardfeil, én per kolonne i ``X`` (lengde ``k``).
    """
    n, k = X.shape
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    # S = Σ_0 + Σ_{l=1}^{L} w_l (Σ_l + Σ_l')
    e = np.asarray(residuals).reshape(-1, 1)
    S = np.zeros((k, k))

    for lag in range(max_lag + 1):
        w = 1.0 if lag == 0 else 1 - lag / (max_lag + 1)
        for t in range(lag, n):
            xt = X[t : t + 1].T
            xs = X[t - lag : t - lag + 1].T
            contrib = (e[t] * e[t - lag]) * (xt @ xs.T)
            if lag == 0:
                S += w * contrib
            else:
                S += w * (contrib + contrib.T)

    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(V))


def adjust_pvalues(
    pvalues: np.ndarray, method: str = "fdr_bh", alpha: float = 0.05
) -> np.ndarray:
    """Korriger p-verdier for multippel testing innen en hypotesefamilie.

    Brukes der mange nær-parallelle korrelasjoner/regresjoner kjøres på tvers av
    utfall, år og sektorer (jf. CRITIQUE_METHOD_AND_CODE.md, punkt om multippel
    testing). NaN-verdier bevares og holdes utenfor korreksjonen.

    Args:
        pvalues: Rå p-verdier for hypotesefamilien.
        method: Korreksjonsmetode som ``statsmodels.multipletests`` støtter,
            f.eks. ``"fdr_bh"`` (Benjamini–Hochberg) eller ``"holm"``.
        alpha: Familievis feilrate brukt av korreksjonen.

    Returns:
        Korrigerte p-verdier med samme form som ``pvalues``; posisjoner som var
        NaN forblir NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    finite = ~np.isnan(p)
    if finite.sum() == 0:
        return adjusted
    _, p_adj, _, _ = multipletests(p[finite], alpha=alpha, method=method)
    adjusted[finite] = p_adj
    return adjusted
