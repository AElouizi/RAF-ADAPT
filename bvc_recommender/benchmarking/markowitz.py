"""
Optimisation Markowitz (minimum variance) — remplace NSGA-III (Composante D).

Contraintes : Σw=1, w∈[2%, 10%]. Pas d'objectif liquidité / CVaR séparé.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bvc_recommender.models.portfolio_optimizer import (
    PORTFOLIO_MAX_WEIGHT,
    PORTFOLIO_MIN_WEIGHT,
    RETURNS_LOOKBACK_DAYS,
    project_weights,
)


def _returns_matrix(cours: pd.DataFrame, tickers: list[str], as_of: pd.Timestamp) -> tuple[np.ndarray, list[str]]:
    c = cours.copy()
    c["date_cours"] = pd.to_datetime(c["date_cours"], errors="coerce")
    price_col = "prix_cloture" if "prix_cloture" in c.columns else "prix_courant"
    c[price_col] = pd.to_numeric(c[price_col], errors="coerce")
    c = c[(c["ticker"].isin(tickers)) & (c["date_cours"] <= as_of)]
    wide = c.pivot_table(index="date_cours", columns="ticker", values=price_col, aggfunc="last").sort_index()
    rets = wide.pct_change(fill_method=None).dropna(how="all")
    if len(rets) > RETURNS_LOOKBACK_DAYS:
        rets = rets.iloc[-RETURNS_LOOKBACK_DAYS:]
    common = [t for t in tickers if t in rets.columns and rets[t].notna().sum() > 40]
    if len(common) < 2:
        return np.empty((0, 0)), []
    return rets[common].fillna(0.0).values, common


def markowitz_min_variance(
    ranked: pd.DataFrame,
    cours: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    label_top: str = "TOP",
) -> dict[str, float]:
    """Poids min-variance sur l'univers TOP (ou top score_final si label absent)."""
    if "label" in ranked.columns and (ranked["label"] == label_top).any():
        tickers = ranked.loc[ranked["label"] == label_top, "ticker"].tolist()
    else:
        tickers = ranked.nlargest(min(20, len(ranked)), "score_final")["ticker"].tolist()

    ret_mat, tickers = _returns_matrix(cours, tickers, as_of)
    n = len(tickers)
    if n < 2:
        return {}

    w_min = PORTFOLIO_MIN_WEIGHT
    if n * w_min > 1.0:
        w_min = 1.0 / n
    cov = np.cov(ret_mat, rowvar=False)
    # Régularisation numérique
    cov = cov + np.eye(n) * 1e-8

    def obj(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    w0 = np.full(n, 1.0 / n)
    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(w_min, PORTFOLIO_MAX_WEIGHT)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 400, "ftol": 1e-12},
    )
    w = project_weights(res.x if res.success else w0, w_min=w_min, w_max=PORTFOLIO_MAX_WEIGHT)
    return {t: float(wi) for t, wi in zip(tickers, w)}
