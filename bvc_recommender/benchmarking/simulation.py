"""Helpers de simulation walk-forward (mensuel) + frais de transaction."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from bvc_recommender.benchmarking.config import TEST_END, TEST_START, TRANSACTION_COST
from bvc_recommender.rebalance import rebalance_end_dates


WeightsFn = Callable[[pd.Timestamp], dict[str, float]]


def masi20_equal_weights(cours: pd.DataFrame, as_of: pd.Timestamp, n: int = 20) -> dict[str, float]:
    """Proxy MASI20 : top N capitalisation, pondération égale."""
    c = cours.copy()
    c["date_cours"] = pd.to_datetime(c["date_cours"], errors="coerce")
    snap = (
        c[c["date_cours"] <= as_of]
        .sort_values("date_cours")
        .groupby("ticker", as_index=False)
        .tail(1)
    )
    if "capitalisation" in snap.columns:
        snap["capitalisation"] = pd.to_numeric(snap["capitalisation"], errors="coerce")
        tickers = snap.nlargest(n, "capitalisation")["ticker"].tolist()
    else:
        tickers = snap["ticker"].dropna().astype(str).head(n).tolist()
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def portfolio_daily_returns(
    weights: dict[str, float],
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    tickers = [t for t in weights if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float)
    sub = prices.loc[(prices.index > start) & (prices.index <= end), tickers]
    rets = sub.pct_change(fill_method=None).dropna(how="all")
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()
    return pd.Series(rets.fillna(0.0).values @ w, index=rets.index)


def turnover_cost(old_w: dict[str, float], new_w: dict[str, float], cost: float = TRANSACTION_COST) -> float:
    tickers = set(old_w) | set(new_w)
    turnover = sum(abs(new_w.get(t, 0.0) - old_w.get(t, 0.0)) for t in tickers)
    return turnover * cost


def simulate_monthly_rebalance(
    prices: pd.DataFrame,
    weights_fn: WeightsFn,
    *,
    start: pd.Timestamp | str = TEST_START,
    end: pd.Timestamp | str = TEST_END,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.Series:
    """
    Walk-forward mensuel : à chaque fin de mois, nouveaux poids + frais sur le 1er jour.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    # Borner à l'historique prix disponible
    if not prices.empty:
        end = min(end, prices.index.max())
        start = max(start, prices.index.min())

    ends = rebalance_end_dates(prices.index, start, end, frequency="monthly")
    if len(ends) < 2:
        return pd.Series(dtype=float)

    parts: list[pd.Series] = []
    prev: dict[str, float] = {}
    for i, reb in enumerate(ends[:-1]):
        nxt = ends[i + 1]
        weights = weights_fn(reb)
        if not weights:
            continue
        daily = portfolio_daily_returns(weights, prices, reb, nxt)
        cost = turnover_cost(prev, weights, transaction_cost)
        if not daily.empty and cost > 0:
            daily = daily.copy()
            daily.iloc[0] -= cost
        if not daily.empty:
            parts.append(daily)
        prev = weights

    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()
