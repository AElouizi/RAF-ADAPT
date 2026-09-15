"""
Métriques de performance backtest — Étape 8.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_YEAR = 252


def compute_backtest_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    name: str = "portfolio",
) -> dict[str, Any]:
    """Calcule les métriques de performance vs benchmark."""
    pr = pd.to_numeric(portfolio_returns, errors="coerce").dropna()
    br = pd.to_numeric(benchmark_returns, errors="coerce").reindex(pr.index).fillna(0.0)

    if pr.empty:
        return {"name": name, "error": "série vide"}

    cum = (1 + pr).cumprod()
    cum_bench = (1 + br).cumprod()
    n_days = len(pr)
    ann_factor = TRADING_DAYS_YEAR / max(n_days, 1)

    total_ret = float(cum.iloc[-1] - 1)
    bench_ret = float(cum_bench.iloc[-1] - 1)
    ann_ret = float((1 + total_ret) ** ann_factor - 1)
    ann_bench = float((1 + bench_ret) ** ann_factor - 1)

    excess = pr - br
    alpha_ann = float(excess.mean() * TRADING_DAYS_YEAR)
    vol = float(pr.std() * np.sqrt(TRADING_DAYS_YEAR))
    sharpe = float(pr.mean() / pr.std() * np.sqrt(TRADING_DAYS_YEAR)) if pr.std() > 0 else np.nan

    down = pr.clip(upper=0)
    down_std = float(down.std() * np.sqrt(TRADING_DAYS_YEAR))
    sortino = float(pr.mean() / down.std() * np.sqrt(TRADING_DAYS_YEAR)) if down.std() > 0 else np.nan

    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    max_dd = float(drawdown.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else np.nan

    cvar = float(-pr[pr <= pr.quantile(0.05)].mean()) if len(pr) > 5 else np.nan
    hit_ratio = float((pr > 0).mean())

    tracking_error = float(excess.std() * np.sqrt(TRADING_DAYS_YEAR))

    return {
        "name": name,
        "days": n_days,
        "total_return": round(total_ret, 4),
        "annualized_return": round(ann_ret, 4),
        "benchmark_total_return": round(bench_ret, 4),
        "benchmark_annualized_return": round(ann_bench, 4),
        "alpha_annualized": round(alpha_ann, 4),
        "sharpe": round(sharpe, 3) if not np.isnan(sharpe) else None,
        "sortino": round(sortino, 3) if not np.isnan(sortino) else None,
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 3) if not np.isnan(calmar) else None,
        "cvar_95": round(cvar, 4) if not np.isnan(cvar) else None,
        "hit_ratio": round(hit_ratio, 4),
        "volatility_ann": round(vol, 4),
        "tracking_error": round(tracking_error, 4),
    }
