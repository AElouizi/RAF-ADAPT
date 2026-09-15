"""
Métriques de benchmarking — période de test 2023–2025.

Inclut Omega Ratio et Hit Ratio mensuel (spécification article).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bvc_recommender.benchmarking.config import HIT_RATIO_FREQ, RISK_FREE_RATE

TRADING_DAYS_YEAR = 252


def compute_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    name: str = "portfolio",
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, Any]:
    """
    Calcule les métriques de performance / risque vs benchmark (MASI).

    Parameters
    ----------
    portfolio_returns : rendements journaliers nets de frais
    benchmark_returns : rendements journaliers MASI (alignés)
    """
    pr = pd.to_numeric(portfolio_returns, errors="coerce").dropna()
    br = pd.to_numeric(benchmark_returns, errors="coerce").reindex(pr.index).fillna(0.0)

    if pr.empty:
        return {"name": name, "error": "série vide"}

    cum = (1 + pr).cumprod()
    n_days = len(pr)
    ann_factor = TRADING_DAYS_YEAR / max(n_days, 1)

    total_ret = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total_ret) ** ann_factor - 1)

    excess_vs_bench = pr - br
    alpha_ann = float(excess_vs_bench.mean() * TRADING_DAYS_YEAR)

    rf_daily = risk_free_rate / TRADING_DAYS_YEAR
    excess_rf = pr - rf_daily
    vol = float(pr.std() * np.sqrt(TRADING_DAYS_YEAR))
    sharpe = (
        float(excess_rf.mean() / pr.std() * np.sqrt(TRADING_DAYS_YEAR))
        if pr.std() > 0
        else np.nan
    )

    downside = pr.clip(upper=rf_daily) - rf_daily
    down_std = float(downside.std() * np.sqrt(TRADING_DAYS_YEAR)) if downside.std() > 0 else np.nan
    sortino = (
        float(excess_rf.mean() / downside.std() * np.sqrt(TRADING_DAYS_YEAR))
        if downside.std() > 0
        else np.nan
    )

    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    max_dd = float(drawdown.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else np.nan

    cvar = float(-pr[pr <= pr.quantile(0.05)].mean()) if len(pr) > 5 else np.nan

    # Hit ratio mensuel : % de mois avec rendement > 0
    monthly = (1 + pr).resample(HIT_RATIO_FREQ).prod() - 1
    hit_ratio = float((monthly > 0).mean()) if len(monthly) else np.nan

    # Omega (seuil = rf quotidien) : E[max(R−τ,0)] / E[max(τ−R,0)]
    gains = (pr - rf_daily).clip(lower=0.0).sum()
    losses = (rf_daily - pr).clip(lower=0.0).sum()
    omega = float(gains / losses) if losses > 0 else np.nan

    return {
        "name": name,
        "days": n_days,
        "n_months": int(len(monthly)),
        "total_return": round(total_ret, 4),
        "annualized_return": round(ann_ret, 4),
        "alpha_annualized": round(alpha_ann, 4),
        "sharpe": round(sharpe, 3) if not np.isnan(sharpe) else None,
        "sortino": round(sortino, 3) if not np.isnan(sortino) else None,
        "cvar_95": round(cvar, 4) if not np.isnan(cvar) else None,
        "max_drawdown": round(max_dd, 4),
        "hit_ratio": round(hit_ratio, 4) if not np.isnan(hit_ratio) else None,
        "calmar": round(calmar, 3) if not np.isnan(calmar) else None,
        "omega": round(omega, 3) if not np.isnan(omega) else None,
        "volatility_ann": round(vol, 4),
    }
