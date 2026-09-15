"""
Métriques de validation des modèles de scoring — Étape 5.

IC (Information Coefficient), Hit Ratio, Sharpe long-short.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_YEAR = 252


def _cross_sectional_ic(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    date_col: str,
    method: str = "pearson",
) -> list[float]:
    ics: list[float] = []
    for _, grp in df.groupby(date_col, sort=True):
        sub = grp[[pred_col, actual_col]].dropna()
        if len(sub) < 5:
            continue
        if sub[pred_col].std() == 0 or sub[actual_col].std() == 0:
            continue
        ic = sub[pred_col].corr(sub[actual_col], method=method)
        if pd.notna(ic):
            ics.append(float(ic))
    return ics


def information_coefficient(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    *,
    date_col: str = "date_cours",
) -> dict[str, float]:
    """IC moyen (Pearson) et IC médian cross-sectionnel par date."""
    ics = _cross_sectional_ic(df, pred_col, actual_col, date_col, method="pearson")
    rank_ics = _cross_sectional_ic(df, pred_col, actual_col, date_col, method="spearman")
    return {
        "ic_mean": float(np.mean(ics)) if ics else np.nan,
        "ic_median": float(np.median(ics)) if ics else np.nan,
        "ic_std": float(np.std(ics)) if ics else np.nan,
        "rank_ic_mean": float(np.mean(rank_ics)) if rank_ics else np.nan,
        "n_dates": len(ics),
    }


def hit_ratio(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    *,
    date_col: str = "date_cours",
) -> dict[str, float]:
    """Part des prédictions dont le signe correspond à la cible."""
    sub = df[[pred_col, actual_col, date_col]].dropna()
    if sub.empty:
        return {"hit_ratio": np.nan, "n_obs": 0}

    hits = (np.sign(sub[pred_col]) == np.sign(sub[actual_col])).astype(float)
    hits = hits[(sub[pred_col] != 0) & (sub[actual_col] != 0)]

    by_date: list[float] = []
    for _, grp in sub.groupby(date_col, sort=True):
        g = grp[(grp[pred_col] != 0) & (grp[actual_col] != 0)]
        if len(g) < 5:
            continue
        by_date.append(float((np.sign(g[pred_col]) == np.sign(g[actual_col])).mean()))

    return {
        "hit_ratio": float(hits.mean()) if len(hits) else np.nan,
        "hit_ratio_cs_mean": float(np.mean(by_date)) if by_date else np.nan,
        "n_obs": int(len(hits)),
    }


def sharpe_long_short(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    *,
    date_col: str = "date_cours",
    quintile: float = 0.2,
    rebalance: str = "monthly",
) -> dict[str, float]:
    """
    Sharpe d'une stratégie long-short (top vs bottom quintile prédit).

    Rebalancement mensuel par défaut (dernier jour de bourse du mois).
    """
    sub = df[[pred_col, actual_col, date_col]].dropna().copy()
    if sub.empty:
        return {"sharpe": np.nan, "mean_spread": np.nan, "n_periods": 0}

    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    if rebalance == "monthly":
        sub["period"] = sub[date_col].dt.to_period("M")
        periods = sub.groupby("period")[date_col].max().reset_index()
        snaps = []
        for _, row in periods.iterrows():
            snap = sub[sub[date_col] == row[date_col]]
            snaps.append(snap)
    else:
        snaps = [grp for _, grp in sub.groupby(date_col, sort=True)]

    spreads: list[float] = []
    for snap in snaps:
        if len(snap) < 10:
            continue
        k = max(1, int(len(snap) * quintile))
        top = snap.nlargest(k, pred_col)[actual_col].mean()
        bottom = snap.nsmallest(k, pred_col)[actual_col].mean()
        if pd.notna(top) and pd.notna(bottom):
            spreads.append(float(top - bottom))

    if len(spreads) < 2:
        return {"sharpe": np.nan, "mean_spread": np.nan, "n_periods": len(spreads)}

    arr = np.array(spreads)
    std = arr.std()
    periods_per_year = 12 if rebalance == "monthly" else TRADING_DAYS_YEAR
    sharpe = float(arr.mean() / std * np.sqrt(periods_per_year)) if std > 0 else np.nan
    return {
        "sharpe": sharpe,
        "mean_spread": float(arr.mean()),
        "n_periods": len(spreads),
    }


def evaluate_scoring_model(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    *,
    date_col: str = "date_cours",
) -> dict[str, Any]:
    """Agrège IC, Hit Ratio et Sharpe pour un jeu de prédictions."""
    ic = information_coefficient(df, pred_col, actual_col, date_col=date_col)
    hit = hit_ratio(df, pred_col, actual_col, date_col=date_col)
    sharpe = sharpe_long_short(df, pred_col, actual_col, date_col=date_col)
    return {
        "ic_mean": ic["ic_mean"],
        "ic_median": ic["ic_median"],
        "rank_ic_mean": ic["rank_ic_mean"],
        "ic_n_dates": ic["n_dates"],
        "hit_ratio": hit["hit_ratio"],
        "hit_ratio_cs_mean": hit["hit_ratio_cs_mean"],
        "sharpe_long_short": sharpe["sharpe"],
        "mean_ls_spread": sharpe["mean_spread"],
        "ls_n_periods": sharpe["n_periods"],
    }
