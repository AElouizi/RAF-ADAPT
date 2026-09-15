"""
Courbes de performance cumulées pour le dashboard.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from bvc_recommender.dashboard.data import load_cours, load_indices, load_recommendations, list_periods


def _price_panel(cours: pd.DataFrame) -> pd.DataFrame:
    c = cours.copy()
    c["date_cours"] = pd.to_datetime(c["date_cours"], errors="coerce")
    price_col = "prix_cloture" if "prix_cloture" in c.columns else "prix_courant"
    c[price_col] = pd.to_numeric(c[price_col], errors="coerce")
    return c.pivot_table(index="date_cours", columns="ticker", values=price_col, aggfunc="last").sort_index()


def _index_returns(indices: pd.DataFrame, code: str = "MASI") -> pd.Series:
    df = indices.copy()
    code_col = "code_index" if "code_index" in df.columns else "code"
    date_col = "date_index" if "date_index" in df.columns else "date"
    val_col = "valeur_index" if "valeur_index" in df.columns else "valeur"
    sub = df[df[code_col].astype(str).str.upper() == code.upper()].copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce")
    levels = (
        sub.dropna(subset=[date_col, val_col])
        .drop_duplicates(date_col, keep="last")
        .set_index(date_col)[val_col]
        .sort_index()
    )
    return levels.pct_change(fill_method=None).dropna()


def _portfolio_returns(
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
    w = [weights[t] for t in tickers]
    w_sum = sum(w)
    w = [x / w_sum for x in w]
    port = rets.fillna(0.0).values @ w
    return pd.Series(port, index=rets.index)


@st.cache_data(show_spinner="Calcul des courbes de performance…")
def compute_equity_curves() -> pd.DataFrame:
    from bvc_recommender.config import DATASET_DIR
    from bvc_recommender.dashboard.data import load_step8_report

    report = load_step8_report()
    curves_path = Path(report.get("equity_curves_path") or "")
    if not curves_path.is_file():
        curves_path = DATASET_DIR / "backtest_equity_curves.parquet"
    if curves_path.is_file():
        cum = pd.read_parquet(curves_path)
        cum.columns = [str(c).replace("P_", "").replace("_proxy", "") for c in cum.columns]
        cum.index = pd.to_datetime(cum.index)
        cum.index.name = "date"
        return cum.sort_index()

    periods = list_periods()
    # Ignorer les agrégats trimestriels YYYY-QX si présents
    periods = [p for p in periods if "-Q" not in p.upper()]
    if len(periods) < 2:
        return pd.DataFrame()

    cours = load_cours()
    indices = load_indices()
    if cours.empty:
        return pd.DataFrame()

    prices = _price_panel(cours)
    masi_rets = _index_returns(indices, "MASI")

    portfolio_keys = ["P_agressif", "P_equilibre", "P_defensif"]
    series: dict[str, list[pd.Series]] = {k: [] for k in portfolio_keys}
    series["MASI"] = []

    rebalance_dates: list[pd.Timestamp] = []
    usable_periods: list[str] = []
    for p in periods:
        rec = load_recommendations(p)
        if rec.empty:
            continue
        rebalance_dates.append(pd.Timestamp(rec["rebalance_date"].iloc[0]))
        usable_periods.append(p)

    if len(rebalance_dates) < 2:
        return pd.DataFrame()

    for i in range(len(rebalance_dates) - 1):
        start, end = rebalance_dates[i], rebalance_dates[i + 1]
        p = usable_periods[i]
        rec = load_recommendations(p)

        for key in portfolio_keys:
            sub = rec[rec["portfolio"] == key]
            weights = dict(zip(sub["ticker"], sub["weight"]))
            daily = _portfolio_returns(weights, prices, start, end)
            if not daily.empty:
                series[key].append(daily)

        masi_seg = masi_rets.loc[(masi_rets.index > start) & (masi_rets.index <= end)]
        if not masi_seg.empty:
            series["MASI"].append(masi_seg)

    curves: dict[str, pd.Series] = {}
    for key, parts in series.items():
        if parts:
            curves[key] = pd.concat(parts).sort_index()

    if not curves:
        return pd.DataFrame()

    df = pd.DataFrame(curves)
    cum = (1 + df.fillna(0)).cumprod()
    cum.columns = [c.replace("P_", "") for c in cum.columns]
    cum.index.name = "date"
    return cum
