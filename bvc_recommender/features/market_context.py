"""
Étape 3 — Variables d'état du marché (features_indices).

Calculées uniquement depuis les données BVC (cours + MASI), sans source externe.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bvc_recommender.features._utils import find_col
from bvc_recommender.features.technical import TRADING_DAYS_YEAR, prepare_masi_returns

logger = logging.getLogger(__name__)

# herfindahl_volume retiré : volumes indisponibles avant le 24/10/2022 (couverture 21 %)
MARKET_CONTEXT_COLUMNS = [
    "masi_mom_3m",
    "return_dispersion",
    "breadth_ma50",
    "vol_universe_mean",
]


def _prepare_daily_cours(cours: pd.DataFrame) -> pd.DataFrame:
    if cours.empty:
        return pd.DataFrame()

    c = cours.copy()
    date_col = find_col(c, "date_cours", "date") or "date_cours"
    price_col = find_col(c, "prix_cloture", "prix_courant") or "prix_cloture"
    c["date_cours"] = pd.to_datetime(c[date_col], errors="coerce")
    c["prix_cloture"] = pd.to_numeric(c[price_col], errors="coerce")
    if "volume" in c.columns:
        c["volume"] = pd.to_numeric(c["volume"], errors="coerce")
    elif "titres_echanges" in c.columns:
        c["volume"] = pd.to_numeric(c["titres_echanges"], errors="coerce")
    else:
        c["volume"] = np.nan
    return c.dropna(subset=["date_cours", "prix_cloture", "ticker"])


def _compute_masi_momentum(indices: pd.DataFrame) -> pd.DataFrame:
    if indices.empty:
        return pd.DataFrame(columns=["date_cours", "masi_mom_3m"])

    df = indices.copy()
    code_col = find_col(df, "code_index", "code") or "code_index"
    date_col = find_col(df, "date_index", "date") or "date_index"
    val_col = find_col(df, "valeur_index", "valeur") or "valeur_index"

    masi = df.loc[df[code_col].astype(str).str.upper() == "MASI"].copy()
    if masi.empty:
        return pd.DataFrame(columns=["date_cours", "masi_mom_3m"])

    masi["date_cours"] = pd.to_datetime(masi[date_col], errors="coerce")
    masi["masi_close"] = pd.to_numeric(masi[val_col], errors="coerce")
    masi = masi.dropna(subset=["date_cours", "masi_close"]).sort_values("date_cours")
    masi = masi.drop_duplicates("date_cours", keep="last")
    masi["masi_mom_3m"] = masi["masi_close"] / masi["masi_close"].shift(63) - 1
    return masi[["date_cours", "masi_mom_3m"]]


def _cross_sectional_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Dispersion, breadth MA50, vol moyenne — par date."""
    parts: list[pd.DataFrame] = []
    for ticker, grp in daily.groupby("ticker", sort=False):
        g = grp.sort_values("date_cours").copy()
        price = g["prix_cloture"]
        ret = price.pct_change(fill_method=None)
        ma50 = price.rolling(50).mean()
        vol = ret.rolling(20).std() * np.sqrt(TRADING_DAYS_YEAR)
        g["daily_ret"] = ret
        g["above_ma50"] = (price > ma50).astype(float)
        g["vol_20d"] = vol
        parts.append(g[["date_cours", "ticker", "daily_ret", "above_ma50", "vol_20d"]])

    if not parts:
        return pd.DataFrame()

    panel = pd.concat(parts, ignore_index=True)

    def _agg(group: pd.DataFrame) -> pd.Series:
        ret = group["daily_ret"].dropna()
        vol = group["vol_20d"].dropna()
        above = group["above_ma50"].dropna()
        return pd.Series(
            {
                "return_dispersion": ret.std() if len(ret) > 1 else np.nan,
                "breadth_ma50": above.mean() if len(above) else np.nan,
                "vol_universe_mean": vol.mean() if len(vol) else np.nan,
            }
        )

    agg = panel.groupby("date_cours", sort=True).apply(_agg, include_groups=False).reset_index()
    return agg


def build_market_context(
    cours: pd.DataFrame,
    indices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construit le panel journalier features_indices.

    Returns DataFrame avec date_cours et MARKET_CONTEXT_COLUMNS.
    """
    daily = _prepare_daily_cours(cours)
    if daily.empty:
        logger.warning("Cours vide — contexte marché impossible.")
        return pd.DataFrame()

    masi_mom = _compute_masi_momentum(indices if indices is not None else pd.DataFrame())
    cross = _cross_sectional_metrics(daily)

    if cross.empty:
        return pd.DataFrame()

    out = cross.merge(masi_mom, on="date_cours", how="left")
    if out["masi_mom_3m"].isna().all() and not masi_mom.empty:
        masi_ret = prepare_masi_returns(indices if indices is not None else pd.DataFrame())
        if not masi_ret.empty:
            masi_ret = masi_ret.sort_values("date_cours")
            masi_ret["masi_close"] = (1 + masi_ret["masi_ret"].fillna(0)).cumprod()
            masi_ret["masi_mom_3m"] = masi_ret["masi_close"] / masi_ret["masi_close"].shift(63) - 1
            out = out.drop(columns=["masi_mom_3m"], errors="ignore").merge(
                masi_ret[["date_cours", "masi_mom_3m"]],
                on="date_cours",
                how="left",
            )

    keep = ["date_cours", *MARKET_CONTEXT_COLUMNS]
    out = out[keep].sort_values("date_cours").reset_index(drop=True)

    logger.info(
        "Contexte marché : %s lignes, couverture %s → %s",
        len(out),
        out["date_cours"].min().date() if not out.empty else "—",
        out["date_cours"].max().date() if not out.empty else "—",
    )
    return out
