"""
Étape 2 — Features techniques journalières.

Source : market_data_cours_historique
Benchmark beta : MASI (market_data_indices_historique)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bvc_recommender.config import LIQUIDITY_VMQ_THRESHOLD_MAD
from bvc_recommender.features._utils import find_col, safe_div
from bvc_recommender.features.fundamental import filter_from_listing_date, filter_from_min_date

logger = logging.getLogger(__name__)

TRADING_DAYS_YEAR = 252

TECHNICAL_FEATURE_COLUMNS = [
    "ret_1m",
    "ret_3m",
    "ret_6m",
    "ret_12m",
    "mom_12_1",
    "rsi_14",
    "dist_ma20",
    "dist_ma50",
    "dist_ma200",
    "vol_20d",
    "vol_60d",
    "vol_baissiere_20d",
    "beta_masi",
    "vmq_20j",
    "vmq_60j",
    "turnover_ratio",
    "indicateur_liquidite",
]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def prepare_masi_returns(indices: pd.DataFrame) -> pd.DataFrame:
    """Série journalière des rendements MASI."""
    if indices.empty:
        return pd.DataFrame(columns=["date_cours", "masi_ret"])

    df = indices.copy()
    code_col = find_col(df, "code_index", "code") or "code_index"
    date_col = find_col(df, "date_index", "date") or "date_index"
    val_col = find_col(df, "valeur_index", "valeur") or "valeur_index"

    masi = df.loc[df[code_col] == "MASI"].copy()
    if masi.empty:
        logger.warning("Aucune ligne MASI dans indices_historique.")
        return pd.DataFrame(columns=["date_cours", "masi_ret"])

    masi["date_cours"] = pd.to_datetime(masi[date_col], errors="coerce")
    masi["masi_close"] = pd.to_numeric(masi[val_col], errors="coerce")
    masi = masi.dropna(subset=["date_cours", "masi_close"]).sort_values("date_cours")
    masi = masi.drop_duplicates("date_cours", keep="last")
    masi["masi_ret"] = masi["masi_close"].pct_change(fill_method=None)
    return masi[["date_cours", "masi_ret"]]


def compute_ticker_technical(
    daily: pd.DataFrame,
    masi_returns: pd.DataFrame,
    shares_outstanding: float | None = None,
) -> pd.DataFrame:
    """Indicateurs techniques journaliers pour un ticker."""
    g = daily.sort_values("date_cours").copy()
    price = g["prix_cloture"]
    ret = price.pct_change(fill_method=None)

    vol_col = g["volume"] if "volume" in g.columns else pd.Series(0.0, index=g.index)
    titres = g["titres_echanges"] if "titres_echanges" in g.columns else vol_col
    vol = pd.to_numeric(titres, errors="coerce").fillna(0)

    g["ret_1m"] = price / price.shift(21) - 1
    g["ret_3m"] = price / price.shift(63) - 1
    g["ret_6m"] = price / price.shift(126) - 1
    g["ret_12m"] = price / price.shift(252) - 1
    g["mom_12_1"] = g["ret_12m"] - g["ret_1m"]
    g["rsi_14"] = _rsi(price, 14)

    ma20 = price.rolling(20).mean()
    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()
    g["dist_ma20"] = price / ma20 - 1
    g["dist_ma50"] = price / ma50 - 1
    g["dist_ma200"] = price / ma200 - 1

    g["vol_20d"] = ret.rolling(20).std() * np.sqrt(TRADING_DAYS_YEAR)
    g["vol_60d"] = ret.rolling(60).std() * np.sqrt(TRADING_DAYS_YEAR)
    down = ret.clip(upper=0)
    g["vol_baissiere_20d"] = np.sqrt((down**2).rolling(20).mean()) * np.sqrt(TRADING_DAYS_YEAR)

    g["vmq_20j"] = (price * vol).rolling(20).mean()
    g["vmq_60j"] = (price * vol).rolling(60).mean()

    if shares_outstanding and shares_outstanding > 0:
        g["turnover_ratio"] = vol.rolling(20).mean() / shares_outstanding
    else:
        cap = pd.to_numeric(g.get("capitalisation"), errors="coerce")
        g["turnover_ratio"] = safe_div(vol.rolling(20).mean() * price, cap)

    g["indicateur_liquidite"] = (g["vmq_20j"] >= LIQUIDITY_VMQ_THRESHOLD_MAD).astype(float)

    if not masi_returns.empty:
        g = g.merge(masi_returns, on="date_cours", how="left")
        stock_ret = g["prix_cloture"].pct_change(fill_method=None)
        cov = stock_ret.rolling(60).cov(g["masi_ret"])
        var = g["masi_ret"].rolling(60).var()
        g["beta_masi"] = cov / var.replace(0, np.nan)
        g = g.drop(columns=["masi_ret"], errors="ignore")
    else:
        g["beta_masi"] = np.nan

    return g


def build_technical_features(
    cours: pd.DataFrame,
    indices: pd.DataFrame | None = None,
    entreprises: pd.DataFrame | None = None,
    *,
    tickers: list[str] | None = None,
    listing_dates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calcule les features techniques journalières pour l'univers cours.

    Returns DataFrame avec ticker, date_cours et TECHNICAL_FEATURE_COLUMNS.
    """
    if cours.empty:
        logger.warning("Cours vide — features techniques impossibles.")
        return pd.DataFrame()

    c = cours.copy()
    date_col = find_col(c, "date_cours", "date") or "date_cours"
    c["date_cours"] = pd.to_datetime(c[date_col], errors="coerce")
    price_col = find_col(c, "prix_cloture", "prix_courant") or "prix_cloture"
    c["prix_cloture"] = pd.to_numeric(c[price_col], errors="coerce")

    for col in ("volume", "titres_echanges", "capitalisation"):
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce")

    if tickers:
        c = c[c["ticker"].isin(tickers)].copy()

    if listing_dates is not None and not listing_dates.empty:
        c = filter_from_listing_date(c, listing_dates, "date_cours", label="jours")
    c = filter_from_min_date(c, "date_cours", label="jours")

    shares_map: dict[str, float] = {}
    if entreprises is not None and not entreprises.empty:
        ent = entreprises.copy()
        tcol = find_col(ent, "ticker") or "ticker"
        if "nombre_actions" in ent.columns:
            ent["nombre_actions"] = pd.to_numeric(ent["nombre_actions"], errors="coerce")
            shares_map = (
                ent.dropna(subset=[tcol, "nombre_actions"])
                .drop_duplicates(tcol)
                .set_index(tcol)["nombre_actions"]
                .to_dict()
            )

    masi_returns = prepare_masi_returns(indices if indices is not None else pd.DataFrame())

    parts: list[pd.DataFrame] = []
    for ticker, grp in c.groupby("ticker", sort=False):
        if grp["prix_cloture"].notna().sum() < 30:
            continue
        feat = compute_ticker_technical(
            grp,
            masi_returns,
            shares_outstanding=shares_map.get(ticker),
        )
        parts.append(feat)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    keep = ["ticker", "date_cours", *TECHNICAL_FEATURE_COLUMNS]
    keep = [x for x in keep if x in out.columns]
    out = out[keep].sort_values(["ticker", "date_cours"]).reset_index(drop=True)

    logger.info(
        "Features techniques : %s lignes, %s tickers",
        len(out),
        out["ticker"].nunique(),
    )
    return out
