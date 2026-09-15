"""
Étape 3 — Assemblage du dataset ML final.

Joint features fondamentales (point-in-time), techniques, contexte marché,
calcule la cible alpha_ajuste_risque et applique la normalisation cross-sectionnelle.

Traçabilité : version post-correction des splits d'actions BVC (2026-07-28).
Les cours source sont ceux de market_data_cours_historique corrigés ; ne pas
réutiliser un ml_dataset.parquet antérieur à cette date.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from bvc_recommender.config import (
    DATASET_DIR,
    FORWARD_HORIZON_DAYS,
    RANDOM_STATE,
    SPLIT_TEST_START,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
)
from bvc_recommender.features._utils import asof_merge_by_ticker, find_col
from bvc_recommender.features.fundamental import FUNDAMENTAL_FEATURE_COLUMNS
from bvc_recommender.features.market_context import MARKET_CONTEXT_COLUMNS
from bvc_recommender.features.technical import TECHNICAL_FEATURE_COLUMNS, prepare_masi_returns

logger = logging.getLogger(__name__)

# Provenance méthodologie article — post-correction splits (2026-07-28)
DATA_VERSION_NOTE = "post_split_correction_2026-07-28"

TARGET_COLUMN = "alpha_ajuste_risque"
# Variante plus stable pour l'apprentissage (sans division par vol baissière).
EXCESS_TARGET_COLUMN = "excess_vs_masi"
# Percentile rank mensuel de ``alpha_ajuste_risque`` (cible TFT alternative).
RANK_TARGET_COLUMN = "rang_cross_sectionnel"
META_COLUMNS = ["ticker", "date_cours", "publication_date", "sector_type", "split"]


def add_cross_sectional_rank_target(
    df: pd.DataFrame,
    *,
    source_col: str = TARGET_COLUMN,
    out_col: str = RANK_TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Rang cross-sectionnel mensuel ∈ (0, 1] de ``source_col``.

    Pour chaque mois calendaire m, le percentile est calculé uniquement parmi
    les observations disponibles ce mois (pas sur tout l'historique, pas de
    mois futurs). Source par défaut : ``alpha_ajuste_risque``.
    """
    if df.empty:
        out = df.copy()
        out[out_col] = np.nan
        return out

    out = df.copy()
    dates = pd.to_datetime(out["date_cours"], errors="coerce")
    ym = dates.dt.to_period("M")
    src = pd.to_numeric(out[source_col], errors="coerce")
    # rank(pct=True) ignore les NaN et ne voit que le groupe du mois courant
    out[out_col] = src.groupby(ym, sort=False).rank(method="average", pct=True)
    return out


def _prepare_masi_prices(indices: pd.DataFrame) -> pd.DataFrame:
    if indices.empty:
        return pd.DataFrame(columns=["date_cours", "masi_close"])

    df = indices.copy()
    code_col = find_col(df, "code_index", "code") or "code_index"
    date_col = find_col(df, "date_index", "date") or "date_index"
    val_col = find_col(df, "valeur_index", "valeur") or "valeur_index"

    masi = df.loc[df[code_col].astype(str).str.upper() == "MASI"].copy()
    masi["date_cours"] = pd.to_datetime(masi[date_col], errors="coerce")
    masi["masi_close"] = pd.to_numeric(masi[val_col], errors="coerce")
    masi = masi.dropna(subset=["date_cours", "masi_close"]).sort_values("date_cours")
    return masi.drop_duplicates("date_cours", keep="last")[["date_cours", "masi_close"]]


def attach_fundamentals_point_in_time(
    technical: pd.DataFrame,
    fundamental: pd.DataFrame,
) -> pd.DataFrame:
    """Joint fondamentaux via publication_date (backward, no look-ahead)."""
    if technical.empty or fundamental.empty:
        return technical

    tech = technical.copy()
    tech["date_cours"] = pd.to_datetime(tech["date_cours"], errors="coerce")

    fond = fundamental.copy()
    if "publication_date" not in fond.columns:
        fond["publication_date"] = pd.to_datetime(fond.get("date_fin"), errors="coerce")
    fond["publication_date"] = pd.to_datetime(fond["publication_date"], errors="coerce")
    fond = fond.dropna(subset=["ticker", "publication_date"])

    value_cols = [c for c in FUNDAMENTAL_FEATURE_COLUMNS if c in fond.columns]
    meta_cols = ["ticker", "publication_date", "sector_type"]
    keep = [c for c in meta_cols + value_cols if c in fond.columns]
    fond = fond[keep].sort_values(["ticker", "publication_date"])

    tech["_merge_fond"] = tech["date_cours"]
    fond_panel = fond.rename(columns={"publication_date": "_merge_fond"})
    merged = asof_merge_by_ticker(tech, fond_panel, "_merge_fond", value_cols + ["sector_type"])
    return merged.drop(columns=["_merge_fond"], errors="ignore")


def compute_target(
    panel: pd.DataFrame,
    indices: pd.DataFrame,
    *,
    horizon_days: int = FORWARD_HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Cibles forward sur ``horizon_days`` séances (features à t, cible à t+H) :

    - ``excess_vs_masi`` = r_action - r_MASI
    - ``alpha_ajuste_risque`` = excess_vs_masi / vol_baissiere_20d
    """
    if panel.empty:
        return panel

    out = panel.copy()
    masi = _prepare_masi_prices(indices)
    if masi.empty:
        out[TARGET_COLUMN] = np.nan
        out[EXCESS_TARGET_COLUMN] = np.nan
        return out

    masi = masi.sort_values("date_cours")
    masi["masi_forward_ret"] = masi["masi_close"].shift(-horizon_days) / masi["masi_close"] - 1
    out = out.merge(masi[["date_cours", "masi_forward_ret"]], on="date_cours", how="left")

    parts: list[pd.DataFrame] = []
    price_col = "prix_cloture" if "prix_cloture" in out.columns else None
    for ticker, grp in out.groupby("ticker", sort=False):
        g = grp.sort_values("date_cours").copy()
        if price_col and g[price_col].notna().any():
            price = pd.to_numeric(g[price_col], errors="coerce")
        else:
            price = pd.Series(np.nan, index=g.index)
        g["forward_ret"] = price.shift(-horizon_days) / price - 1
        excess = g["forward_ret"] - g["masi_forward_ret"]
        g[EXCESS_TARGET_COLUMN] = excess
        down_vol = pd.to_numeric(g.get("vol_baissiere_20d"), errors="coerce").replace(0, np.nan)
        g[TARGET_COLUMN] = excess / down_vol
        parts.append(g)

    result = pd.concat(parts, ignore_index=True) if parts else out
    return result.drop(columns=["forward_ret", "masi_forward_ret"], errors="ignore")


def cross_sectional_zscore(
    df: pd.DataFrame,
    columns: Iterable[str],
    date_col: str = "date_cours",
) -> pd.DataFrame:
    """Z-score cross-sectionnel par date (pas par ticker)."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        zcol = f"{col}_z"
        grouped = out.groupby(date_col)[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        out[zcol] = (out[col] - mean) / std
    return out


def assign_temporal_split(df: pd.DataFrame, date_col: str = "date_cours") -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    train_end = pd.Timestamp(SPLIT_TRAIN_END)
    val_end = pd.Timestamp(SPLIT_VAL_END)
    test_start = pd.Timestamp(SPLIT_TEST_START)

    out["split"] = "exclude"
    out.loc[dates <= train_end, "split"] = "train"
    out.loc[(dates > train_end) & (dates <= val_end), "split"] = "val"
    out.loc[dates >= test_start, "split"] = "test"
    return out


def build_ml_dataset(
    technical: pd.DataFrame,
    fundamental: pd.DataFrame,
    market_context: pd.DataFrame,
    cours: pd.DataFrame,
    indices: pd.DataFrame,
    *,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    normalize: bool = True,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Pipeline complet : jointure PIT + cible + normalisation + split temporel.
    """
    if technical.empty:
        logger.warning("Features techniques vides — dataset ML impossible.")
        return pd.DataFrame()

    panel = technical.copy()
    panel["date_cours"] = pd.to_datetime(panel["date_cours"], errors="coerce")
    if tickers:
        panel = panel[panel["ticker"].isin(tickers)].copy()

    if not cours.empty:
        c = cours.copy()
        date_col = find_col(c, "date_cours", "date") or "date_cours"
        price_col = find_col(c, "prix_cloture", "prix_courant") or "prix_cloture"
        c["date_cours"] = pd.to_datetime(c[date_col], errors="coerce")
        c["prix_cloture"] = pd.to_numeric(c[price_col], errors="coerce")
        prices = c[["ticker", "date_cours", "prix_cloture"]].drop_duplicates(
            ["ticker", "date_cours"], keep="last"
        )
        panel = panel.merge(prices, on=["ticker", "date_cours"], how="left")

    panel = attach_fundamentals_point_in_time(panel, fundamental)

    if not market_context.empty:
        ctx = market_context.copy()
        ctx["date_cours"] = pd.to_datetime(ctx["date_cours"], errors="coerce")
        panel = panel.merge(ctx, on="date_cours", how="left")

    panel = compute_target(panel, indices, horizon_days=horizon_days)
    panel = assign_temporal_split(panel)

    feature_cols = [
        c
        for c in (
            list(TECHNICAL_FEATURE_COLUMNS)
            + list(FUNDAMENTAL_FEATURE_COLUMNS)
            + list(MARKET_CONTEXT_COLUMNS)
        )
        if c in panel.columns
    ]

    if normalize and feature_cols:
        panel = cross_sectional_zscore(panel, feature_cols)

    z_cols = [f"{c}_z" for c in feature_cols if f"{c}_z" in panel.columns]
    keep = [c for c in META_COLUMNS if c in panel.columns]
    keep += feature_cols + z_cols + [TARGET_COLUMN, EXCESS_TARGET_COLUMN]
    keep = list(dict.fromkeys(keep))
    out = panel[keep].copy()
    out = out.dropna(subset=[TARGET_COLUMN, EXCESS_TARGET_COLUMN])

    logger.info(
        "Dataset ML : %s lignes, %s tickers, split train=%s val=%s test=%s",
        len(out),
        out["ticker"].nunique() if "ticker" in out.columns else 0,
        int((out["split"] == "train").sum()) if "split" in out.columns else 0,
        int((out["split"] == "val").sum()) if "split" in out.columns else 0,
        int((out["split"] == "test").sum()) if "split" in out.columns else 0,
    )
    return out.sort_values(["ticker", "date_cours"]).reset_index(drop=True)


def save_ml_dataset(df: pd.DataFrame, name: str = "ml_dataset") -> Path:
    """Sauvegarde locale du dataset ML."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path = DATASET_DIR / f"{name}.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception:
        path = DATASET_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
    logger.info("Dataset ML sauvegardé : %s (%s lignes)", path, len(df))
    return path
