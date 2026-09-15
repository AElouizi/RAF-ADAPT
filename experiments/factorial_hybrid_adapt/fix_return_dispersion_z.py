"""
Correction de return_dispersion_z.

Cause : return_dispersion est une feature de marché (identique pour tous les
tickers à une date) → z-score cross-sectionnel a std=0 → NaN partout.

Correctif : z-score **temporel expanding** sur la série journalière unique
(stats ≤ date t uniquement — pas de fuite future).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_COL = "return_dispersion"
Z_COL = "return_dispersion_z"
MIN_PERIODS = 60  # ~3 mois de séances


def expanding_time_zscore_market_feature(
    df: pd.DataFrame,
    *,
    raw_col: str = RAW_COL,
    z_col: str = Z_COL,
    date_col: str = "date_cours",
    min_periods: int = MIN_PERIODS,
) -> pd.DataFrame:
    """
    Recalcule ``z_col`` via expanding z-score de la série marché (1 valeur/date).
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if raw_col not in out.columns:
        raise ValueError(f"Colonne absente : {raw_col}")

    daily = (
        out.groupby(date_col, sort=True)[raw_col]
        .first()
        .astype(float)
        .sort_index()
    )
    mu = daily.expanding(min_periods=min_periods).mean()
    sig = daily.expanding(min_periods=min_periods).std(ddof=0).replace(0, np.nan)
    z_daily = (daily - mu) / sig

    out = out.drop(columns=[z_col], errors="ignore")
    out = out.merge(
        z_daily.rename(z_col).reset_index(),
        on=date_col,
        how="left",
    )
    n_ok = int(out[z_col].notna().sum())
    logger.info(
        "Corrigé %s : non-null=%s/%s (%.1f%%) | expanding min_periods=%s",
        z_col,
        n_ok,
        len(out),
        100 * n_ok / max(len(out), 1),
        min_periods,
    )
    return out


def build_dispersion_fixed_dataset(ml: pd.DataFrame) -> pd.DataFrame:
    """Retourne une copie du ml_dataset avec return_dispersion_z réparé uniquement."""
    fixed = expanding_time_zscore_market_feature(ml)
    # Ne touche pas masi_mom_3m_z / breadth / vol_universe (ablation isolée)
    return fixed
