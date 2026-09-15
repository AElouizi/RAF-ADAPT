"""Utilitaires partagés — jointures point-in-time et helpers numériques."""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def get_numeric(df: pd.DataFrame, *candidates: str) -> pd.Series:
    col = find_col(df, *candidates)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def series_or(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Série alignée sur l'index du panel (évite les scalaires de df.get)."""
    col = find_col(df, *candidates)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    if not isinstance(num, pd.Series):
        num = pd.Series(num, dtype=float)
    if not isinstance(denom, pd.Series):
        denom = pd.Series(denom, dtype=float)
    d = denom.replace(0, np.nan)
    return num / d


def asof_merge_by_ticker(
    left: pd.DataFrame,
    right: pd.DataFrame,
    merge_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """merge_asof par ticker (direction=backward, no look-ahead)."""
    left = left.drop(columns=[c for c in value_cols if c in left.columns], errors="ignore")
    parts: list[pd.DataFrame] = []
    right_cols = [merge_col, *value_cols]
    for ticker, left_g in left.groupby("ticker", sort=False):
        right_g = right.loc[right["ticker"] == ticker, right_cols]
        left_g = left_g.sort_values(merge_col)
        if right_g.empty:
            for col in value_cols:
                left_g[col] = np.nan
            parts.append(left_g)
            continue
        right_g = right_g.sort_values(merge_col)
        parts.append(pd.merge_asof(left_g, right_g, on=merge_col, direction="backward"))
    return pd.concat(parts, ignore_index=True) if parts else left
