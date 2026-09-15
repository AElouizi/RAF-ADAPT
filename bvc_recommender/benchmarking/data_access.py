"""Chargement des données pour le benchmarking (même sources que le backtest)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bvc_recommender.config import DATA_PROCESSED_DIR, DATASET_DIR, FEATURES_DIR


def _read_parquet(name: str, directory: Path) -> pd.DataFrame:
    path = directory / f"{name}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_cours() -> pd.DataFrame:
    return _read_parquet("market_data_cours_historique", DATA_PROCESSED_DIR)


def load_indices() -> pd.DataFrame:
    return _read_parquet("market_data_indices_historique", DATA_PROCESSED_DIR)


def load_technical() -> pd.DataFrame:
    return _read_parquet("features_techniques", FEATURES_DIR)


def load_tft_scores() -> pd.DataFrame:
    """Scores TFT Q50 (val+test) — Composantes B/C pour Top-5 / RAF-ADAPT."""
    for name in ("tft_predictions", "baseline_predictions"):
        df = _read_parquet(name, DATASET_DIR)
        if df.empty:
            continue
        if "split" in df.columns:
            holdout = df[df["split"].isin(["val", "test"])]
            if not holdout.empty:
                df = holdout
        df = df.copy()
        if "q50" in df.columns:
            df["prediction"] = pd.to_numeric(df["q50"], errors="coerce")
        elif "prediction" in df.columns:
            df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
        return df
    return pd.DataFrame()


def price_panel(cours: pd.DataFrame) -> pd.DataFrame:
    c = cours.copy()
    c["date_cours"] = pd.to_datetime(c["date_cours"], errors="coerce")
    price_col = "prix_cloture" if "prix_cloture" in c.columns else "prix_courant"
    c[price_col] = pd.to_numeric(c[price_col], errors="coerce")
    return (
        c.pivot_table(index="date_cours", columns="ticker", values=price_col, aggfunc="last")
        .sort_index()
    )


def index_returns(indices: pd.DataFrame, code: str = "MASI") -> pd.Series:
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
