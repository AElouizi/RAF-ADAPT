"""
Chargement des données pour le dashboard — avec cache Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from bvc_recommender.config import (
    DATA_PROCESSED_DIR,
    DATASET_DIR,
    FEATURES_DIR,
    REBALANCE_FREQUENCY,
    RECOMMENDATIONS_DIR,
    REPORTS_DIR,
)
from bvc_recommender.models.liquidity_filter import rank_and_label_universe
from bvc_recommender.rebalance import period_display_label, regime_for_period

PORTFOLIO_LABELS = {
    "P_agressif": "Agressif",
    "P_equilibre": "Équilibré",
    "P_defensif": "Défensif",
}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(name: str, directory: Path) -> pd.DataFrame:
    path = directory / f"{name}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _quarter_id(period: str) -> str:
    """Convertit 2024-03 → 2024-Q1 ou normalise 2024-Q1."""
    p = str(period).strip()
    if "-Q" in p.upper():
        try:
            year_text, q_text = p.upper().split("-Q", 1)
            return f"{int(year_text):04d}-Q{int(q_text)}"
        except (TypeError, ValueError):
            return p
    try:
        ts = pd.Timestamp(f"{p}-01")
        return f"{ts.year}-Q{ts.quarter}"
    except Exception:
        return p


@st.cache_data(show_spinner=False)
def load_step8_report() -> dict:
    return _read_json(REPORTS_DIR / "step8_validation_report.json")


@st.cache_data(show_spinner=False)
def load_step4_report() -> dict:
    return _read_json(REPORTS_DIR / "step4_validation_report.json")


@st.cache_data(show_spinner=False)
def load_step7_report() -> dict:
    return _read_json(REPORTS_DIR / "step7_validation_report.json")


@st.cache_data(show_spinner=False)
def rebalance_frequency() -> str:
    report = load_step8_report()
    return report.get("rebalance_frequency", REBALANCE_FREQUENCY)


@st.cache_data(show_spinner=False)
def list_periods() -> list[str]:
    report = load_step8_report()
    periods = [
        p.get("period", p.get("quarter"))
        for p in report.get("periods", report.get("quarters", []))
    ]
    periods = [p for p in periods if p]
    if periods:
        monthly = [p for p in periods if "-Q" not in str(p).upper()]
        return monthly or periods
    files = sorted(RECOMMENDATIONS_DIR.glob("*.csv"))
    stems = [f.stem for f in files]
    monthly = [s for s in stems if "-Q" not in s.upper()]
    return monthly or stems


def list_quarters() -> list[str]:
    """Trimestres distincts, ordre chronologique (dérivés des périodes mensuelles)."""
    seen: list[str] = []
    for p in list_periods():
        q = _quarter_id(p)
        if q not in seen:
            seen.append(q)
    if seen:
        return seen
    files = sorted(RECOMMENDATIONS_DIR.glob("*-Q*.csv"))
    return [f.stem for f in files]


def periods_in_quarter(quarter: str) -> list[str]:
    return [p for p in list_periods() if _quarter_id(p) == quarter]


def last_period_of_quarter(quarter: str) -> str | None:
    months = periods_in_quarter(quarter)
    return months[-1] if months else None


def previous_quarter(quarter: str) -> str | None:
    qs = list_quarters()
    if quarter not in qs:
        return None
    idx = qs.index(quarter)
    return qs[idx - 1] if idx > 0 else None


@st.cache_data(show_spinner=False)
def load_recommendations(period: str) -> pd.DataFrame:
    path = RECOMMENDATIONS_DIR / f"{period}.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_tft_scores() -> pd.DataFrame:
    """Scores TFT (Q50) prioritaires pour le ranking dashboard."""
    for name in ("tft_predictions", "baseline_predictions"):
        df = _read_parquet(name, DATASET_DIR)
        if df.empty:
            continue
        if "split" in df.columns:
            holdout = df[df["split"].isin(["val", "test"])]
            if not holdout.empty:
                df = holdout
        if "q50" in df.columns:
            df = df.copy()
            df["prediction"] = pd.to_numeric(df["q50"], errors="coerce")
        elif "prediction" in df.columns:
            df = df.copy()
            df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_baseline_predictions() -> pd.DataFrame:
    return load_tft_scores()


@st.cache_data(show_spinner=False)
def load_technical_features() -> pd.DataFrame:
    return _read_parquet("features_techniques", FEATURES_DIR)


@st.cache_data(show_spinner=False)
def load_fundamental_features() -> pd.DataFrame:
    return _read_parquet("features_fondamentales", FEATURES_DIR)


@st.cache_data(show_spinner=False)
def load_cours() -> pd.DataFrame:
    return _read_parquet("market_data_cours_historique", DATA_PROCESSED_DIR)


@st.cache_data(show_spinner=False)
def load_indices() -> pd.DataFrame:
    return _read_parquet("market_data_indices_historique", DATA_PROCESSED_DIR)


@st.cache_data(show_spinner=False)
def load_ml_dataset() -> pd.DataFrame:
    return _read_parquet("ml_dataset", DATASET_DIR)


@st.cache_data(show_spinner=False)
def get_regime_for_period(period: str) -> dict:
    timeline = load_step4_report().get("timeline", [])
    return regime_for_period(period, timeline)


def get_regime_for_quarter(quarter: str) -> dict:
    last = last_period_of_quarter(quarter) or quarter
    return get_regime_for_period(last)


@st.cache_data(show_spinner=False)
def get_rebalance_date(period: str) -> pd.Timestamp | None:
    report = load_step8_report()
    for row in report.get("periods", report.get("quarters", [])):
        pid = row.get("period", row.get("quarter"))
        if pid == period:
            return pd.Timestamp(row["rebalance_date"])
    rec = load_recommendations(period)
    if not rec.empty and "rebalance_date" in rec.columns:
        return pd.Timestamp(rec["rebalance_date"].iloc[0])
    return None


@st.cache_data(show_spinner=False)
def get_ranked_universe(period: str) -> pd.DataFrame:
    as_of = get_rebalance_date(period)
    if as_of is None:
        return pd.DataFrame()

    scores = load_tft_scores()
    if scores.empty:
        return pd.DataFrame()

    technical = load_technical_features()
    return rank_and_label_universe(scores, technical, as_of_date=as_of)


@st.cache_data(show_spinner=False)
def top_tickers_for_quarter(quarter: str) -> set[str]:
    """Ensemble TOP à la fin du trimestre (dernier mois disponible)."""
    last = last_period_of_quarter(quarter)
    if last is None:
        # Fichier agrégé YYYY-QX
        rec = load_recommendations(quarter)
        if rec.empty:
            return set()
        # Dernière date de rebalance du fichier
        if "rebalance_date" in rec.columns:
            last_date = pd.to_datetime(rec["rebalance_date"], errors="coerce").max()
            rec = rec[pd.to_datetime(rec["rebalance_date"], errors="coerce") == last_date]
        return set(rec["ticker"].unique()) if "ticker" in rec.columns else set()

    ranked = get_ranked_universe(last)
    if ranked.empty:
        return set()
    return set(ranked.loc[ranked["label"] == "TOP", "ticker"])


@st.cache_data(show_spinner=False)
def get_period_metrics(portfolio: str) -> dict:
    return load_step8_report().get("metrics", {}).get(portfolio, {})


def get_quarter_metrics(portfolio: str) -> dict:
    return get_period_metrics(portfolio)


def period_label(period_id: str) -> str:
    return period_display_label(period_id)


def quarter_label(quarter: str) -> str:
    return period_display_label(quarter)
