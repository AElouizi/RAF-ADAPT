"""
Utilitaires de rebalancement (mensuel ou trimestriel).
"""

from __future__ import annotations

import pandas as pd

from bvc_recommender.config import REBALANCE_FREQUENCY

MONTH_NAMES_FR = {
    1: "Janvier",
    2: "Février",
    3: "Mars",
    4: "Avril",
    5: "Mai",
    6: "Juin",
    7: "Juillet",
    8: "Août",
    9: "Septembre",
    10: "Octobre",
    11: "Novembre",
    12: "Décembre",
}


def rebalance_end_dates(
    cours_dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str | None = None,
) -> list[pd.Timestamp]:
    """Dernières dates de bourse par période de rebalancement."""
    freq = frequency or REBALANCE_FREQUENCY
    dates = cours_dates[(cours_dates >= start) & (cours_dates <= end)]
    if dates.empty:
        return []
    s = pd.Series(dates, index=dates)
    if freq == "quarterly":
        ends = s.groupby([s.index.year, s.index.quarter]).max()
    else:
        ends = s.groupby([s.index.year, s.index.month]).max()
    return sorted(ends.tolist())


def period_label(ts: pd.Timestamp, frequency: str | None = None) -> str:
    """Identifiant de période pour nom de fichier (ex. 2024-03 ou 2024-Q1)."""
    freq = frequency or REBALANCE_FREQUENCY
    if freq == "quarterly":
        return f"{ts.year}-Q{ts.quarter}"
    return f"{ts.year}-{ts.month:02d}"


def period_display_label(period_id: str) -> str:
    """Libellé affichage dashboard."""
    if "-Q" in period_id.upper():
        year, q = period_id.upper().split("-Q")
        return f"T{q} {year}"
    try:
        ts = pd.Timestamp(f"{period_id}-01")
        return f"{MONTH_NAMES_FR[ts.month]} {ts.year}"
    except Exception:
        return period_id


def turnover_annual_factor(frequency: str | None = None) -> int:
    freq = frequency or REBALANCE_FREQUENCY
    return 4 if freq == "quarterly" else 12


def regime_for_period(period_id: str, timeline: list[dict]) -> dict:
    """
    Régime discret pour une période de rebalancement.
    Le régime est mensuel (dernier jour du mois) ; pour un rebalancement
    trimestriel, on prend le dernier mois du trimestre.
    """
    default = {
        "is_bull": 0,
        "is_neutral": 1,
        "is_sideways": 1,  # miroir rétrocompat
        "is_bear": 0,
        "dominant": "neutral",
        "month": "—",
    }
    if not timeline:
        return default

    if "-Q" in period_id.upper():
        try:
            year_text, quarter_text = period_id.upper().split("-Q", 1)
            month_label = f"{int(year_text):04d}-{int(quarter_text) * 3:02d}"
        except (TypeError, ValueError):
            return default
        for row in timeline:
            if row.get("month") == month_label:
                return row
        return default

    try:
        month_label = pd.Timestamp(f"{period_id}-01").strftime("%Y-%m")
    except Exception:
        return default

    for row in timeline:
        if row.get("month") == month_label:
            return row
    return default
