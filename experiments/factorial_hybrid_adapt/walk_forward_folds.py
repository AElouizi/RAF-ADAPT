"""
Validation glissante (rolling walk-forward) — générateur de plis.

Design :
- train = 3 ans glissants (36 mois, taille fixe)
- test  = 6 mois, immédiatement après le train (pas d'embargo)
- pas   = 1 mois

Deux familles de plis :
1. Mensuels chevauchants — métriques descriptives / stabilité
2. Blocs non-chevauchants de 6 mois — Diebold-Mariano + allocation (étapes ultérieures)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

TRAIN_MONTHS = 36  # 3 ans
TEST_MONTHS = 6
STEP_MONTHS = 1


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_start_month: str
    train_end_month: str
    test_start_month: str
    test_end_month: str

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("train_start", "train_end", "test_start", "test_end"):
            d[k] = str(pd.Timestamp(d[k]).date())
        return d


def _month_range(start: pd.Period, end: pd.Period) -> list[pd.Period]:
    """Liste inclusive de mois calendaires [start, end]."""
    if end < start:
        return []
    return list(pd.period_range(start, end, freq="M"))


def _period_bounds(months: list[pd.Period]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Début du 1er mois → fin du dernier mois (calendaire)."""
    start = months[0].to_timestamp(how="start")
    end = months[-1].to_timestamp(how="end").normalize()
    return pd.Timestamp(start), pd.Timestamp(end)


def available_months_from_dates(
    dates: Iterable[pd.Timestamp] | pd.Series,
) -> list[pd.Period]:
    s = pd.to_datetime(pd.Series(list(dates) if not isinstance(dates, pd.Series) else dates))
    s = s.dropna()
    if s.empty:
        return []
    lo = s.min().to_period("M")
    hi = s.max().to_period("M")
    return _month_range(lo, hi)


def generate_rolling_folds(
    months: list[pd.Period],
    *,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
    step_months: int = STEP_MONTHS,
) -> list[Fold]:
    """
    Génère tous les plis mensuels chevauchants.

    Pour un index i (pas = step_months) :
      train = months[i : i + train_months]
      test  = months[i + train_months : i + train_months + test_months]
    Contiguïté train→test : pas de mois d'écart (test commence au mois suivant train_end).
    """
    need = train_months + test_months
    if len(months) < need:
        return []

    folds: list[Fold] = []
    fold_id = 0
    i = 0
    while i + need <= len(months):
        train_m = months[i : i + train_months]
        test_m = months[i + train_months : i + need]
        tr_start, tr_end = _period_bounds(train_m)
        te_start, te_end = _period_bounds(test_m)
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=tr_start,
                train_end=tr_end,
                test_start=te_start,
                test_end=te_end,
                train_start_month=str(train_m[0]),
                train_end_month=str(train_m[-1]),
                test_start_month=str(test_m[0]),
                test_end_month=str(test_m[-1]),
            )
        )
        fold_id += 1
        i += step_months
    return folds


def generate_nonoverlapping_blocks(
    months: list[pd.Period],
    *,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
) -> list[Fold]:
    """
    Sous-ensemble de plis dont les fenêtres de test ne se chevauchent pas
    (pas = test_months). Destiné au Diebold-Mariano / allocation.
    """
    return generate_rolling_folds(
        months,
        train_months=train_months,
        test_months=test_months,
        step_months=test_months,
    )


def folds_to_frame(folds: list[Fold]) -> pd.DataFrame:
    return pd.DataFrame([f.to_dict() for f in folds])


def summarize_fold_grid(
    overlapping: list[Fold],
    nonoverlapping: list[Fold],
    *,
    months: list[pd.Period],
    expected_overlapping: int = 78,
    expected_blocks: int = 13,
) -> dict:
    n_ov = len(overlapping)
    n_no = len(nonoverlapping)
    return {
        "data_first_month": str(months[0]) if months else None,
        "data_last_month": str(months[-1]) if months else None,
        "n_months_available": len(months),
        "train_months": TRAIN_MONTHS,
        "test_months": TEST_MONTHS,
        "step_months_overlapping": STEP_MONTHS,
        "step_months_nonoverlapping": TEST_MONTHS,
        "n_folds_overlapping": n_ov,
        "n_blocks_nonoverlapping": n_no,
        "expected_overlapping_approx": expected_overlapping,
        "expected_blocks_approx": expected_blocks,
        "delta_overlapping_vs_expected": n_ov - expected_overlapping,
        "delta_blocks_vs_expected": n_no - expected_blocks,
        "alert_overlapping": abs(n_ov - expected_overlapping) > 5,
        "alert_blocks": abs(n_no - expected_blocks) > 2,
    }
