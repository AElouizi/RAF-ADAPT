"""Interface commune des modèles de benchmarking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BenchmarkResult:
    model_id: str
    model_name: str
    level: int
    replaces: str
    daily_returns: pd.Series
    metrics: dict[str, Any] = field(default_factory=dict)
    monthly_returns: pd.Series | None = None

    def with_monthly(self) -> "BenchmarkResult":
        if self.daily_returns.empty:
            self.monthly_returns = pd.Series(dtype=float)
        else:
            self.monthly_returns = (1 + self.daily_returns).resample("ME").prod() - 1
        return self


class BenchmarkModel(ABC):
    model_id: str
    model_name: str
    level: int
    replaces: str  # composante remplacée / désactivée

    @abstractmethod
    def run(
        self,
        *,
        cours: pd.DataFrame,
        indices: pd.DataFrame,
        prices: pd.DataFrame,
        masi_returns: pd.Series,
        scores: pd.DataFrame | None = None,
        technical: pd.DataFrame | None = None,
    ) -> BenchmarkResult:
        """Produit la série de rendements journaliers nets sur la période de test."""
