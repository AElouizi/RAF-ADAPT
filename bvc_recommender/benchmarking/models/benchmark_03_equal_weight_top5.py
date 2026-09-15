"""
Benchmark 03 — Equal-Weight Top 5 (Niveau 1 — naïf côté allocation).

Composantes RAF-ADAPT :
  A (régime)     : utilisée indirectement via scores TFT entraînés avec régime
  B (TFT)        : ACTIVÉE (score_final = TFT × liquidité)
  C (liquidité)  : ACTIVÉE (pénalité sigmoïde dans rank_and_label_universe)
  D (NSGA-III)   : DÉSACTIVÉE — remplacée par pondération égale des 5 meilleurs

Méthode : à chaque fin de mois, classer l'univers via A+B+C, prendre les 5 tickers
au plus haut score_final, poids 1/5, frais 0.3 % par trade.
"""

from __future__ import annotations

import pandas as pd

from bvc_recommender.benchmarking.base import BenchmarkModel, BenchmarkResult
from bvc_recommender.benchmarking.metrics import compute_metrics
from bvc_recommender.benchmarking.simulation import simulate_monthly_rebalance
from bvc_recommender.models.liquidity_filter import rank_and_label_universe


class EqualWeightTop5(BenchmarkModel):
    model_id = "03_equal_weight_top5"
    model_name = "Equal-Weight Top 5"
    level = 1
    replaces = "Composante D (NSGA-III → equal-weight Top 5)"

    def __init__(self, n_top: int = 5) -> None:
        self.n_top = n_top

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
        if scores is None or scores.empty:
            raise ValueError("Scores TFT requis pour Equal-Weight Top 5.")
        tech = technical if technical is not None else pd.DataFrame()

        def weights_fn(as_of: pd.Timestamp) -> dict[str, float]:
            ranked = rank_and_label_universe(scores, tech, as_of_date=as_of)
            if ranked.empty:
                return {}
            top = ranked.nlargest(self.n_top, "score_final")
            tickers = top["ticker"].tolist()
            # Restreindre aux tickers cotés à as_of
            tickers = [t for t in tickers if t in prices.columns]
            if not tickers:
                return {}
            w = 1.0 / len(tickers)
            return {t: w for t in tickers}

        daily = simulate_monthly_rebalance(prices, weights_fn)
        metrics = compute_metrics(daily, masi_returns, name=self.model_name)
        return BenchmarkResult(
            model_id=self.model_id,
            model_name=self.model_name,
            level=self.level,
            replaces=self.replaces,
            daily_returns=daily,
            metrics=metrics,
        ).with_monthly()
