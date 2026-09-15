"""
Benchmark 04 — Ridge Regression (Niveau 2).

Composantes RAF-ADAPT :
  A (régime)     : intacte (features régime dans ml_dataset si présentes ; pas d'input TFT)
  B (TFT)        : REMPLACÉE par Ridge (sklearn) sur features *_z aplaties
  C (liquidité)  : intacte
  D (NSGA-III)   : intacte (portefeuille rapporté = P_equilibre)

Cible : alpha_ajuste_risque | splits train≤2020 / val≤2022 / test≥2023
"""

from __future__ import annotations

from bvc_recommender.benchmarking.models._level2_base import Level2ScorerBenchmark


class RidgeBenchmark(Level2ScorerBenchmark):
    model_id = "04_ridge"
    model_name = "Ridge Regression"
    scorer_name = "ridge"
