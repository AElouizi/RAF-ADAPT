"""
Benchmark 05 — LightGBM (Niveau 2).

Composantes RAF-ADAPT :
  A (régime)     : intacte (features dataset)
  B (TFT)        : REMPLACÉE par LightGBM sur features *_z aplaties
  C (liquidité)  : intacte
  D (NSGA-III)   : intacte (portefeuille rapporté = P_equilibre)

Cible : alpha_ajuste_risque | splits train≤2020 / val≤2022 / test≥2023
"""

from __future__ import annotations

from bvc_recommender.benchmarking.models._level2_base import Level2ScorerBenchmark


class LightGBMBenchmark(Level2ScorerBenchmark):
    model_id = "05_lightgbm"
    model_name = "LightGBM"
    scorer_name = "lightgbm"
