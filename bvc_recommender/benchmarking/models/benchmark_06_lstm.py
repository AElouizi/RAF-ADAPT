"""
Benchmark 06 — LSTM standard (Niveau 2).

Composantes RAF-ADAPT :
  A (régime)     : NON injecté comme conditionnement (LSTM naïf)
  B (TFT)        : REMPLACÉE par LSTM 1 couche + dense (PyTorch), prédiction ponctuelle
  C (liquidité)  : intacte
  D (NSGA-III)   : intacte (portefeuille rapporté = P_equilibre)

Sans mécanisme d'attention, sans quantiles, sans vecteur de régime flou.
"""

from __future__ import annotations

from bvc_recommender.benchmarking.models._level2_base import Level2ScorerBenchmark


class LSTMBenchmark(Level2ScorerBenchmark):
    model_id = "06_lstm"
    model_name = "LSTM standard"
    scorer_name = "lstm"
