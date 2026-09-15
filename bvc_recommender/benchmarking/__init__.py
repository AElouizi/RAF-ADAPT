"""
Module benchmarking RAF-ADAPT — comparaison de 10 modèles / variantes.

Structure
---------
benchmarking/
  __init__.py
  config.py                 # périodes, coûts, taux sans risque
  metrics.py                # compute_metrics (alpha, Sharpe, … Omega)
  data_access.py            # chargement cours / indices / scores TFT
  simulation.py             # helpers walk-forward + frais 0.3 %
  base.py                   # interface BenchmarkModel
  models/
    benchmark_01_equal_weight_masi20.py   # Niveau 1
    benchmark_02_buyhold_masi.py          # Niveau 1
    benchmark_03_equal_weight_top5.py     # Niveau 1 (A+B+C, pas D)
    benchmark_04_ridge.py                 # Niveau 2 (à venir)
    …
    benchmark_10_raf_adapt_full.py        # Niveau 3 (à venir)
  run_all_benchmarks.py                   # orchestre les modèles
  generate_comparison_table.py            # tableau CSV + figure

Niveaux
-------
1. Benchmarks naïfs (pas de ML) — fichiers 01–03
2. Remplacent Composante B (TFT) — 04–06
3. Ablations architecturales — 07–10
"""

from bvc_recommender.benchmarking.metrics import compute_metrics

__all__ = ["compute_metrics"]
