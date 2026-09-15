# Sous-module B — TFT vs Baselines

- **Modèle** : Temporal Fusion Transformer (pytorch-forecasting)
- **Cible** : alpha ajusté au risque
- **Régime injecté** : is_bull, is_sideways, is_bear (HMM one-hot, known future)
- **Quantiles** : Q10 / Q50 / Q90
- **Validation** : train ≤ 2020-12-31 | val ≤ 2022-12-31 | test ≥ 2023-01-01

## TFT — métriques

| Split | Lignes | IC | Hit Ratio | Sharpe L/S |
|-------|--------|-----|-----------|------------|
| val | 26801 | -0.0076 | 46.6% | -0.14 |
| test | 36085 | -0.1293 | 50.1% | -2.40 |

## Comparaison test (IC)

- **TFT** : -0.1293
- **ridge** : 0.0726
- **lightgbm** : -0.0578
- **xgboost** : -0.0532

**Meilleur IC test** : ridge
