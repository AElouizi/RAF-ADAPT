# Baselines — Étape 5

- **Cible** : `alpha_ajuste_risque`
- **Features** : 33 colonnes z-scorées
- **Train** : ≤ 2020-12-31
- **Val** : → 2022-12-31
- **Test** : ≥ 2023-01-01

## Leaderboard (IC / Hit Ratio / Sharpe)

| Modèle | Split | IC moyen | Rank IC | Hit Ratio | Sharpe L/S |
|--------|-------|----------|---------|-----------|------------|
| ridge | val | -0.0419 | -0.0854 | 49.2% | -1.25 |
| ridge | test | 0.0726 | 0.0972 | 48.9% | 2.05 |
| lightgbm | val | 0.0764 | 0.0079 | 52.6% | 1.42 |
| lightgbm | test | -0.0578 | -0.0628 | 43.5% | -1.24 |
| xgboost | val | 0.0893 | 0.0198 | 52.8% | 2.33 |
| xgboost | test | -0.0532 | -0.0644 | 43.3% | -0.10 |

## Meilleur modèle (test — IC)

- **ridge** : IC = 0.0726, Hit = 48.9%, Sharpe = 2.05
