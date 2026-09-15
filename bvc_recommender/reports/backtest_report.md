# Backtest walk-forward — Étape 8

- **Période** : 2023-01-01 → 2025-08-13
- **Rebalancement** : Mensuel
- **Scores** : tft_q50
- **Coût transaction** : 0.3%
- **Périodes** : 31
- **Recommandations** : `reports/recommendations/YYYY-MM.csv` (+ agrégats `YYYY-QX.csv`)

## Métriques vs MASI

| Stratégie | Rend. ann. | Alpha | Sharpe | Sortino | Max DD | Calmar | CVaR 95% | Hit Ratio | Turnover trim. |
|-----------|------------|-------|--------|---------|--------|--------|----------|-----------|----------------|
| P_agressif | 35.8% | 5.5% | 1.339 | 2.557 | -22.6% | 1.585 | 2.03% | 55.0% | 204% |
| P_equilibre | 42.9% | 10.2% | 1.626 | 3.474 | -15.6% | 2.757 | 1.89% | 58.5% | 200% |
| P_defensif | 44.6% | 11.0% | 1.796 | 5.856 | -10.1% | 4.402 | 1.51% | 56.6% | 205% |
| MASI20_proxy | 26.3% | -4.0% | 1.959 | 3.293 | -10.1% | 2.615 | 1.70% | 55.5% | — |
| MASI | 30.3% | 0.0% | 2.349 | 4.198 | -10.2% | 2.957 | 1.46% | 57.1% | — |
