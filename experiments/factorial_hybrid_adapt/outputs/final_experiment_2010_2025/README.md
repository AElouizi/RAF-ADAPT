# Livrable expérimental FINAL (2010–2025)

Racine : `experiments/factorial_hybrid_adapt/outputs/final_experiment_2010_2025/`

| Dossier | Contenu |
|---|---|
| `01_protocol/` | Fenêtre figée, mois impossibles, checklist, couverture liquidité, algorithme knee |
| `02_stage1_recommendations/` | Recos C1–C4 et inputs Stage 2 (84 mois) |
| `03_stage2_pareto/` | Front Pareto **complet** + poids + objectifs NSGA-III 3 obj |
| `04_final_portfolios/` | Holdings Knee (`P_selected`) C1–C4 |
| `05_backtest/` | Rendements journaliers et mensuels nets (TC=0,3 %) |
| `06_benchmarks/` | MASI, Equal Weight par cellule, table de synthèse |
| `07_statistical_validation/` | Tests C3 vs MASI / EW / C1 / C2 / C4 |
| `08_robustness/` | Sous-périodes, sensibilité de sélection, 2 obj vs 3 obj |
| `09_figures/` | Figures haute résolution |
| `10_reports/` | Rapport thèse (MD + HTML) et résumé exécutif |

Fenêtre **demandée** : 2010-01 → 2025-06.
Fenêtre **testable sans réentraîner C1–C4** : **2018-07 → 2025-06 (84 mois)**.
Mois impossibles : 2010-01 → 2018-06 (walk-forward 36 mois, panel ML dès 2015-07).

Aucun filtre VMQ ≥ 500k. Liquidité = objectif Pareto uniquement. **Poids max par titre = 10 %**. Modèle livré = **C3 Hybrid** (meilleur Sharpe).
