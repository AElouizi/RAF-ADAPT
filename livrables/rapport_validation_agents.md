# Rapport de validation — architecture multi-agents

Date : 2026-08-15  
Mode testé : `ingest_frozen` (publication du livrable scientifique, **aucun recalcul**).  
Commande : `py -3 bvc_recommender/agents/test_orchestrator.py` → **ALL TESTS PASSED**.

`model_version` : `frozen_final_201807_202506_wmax10_pareto_knee`

Les fichiers `outputs/final_experiment_2010_2025/` et `outputs/platform_mart/` n’ont pas été écrits (somme SHA-256 identique avant/après ingest).

---

## Agent 1 — Feature

| | |
|---|---|
| Rôle | Vérifier la disponibilité des features techniques, fondamentales et indices ; optionnellement relancer `run_step2/3/4` **hors** fenêtre figée |
| Entrées | Tables Supabase marché / fondamentaux ; ou fichiers locaux `features_*` ; ou livrable FINAL déjà validé |
| Traitement | `ingest_frozen` : contrôle + manifeste. `compute_new` : scripts existants inchangés, **refusé** sur 2018-07→2025-06 |
| Sorties | Événement `FEATURES_READY` ; manifeste sous `outputs/agent_runs/<run_id>/features/` |
| Tables | Lecture `features_techniques`, `features_fondamentales`, `features_indices` (pas d’écrasement du mart) |
| Statut | SUCCESS si livrable figé ou features présentes ; FAILED sinon (pipeline stoppée) |
| Tests | Fail-stop si Feature FAILED ; ingest SUCCESS sans toucher au livrable |

## Agent 2 — Stock Selection (C1–C4)

| | |
|---|---|
| Rôle | Publier score + BUY/NEUTRAL/SELL par titre et mois, cellules C1–C4 **sans retuner** |
| Entrées | `stage1_recommendations` / `stage2_inputs` si présents ; sinon mart C2 + holdings knee C1–C4 |
| Traitement | Filtre période / cellules ; pas d’appel à `train_base_models` |
| Sorties | `SELECTION_READY` ; parquet `agent_runs/.../selection/recommendations.parquet` |
| Tables | `agent_recommendations` (append, `run_id`) |
| Statut | SUCCESS si lignes > 0 |
| Tests | Ingest C1–C4 inclus dans le workflow complet |

## Agent 3 — Portfolio Allocation (NSGA-III)

| | |
|---|---|
| Rôle | Publier les portefeuilles Stage 2 validés (3 obj. Alpha↑ CVaR↓ Liquidité↑, knee, w_i ≤ 10 %) |
| Entrées | `C{1-4}_holdings_knee.csv` du livrable FINAL |
| Traitement | Contrôle `max(w) ≤ 0.10` ; paramètres lus via `validated_stage2_config()` (module existant) |
| Sorties | `ALLOCATION_READY` |
| Tables | `agent_portfolio_holdings` |
| Statut | FAILED si holdings absents ou contrainte de poids violée |
| Tests | Ingest SUCCESS ; contrainte 10 % vérifiée |

## Agent 4 — Backtesting

| | |
|---|---|
| Rôle | Publier rendements, turnover, coûts, richesse (protocole existant, TC = 0,3 %) |
| Entrées | `05_backtest/C*_monthly.csv` et pointeurs `C*_daily.csv` |
| Traitement | Aucun nouveau moteur ; reconstruction `wealth100` à partir des `net_return` mensuels figés |
| Sorties | `BACKTEST_READY` |
| Tables | `agent_backtest_monthly` |
| Statut | SUCCESS si CSV mensuels présents |
| Tests | Inclus dans l’ingest |

## Agent 5 — Evaluation

| | |
|---|---|
| Rôle | Sharpe, Sortino, CVaR, Max DD, rendement annualisé, comparaison C1–C4 / EW / MASI |
| Entrées | `platform_mart/kpi_C1C4.csv` (+ tests, sous-périodes en artefacts) |
| Traitement | Recopie des indicateurs **déjà validés** (source de vérité plateforme) |
| Sorties | `EVALUATION_READY` |
| Tables | `agent_evaluation_kpis` |
| Statut | SUCCESS ; test exige `rows ≥ 5` |
| Tests | Workflow SUCCESS ; KPI C2 Sharpe 1,035 inchangé dans le mart |

## Agent 6 — Orchestrator

| | |
|---|---|
| Rôle | Ordre Feature → Selection → Allocation → Backtest → Evaluation ; PENDING/RUNNING/SUCCESS/FAILED |
| Entrées | `WorkflowRequest` (période, cellules, mode) |
| Traitement | Crée un job enfant, attend SUCCESS, émet l’événement, sinon stop + `WORKFLOW_FAILED` |
| Sorties | `WORKFLOW_SUCCESS` / `WORKFLOW_FAILED` ; `workflow_id` |
| Tables | `agent_jobs`, `agent_logs`, `agent_events`, `agent_result_registry` |
| Statut | SUCCESS seulement si les 5 agents SUCCESS |
| Tests | (1) ingest complet SUCCESS et checksums livrable identiques (2) Feature FAILED ⇒ Selection non lancée (3) `compute_new` refusé sur la fenêtre figée |

---

## Interface

- **Investisseur** : onglets existants inchangés (recommandations, portefeuille, historique, performance, C1–C4, marché).  
- **Administrateur** : expander Administration + jeton `BVC_ADMIN_TOKEN` → onglet Agents (lancer, période, C1–C4, statuts, logs, erreurs, reprise).  
- SQL : `bvc_recommender/scripts/sql/create_agent_tables.sql`.  
- CLI : `py -m bvc_recommender.agents --mode ingest_frozen`.

## Non-régression scientifique

Aucun changement dans `factorial_cells.py`, `recommendations.py`, `stage2_allocation.py`, `portfolio_optimizer.py` (NSGA), ni remplacement des CSV/parquet du livrable FINAL / mart.
