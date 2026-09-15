# Architecture multi-agents — analyse et transformation

Objectif : orchestrer la plateforme existante **sans modifier** C1–C4, hyperparamètres, NSGA-III, protocole Stage 2, données historiques ni résultats scientifiques.

## 1. Architecture actuelle

Pipeline **séquentiel par scripts**, mémoire = fichiers locaux + quelques tables features Supabase. Le dashboard investisseur lit un **mart figé** (`outputs/platform_mart`), pas un graphe d’agents.

```
Admin / scripts CLI
        │
        ▼
Supabase (cours, fondamentaux, histo, features_*)
        │
        ▼
run_step2 → features techniques / fondamentales
run_step3 → ml_dataset
run_step4 → régimes (features_indices)
        │
        ▼
Walk-forward C1–C4 (factorial_cells, run_wf_etape3_train)
        │
        ▼
Stage 1 recos BUY/NEUTRAL/SELL (recommendations.py)
        │
        ▼
Stage 2 NSGA-III pareto 3 obj + knee, w_i ≤ 10 % (stage2_allocation.py)
        │
        ▼
Backtest + assemble_final_experiment + build_platform_mart
        │
        ▼
Dashboard Streamlit (app_c4.py) — lecture seule
```

Couplage : `run_from_2010_full.py` enchaîne des **sous-processus**. Pas de statuts PENDING/RUNNING/SUCCESS/FAILED, pas de reprise d’étape, pas d’UI admin.

Livrable scientifique figé : `outputs/final_experiment_2010_2025/` (2018-07 → 2025-06, 84 mois). Dashboard principal = **C2**.

## 2. Architecture multi-agents cible

Supabase (ou JSON local de repli) = **mémoire partagée**. Les agents ne s’appellent pas entre eux : l’orchestrateur lit le statut / l’événement de l’étape N avant de lancer N+1.

```
ADMIN (jeton) ──► Orchestrator ──► agent_jobs / agent_events / agent_logs
                         │
         FEATURES_READY  │  SELECTION_READY  │  ALLOCATION_READY
                         ▼
              Feature → Selection C1–C4 → Allocation NSGA-III
                         │
                         ▼
                    Backtest → Evaluation
                         │
                         ▼
              tables agent_* (append-only, run_id)
                         │
Investisseur ──► mart figé (inchangé) ──► app_c4.py
```

Mode par défaut **`ingest_frozen`** : publication / traçabilité du livrable validé, **aucun recalcul**.  
`compute_new` est **refusé** sur 2018-07→2025-06.

## 3. Fichiers créés ou modifiés

| Fichier | Rôle |
|---|---|
| `bvc_recommender/agents/*` | Contrats, mémoire, 6 agents, CLI, tests |
| `bvc_recommender/scripts/sql/create_agent_tables.sql` | Jobs, logs, events, registry, résultats |
| `bvc_recommender/config.py` | Noms de tables agents |
| `experiments/.../web/admin_agents.py` | UI administration |
| `experiments/.../web/app_c4.py` | Onglet admin derrière jeton |
| **Non modifiés** | `factorial_cells.py`, `recommendations.py`, `stage2_allocation.py`, NSGA, mart, FINAL |

## 4. Interactions (événements)

1. Data disponible (responsabilité admin)  
2. Feature Agent → `FEATURES_READY`  
3. Selection Agent → `SELECTION_READY`  
4. Allocation Agent → `ALLOCATION_READY`  
5. Backtest Agent → `BACKTEST_READY`  
6. Evaluation Agent → `EVALUATION_READY`  
7. Orchestrator → `WORKFLOW_SUCCESS` ou `WORKFLOW_FAILED` (stop immédiat)

## 5. Tables Supabase

**Existantes (inchangées)** : `market_data_*`, `fondamentaux_*`, `histo_const_ind`, `features_techniques`, `features_fondamentales`, `features_indices`.

**Ajoutées (orchestration uniquement)** : `agent_jobs`, `agent_logs`, `agent_events`, `agent_result_registry`, `agent_recommendations`, `agent_portfolio_holdings`, `agent_backtest_monthly`, `agent_evaluation_kpis`.

Toute ligne de résultat porte `run_id`, agent, période, configuration C1–C4, `model_version`, paramètres, timestamp, statut. **Append-only** : pas d’UPDATE du livrable historique.

## 6. Workflow

```
py -m bvc_recommender.agents --mode ingest_frozen --cells 1,2,3,4 --period-start 2018-07 --period-end 2025-06
```

Dashboard : expander « Administration » + jeton `BVC_ADMIN_TOKEN` (secours local `these-admin` si le jeton n’est pas dans `.env`).

SQL à exécuter une fois : `create_agent_tables.sql`.

## 7. Risques

| Risque | Mitigation |
|---|---|
| Recalcul accidentel C1–C4 / NSGA | `refuse_recompute_frozen` ; écriture interdite dans FINAL/mart |
| RLS Supabase | Repli JSON `bvc_recommender/data/agent_memory/` |
| Parquet Stage 1 absent | Repli mart C2 + holdings C1–C4 |
| Exposition investisseur | Onglet admin uniquement après jeton |
| Dérive des hyperparamètres | Paramètres **lus** depuis les modules existants, jamais recopiés dans une nouvelle logique modèle |

## 8. Traçabilité

`model_version = frozen_final_201807_202506_wmax10_pareto_knee`  
Stage 2 : liquidité Pareto, knee, `w_max=0.10`, pop/gen existants, coût de transaction du projet.
