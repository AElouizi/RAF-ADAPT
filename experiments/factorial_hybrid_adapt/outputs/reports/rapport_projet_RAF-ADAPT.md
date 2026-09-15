# Rapport de synthèse — Projet RAF-ADAPT (factorial_hybrid_adapt)

**Date** : 27 août 2026  
**Période d'évaluation principale** : juillet 2018 → juin 2025 (84 mois)  
**Données cours** : janvier 2010 → juin 2025  

> **Application web officielle** : `experiments/factorial_hybrid_adapt/web/app_c4.py`  
> **Protocole actif** : stratégies **C1–C5** (sélecteur sidebar).  
> Lancement : double-clic `Lancer_RAF_ADAPT.bat` ou `py -m streamlit run experiments/factorial_hybrid_adapt/web/app_c4.py` → http://localhost:8501

---

## Vue d'ensemble

RAF-ADAPT est un système de recommandation d'actions pour la Bourse de Casablanca (BVC), structuré en blocs fonctionnels :

| Bloc | Rôle | Statut |
|------|------|--------|
| **A** | Régime de marché (haussier / neutre / baissier) | ✅ Validé |
| **B** | Scoring C1–C5 (sélection) | ✅ 5 stratégies actives |
| **C** | Liquidité (filtre VMQ + sigmoïde) | ✅ Intégré NSGA |
| **D** | Allocation NSGA-III (α ↑, CVaR ↓) | ✅ 5 × 14 blocs |

**Protocole walk-forward** : train 36 mois → test 6 mois → pas 1 mois → **81 plis chevauchants** + **14 blocs indépendants**.

**Chaîne** : Données BVC → features z-scorées → Bloc A (régime) → Bloc B (scores ML) → reco BUY/NEUTRAL/SELL (τ causal) → Bloc C (liquidité) → Bloc D (NSGA knee, wᵢ ≤ 10 %) → mart `c1c5/` → dashboard Streamlit.

---

## Étape 0 — Données & features

Chargement Supabase → nettoyage splits → features techniques/fondamentales z-scorées cross-sectionnellement → `ml_dataset.parquet`.

- Panel ML depuis **juin 2015** (features z-scorées)
- Cible : `alpha_ajuste_risque` (excès vs MASI / vol baissière 20j)
- Cours MASI disponibles depuis **janvier 2010**

---

## Étape 1 — Bloc A : Régime de marché

Détection mensuelle via `masi_mom_3m`, `breadth_ma50` (terciles calibrés train 2015–2020). Variables : `is_bull`, `is_neutral`, `is_bear`. Utilisé par **C5** et le dashboard.

---

## Étape 2 — Walk-forward

- 81 plis overlapping (pas 1 mois) pour métriques descriptives
- 14 blocs indépendants (fold_ids 0, 6, 12, …, 78) pour tests statistiques
- Premier test OOS : **juillet 2018**

---

## Étape 3 — Bloc B : Stratégies C1–C5 (sélection)

| Strat. | model_id | Algorithme | Régime Bloc A |
|--------|----------|------------|---------------|
| **C1** | `c1_ridge` | Ridge | Non |
| **C2** | `c2_rf` | Random Forest | Non |
| **C3** | `c3_lightgbm` | LightGBM (Optuna) | Non |
| **C4** | `c4_hybrid_tri` | Ridge + RF + LGBM (z-score CS, ⅓ chacun) | Non |
| **C5** | `c5_hybrid_tri_regime` | Ridge + RF + LGBM + régime | Oui |

Chaque stratégie produit des scores puis **BUY / NEUTRAL / SELL** via seuil τ expanding causal. Les titres **SELL** sont exclus avant NSGA.

**Config figée** : `configs/selection_c1c5.json`

**Artefacts** : `bloc_b_wf_overlapping/`, `bloc_b_results_long_overlapping.csv`, `bloc_b_summary_all.json`

---

## Étape 4 — Recommandations BUY / NEUTRAL / SELL

Panel `stage2_inputs_14blocks.parquet` (score + reco + risque + liquidité VMQ). Métriques : spread BUY-SELL, hit ratio, turnover reco.

---

## Étape 5 — Bloc C : Liquidité

Filtre VMQ optionnel ; facteur sigmoïde L(VMQ). Mode protocole : **pareto** (liquidité = 3ᵉ objectif NSGA, pas de filtre VMQ dur).

---

## Étape 6 — Bloc D : NSGA-III × C1–C5 × 14 blocs

Pour chaque **model_id** C1–C5 et chaque bloc de test : NSGA-III (max α ajusté, min CVaR 95 %, max L(VMQ)), sélection **knee**, Σw = 1, wᵢ ≤ 10 %, pop = 36, gen = 30, sell_mode = exclude. Pas de filtre VMQ ≥ 500 k (liquidité = objectif Pareto).

Les résultats agrégés sont dans `bloc_d_summary_14blocks.csv` (Sharpe moyen, rendement annuel, max drawdown par stratégie).

Benchmarks de référence : MASI Sharpe ~0,67 ; EW MASI20 ~0,99.

**Artefacts** : `bloc_d_wf_14blocks/fold_*_{model_id}_weights.parquet`, `bloc_d_report_14blocks.json`

---

## Étape 7 — Mart plateforme C1–C5

Répertoire : `outputs/platform_mart/c1c5/`

| Fichier | Contenu |
|---------|---------|
| `fact_recommendations_c1c5.parquet` | Reco mensuelles par `model_id` |
| `fact_portfolio_holdings_c1c5.parquet` | Poids NSGA knee |
| `fact_portfolio_monthly_c1c5.parquet` | Richesse / rendements mensuels |
| `meta_c1c5.json` | Calendrier, métadonnées |

Génération : `build_c1c5_platform_mart.py` (après Bloc B + Bloc D).

---

## Étape 8 — Application web

Dashboard Streamlit avec **sélecteur C1–C5** dans la sidebar (stratégie active pour reco, portefeuille, historique).

```bash
cd "Projet_Cursor_RS_final"
py -m streamlit run experiments/factorial_hybrid_adapt/web/app_c4.py
```

### Architecture

| Composant | Fichier / source |
|-----------|------------------|
| UI principale | `web/app_c4.py` |
| Chargeurs données | `web/data_c4.py` (mart `c1c5/` + Bloc D) |
| Simulation | `web/simulation_ui.py`, `simulation_engine.py` |
| Mart C1–C5 | `outputs/platform_mart/c1c5/*.parquet` |
| Performance | `bloc_d_wf_14blocks/` + `bloc_d_summary_14blocks.csv` |

### Onglets

1. **Recommandations** — titres de la stratégie sélectionnée
2. **Portefeuille** — poids NSGA knee
3. **Simulation** — buy-and-hold utilisateur vs IA vs MASI
4. **Historique 2018–2025** — richesse mensuelle par stratégie
5. **Performance** — 5 courbes C1–C5 + MASI
6. **C1–C5 & tests** — tableau Bloc D
7. **Marché depuis 2010** — MASI long terme
8. **Administration** — agents (optionnel)

**PDF** : `outputs/reports/rapport_projet_RAF-ADAPT.pdf` (`build_rapport_pdf.py`)

---

## Recalcul complet

```bash
py experiments/factorial_hybrid_adapt/run_c1c5_pipeline.py
```

Séquence : Bloc B (81 folds) → Bloc D (14 blocs) → mart `c1c5/` → rapport PDF.

Smoke test rapide (1 fold) :

```bash
py -m experiments.factorial_hybrid_adapt.run_bloc_b_smoke_test --no-tune
```

---

## Synthèse

| Question | Réponse |
|----------|---------|
| Stratégies actives | C1 Ridge … C5 Hybrid triple + régime |
| Sélection vs allocation | Bloc B (scores/reco) puis Bloc D (NSGA) |
| Dashboard | Meta-sélection + NSGA **3 objectifs** (pareto) ; mart `platform_mart/best/` |
| Allocation NSGA | max α, min CVaR, max L(VMQ) — knee, wᵢ ≤ 10 % |
| Legacy | Anciens IDs (`ridge`, `lightgbm`, …) et mart C2 figé remplacés par C1–C5 |

---

## Arborescence des livrables

```
experiments/factorial_hybrid_adapt/outputs/
├── reports/
│   ├── bloc_b_results_long_overlapping.csv
│   ├── bloc_b_summary_all.json
│   ├── bloc_d_summary_14blocks.csv
│   ├── rapport_projet_RAF-ADAPT.pdf
│   └── rapport_projet_RAF-ADAPT.md
├── platform_mart/c1c5/          ← dashboard web
├── bloc_b_wf_overlapping/       ← preds 81 folds (C1–C5)
└── bloc_d_wf_14blocks/          ← NSGA 5 stratégies
```

---

*Rapport généré automatiquement — ne pas interpréter comme publication académique sans relecture méthodologique.*
