# Rapport de validation C4 — fenêtre figée 2018-07 → 2025-06

Date : 2026-08-15  
Statut : backtest définitif avant validation finale (plateforme web non mise à jour)

## Protocole exécuté

- Modèle C4 inchangé : Hybrid Ridge+RF + régime HMM (prédictions / reco Stage 1 existantes).
- Fenêtre figée **avant** le backtest : 2018-07 → 2025-06 (84 mois de décision).
- VMQ_20j recalculé as-of t : `rolling_mean_20(prix × titres_echanges.fillna(0))`, 20 séances, pas de look-ahead (10 dates 2018–2025, max |Δ| = 0).
- Filtre **VMQ ≥ 500 000 MAD** (NaN exclus, pas d’imputation).
- SELL exclus.
- NSGA-III 3 objectifs : max Alpha, min CVaR 95 %, max L = sigmoid(VMQ) ; k et m inchangés.
- Sélection knee / distance à l’utopie.
- Hyperparamètres inchangés : pop=36, gen=30, w_max=40 %, prefs 1.0 / 0.5, TC = 0,3 % × turnover L1.
- Coûts débités le 1er jour de détention ; rendements sur (t , t_next].

## Contrôles d’univers

| Indicateur | Valeur |
|------------|--------|
| Mois effectivement backtestés | **84 / 84** |
| Mois sans portefeuille | **0** |
| Fallback equal-weight | **0** |
| Couverture VMQ (reco C4, as-of t) | **100 %** non-null |
| Nb moyen de titres éligibles (VMQ≥500k et non-SELL) = n_NSGA | **20,2** (min 4 en 2018-07, max 37) |
| Nb moyen de positions retenues | **17,8** (min 4, max 34) |
| SELL dans les holdings | **0** |
| Premier rendement strictement après la décision | **oui** (84/84) |

## Résultats 2018-07 → 2025-06 (nets de frais)

| | C4 | MASI | Equal Weight |
|--|---:|-----:|-------------:|
| Rendement annualisé | **14,34 %** | 6,44 % | 13,14 % |
| Volatilité annualisée | 14,69 % | 12,31 % | 13,37 % |
| Sharpe | 0,986 | 0,569 | 0,991 |
| Sortino | 1,507 | 0,837 | 1,499 |
| CVaR 95 % | 0,0218 | 0,0190 | 0,0202 |
| Max Drawdown | −28,32 % | −30,73 % | −27,53 % |
| Turnover moyen (L1) | 0,979 | — | 0,722 |
| Liquidité moyenne L | 0,9998 | — | 0,9994 |
| Nb moyen de positions | 17,8 | — | (univers éligible) |
| % mois positifs | 65,5 % | 56,0 % | 65,5 % |
| Richesse finale (base 100) | **264,88** | 157,43 | 245,30 |
| Rendement total | +164,9 % | +57,4 % | +145,3 % |

Equal Weight : même calendrier et **même univers** que NSGA (SELL exclu + VMQ ≥ 500k).

## Tests statistiques (écarts de rendements mensuels, n = 84)

**C4 − MASI**  
- Différence mensuelle moyenne : **+0,645 pp**  
- IC 95 % bootstrap : **[+0,033 pp ; +1,260 pp]** (n’inclut pas 0)  
- t-test apparié : t = 2,07, **p = 0,042**  
- Wilcoxon : **p = 0,038**  
- Mois C4 > MASI : 58,3 %  

→ Surperformance vs MASI **statistiquement significative à 5 %**.

**C4 − Equal Weight**  
- Différence mensuelle moyenne : +0,113 pp  
- IC 95 % bootstrap : [−0,407 pp ; +0,622 pp] (inclut 0)  
- t-test : p = 0,672  
- Wilcoxon : p = 0,755  
- Mois C4 > EW : 47,6 %  

→ **Pas de supériorité statistique** vs Equal Weight. Le Sharpe EW (0,991) est même légèrement supérieur à C4 (0,986). L’avantage économique de C4 est un surplus de rendement annualisé d’environ 1,2 pp, non significatif.

## Sous-périodes C4

| Période | n mois | Rendement | Volatilité | Sharpe | Sortino | CVaR | Max DD | Turnover | Liquidité |
|---------|-------:|----------:|-----------:|-------:|--------:|-----:|-------:|---------:|----------:|
| 2018-07–2019 | 18 | 8,68 % | 10,54 % | 0,843 | 1,383 | 0,0150 | −7,69 % | 1,098 | 0,9997 |
| 2020–2022 | 36 | **−3,30 %** | 15,97 % | −0,129 | −0,179 | 0,0259 | **−28,30 %** | 1,011 | 0,9997 |
| 2023–2025 | 30 | 37,05 % | 15,02 % | 2,175 | 3,699 | 0,0207 | −12,84 % | 0,868 | 0,9999 |

La période 2020–2022 (COVID / bear) porte l’essentiel du drawdown. 2023–2025 reste le régime le plus favorable.

## Comparaison avec l’ancien backtest 2023–2025

Ancien C4 (pareto **sans** filtre VMQ, volumes incomplets avant 2023) :

| Métrique | Ancien 2023–2025 | Nouveau 2023–2025 | Écart |
|----------|-----------------:|------------------:|------:|
| Rendement annualisé | 35,54 % | 37,05 % | +1,51 pp |
| Volatilité | 14,29 % | 15,02 % | +0,73 pp |
| Sharpe | 2,201 | 2,175 | −0,026 |
| Sortino | 3,719 | 3,699 | −0,020 |
| CVaR | 0,0196 | 0,0207 | +0,0011 |
| Max DD | −11,87 % | −12,84 % | −0,97 pp |
| Turnover | 0,919 | 0,868 | −0,051 |
| Liquidité L | 0,898 | 0,9999 | +0,102 |

Origine des écarts (2023–2025) : la couverture volume y était **déjà** bonne. L’essentiel vient du **filtre VMQ ≥ 500k** (univers plus liquide, L ≈ 1) combiné au NSGA 3 objectifs, et d’un re-tirage stochastique NSGA. Ce n’est pas un effet des volumes 2010–2022 sur cette sous-période.

Sur **2018–2022**, les nouveaux volumes changent réellement l’univers (auparavant VMQ ≈ 0). C’est là que la fenêtre allongée est informative : C4 reste positif vs MASI sur 7 ans, mais la robustesse est faible en 2020–2022.

## Fichiers

- `outputs/portfolios_C4_final/`
- `outputs/reports/stage2_C4_final_backtest/`
- `outputs/reports/stage2_C4_final_validation/`
- Archive ancien 2023–2025 : `*_archive_202301_202506`
- NSGA : `outputs/portfolios_stage2_C4_201807_202506/`
