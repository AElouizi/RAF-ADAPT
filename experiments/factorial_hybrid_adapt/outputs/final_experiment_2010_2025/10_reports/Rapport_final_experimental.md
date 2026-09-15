# Rapport expérimental FINAL — système de recommandation et d'allocation BVC

## 1. Objectif du projet
Construire un système en deux étages (recommandation titre puis allocation multi-objectifs) pour le marché actions marocain, évalué hors échantillon par walk-forward, sans look-ahead, avec coûts de transaction.

## 2. Données
Cours `market_data_cours_historique` (2010-01-04 → 2026-05-18), volumes `titres_echanges` complétés depuis `histo_volume` (2010–2022) sans écraser l'existant. Indices MASI. Features fondamentales à partir du 3 juin 2015 (`INDICATORS_MIN_DATE`).

## 3. Période d'étude
Demandée : **2010-01 → 2025-06** (figée avant backtest).
**Mois impossibles** : 2010-01 → 2018-06 — pas de recommandations Stage 1 (walk-forward 36 mois de train, panel ML dès 2015-07). Aucun réentraînement.
**Mois testés** : **2018-07 → 2025-06 (84 mois)**. Out-of-sample walk-forward (blocs de test 6 mois).

## 4. Architecture générale
Données as-of t → features → C1–C4 prédisent `alpha_ajuste_risque` → BUY/NEUTRAL/SELL → univers BUY+NEUTRAL (SELL exclu, **aucun filtre VMQ**) → NSGA-III (Alpha, CVaR, L) → front de Pareto → Knee/Utopie → backtest (t, t_next] − 0,3 % × turnover L1.

## 5–10. Stage 1 et cellules
C1 Ridge ; C2 Ridge + régime HMM ; C3 Hybrid Ridge+RF 50/50 ; C4 Hybrid + régime.
Recommandation : excess = prédiction (benchmark 0) ; BUY si >+tau, SELL si <-tau, sinon NEUTRAL. tau = quantile 0,33 des |pred| des plis passés.

## 11. Alpha
`alpha_ajuste_risque = (forward_ret_titre − forward_ret_MASI) / vol_baissiere_20d`. Le Stage 2 maximise `score_opt = predicted_return × pref` (BUY=1, NEUTRAL=0.5).

## 12. CVaR
CVaR 95 % historique des rendements journaliers du portefeuille, lookback 504 séances, **strictement ≤ date de rebalancement**.

## 13–15. Liquidité, VMQ, sigmoïde
`VMQ_20j = rolling_mean_20(prix × titres_echanges.fillna(0))` (formule existante).
`L = sigmoid(VMQ)` avec k=8/500000, m=250000 — **non modifiés**.
Rôle : **objectif Pareto uniquement**. Pas de VMQ ≥ 500k.

## 16–18. NSGA-III, Pareto, Knee
NSGA-III, pop=36, gen=30, w∈[0;10%]. Front complet persisté (`stage2_pareto_objectives/weights`).
Knee = distance euclidienne à l'utopie après min-max des objectifs F (tous minimisés : −α, CVaR, −L). Aucun rendement futur.

## 19–22. Portefeuille, backtest, benchmarks, métriques
Poids figés à t. EW = équipondération de l'univers BUY+NEUTRAL **de la même cellule**. MASI sur le même calendrier. TC=0,003.

## 23. Tests statistiques
Bootstrap IC 95 %, t-test apparié, Wilcoxon. Seuil 5 %. Distinction économique vs statistique.

## 24–25. Sous-périodes et robustesse
2010–2014 : non calculable. 2015–2019 : seulement 2018-07–2019-12. Sensibilité Knee vs max α / max L / min CVaR. Version A = Alpha+CVaR ; Version B = +Liquidité (principale).

## 26. Résultats C1–C4 / benchmarks
         Strategie  n_months  Rendement_total  Rendement  Volatilite  Sharpe  Sortino   CVaR  Max_DD  Turnover  Liquidite  n_positions_moy  pct_mois_positifs  ret_mensuel_moyen  wealth_final_100  duree_DD_jours
          C1 Ridge        84         0.886130     0.1009      0.1076   0.948    1.426 0.0159 -0.2219    0.9424     0.7106            39.65           0.642857           0.008119        188.612971             465
    EqualWeight C1        84         1.072898     0.1167      0.0996   1.159    1.697 0.0152 -0.2338    0.3503     0.6201            45.29           0.666667           0.009293        207.289766             516
 C2 Ridge + régime        84         1.022774     0.1126      0.1089   1.035    1.575 0.0160 -0.2260    0.9707     0.7026            38.65           0.642857           0.009014        202.277408             414
    EqualWeight C2        84         1.004788     0.1111      0.1003   1.101    1.612 0.0153 -0.2406    0.4382     0.6113            44.13           0.642857           0.008891        200.478843             520
         C3 Hybrid        84         1.021495     0.1125      0.1139   0.994    1.511 0.0165 -0.2417    0.9857     0.7118            31.57           0.619048           0.009039        202.149474             548
    EqualWeight C3        84         1.158628     0.1236      0.1051   1.162    1.733 0.0157 -0.2262    0.5431     0.6261            34.89           0.702381           0.009786        215.862788             518
C4 Hybrid + régime        84         0.813191     0.0943      0.1146   0.844    1.282 0.0167 -0.2434    0.9860     0.7131            31.52           0.595238           0.007780        181.319109             510
    EqualWeight C4        84         1.107330     0.1195      0.1046   1.132    1.689 0.0156 -0.2253    0.5506     0.6252            34.90           0.678571           0.009497        210.733048             518
              MASI        84         0.671608     0.0769      0.1238   0.661    0.979 0.0188 -0.3054       NaN        NaN              NaN           0.559524           0.006969        167.160762             626

## 27–28. Tests
         Comparaison  n_months  diff_moyenne_mensuelle  diff_mediane  pct_mois_A_gt_B   IC95_lo  IC95_hi  t_stat  p_ttest  p_wilcoxon  significatif_5pct  significatif_5pct_ttest  significatif_5pct_wilcoxon
           C3 - MASI        84                0.002070      0.004296           0.5952 -0.003659 0.007744  0.7158   0.4761      0.2785              False                    False                       False
C3 - Equal Weight C3        84               -0.000747      0.000809           0.5357 -0.003555 0.001961 -0.5283   0.5987      0.9822              False                    False                       False
             C3 - C1        84                0.000920      0.000317           0.5000 -0.001237 0.003259  0.7824   0.4362      0.7719              False                    False                       False
             C3 - C2        84                0.000025     -0.001514           0.4524 -0.002246 0.002342  0.0213   0.9831      0.6816              False                    False                       False
             C3 - C4        84                0.001259     -0.000082           0.4881 -0.000574 0.003101  1.3430   0.1829      0.3606              False                    False                       False
           C4 - MASI        84                0.000811      0.002438           0.5714 -0.004716 0.006268  0.2865   0.7752      0.5383              False                    False                       False
             C4 - C1        84               -0.000339      0.000352           0.5595 -0.002761 0.002145 -0.2737   0.7850      0.9538              False                    False                       False

## 29. Sous-périodes
cell   periode  n_months                   note  Rendement  Volatilite  Sharpe  Sortino   CVaR  Max_DD  Turnover  Liquidite
  C1 2010–2014         0 aucun mois de décision        NaN         NaN     NaN      NaN    NaN     NaN       NaN        NaN
  C1 2015–2019        18                    NaN     0.0518      0.0709   0.747    1.299 0.0091 -0.0504    0.8325     0.6335
  C1 2020–2022        36                    NaN    -0.0221      0.1113  -0.145   -0.198 0.0179 -0.2154    0.9862     0.6864
  C1 2023–2025        30                    NaN     0.3050      0.1197   2.285    3.770 0.0169 -0.1055    0.9557     0.7860
  C2 2010–2014         0 aucun mois de décision        NaN         NaN     NaN      NaN    NaN     NaN       NaN        NaN
  C2 2015–2019        18                    NaN     0.0559      0.0743   0.769    1.371 0.0093 -0.0468    0.8504     0.6092
  C2 2020–2022        36                    NaN    -0.0270      0.1155  -0.179   -0.244 0.0191 -0.2214    1.0363     0.6690
  C2 2023–2025        30                    NaN     0.3493      0.1167   2.627    4.522 0.0157 -0.0932    0.9640     0.7989
  C3 2010–2014         0 aucun mois de décision        NaN         NaN     NaN      NaN    NaN     NaN       NaN        NaN
  C3 2015–2019        18                    NaN     0.0484      0.0766   0.655    1.136 0.0098 -0.0635    0.9898     0.5768
  C3 2020–2022        36                    NaN    -0.0435      0.1183  -0.316   -0.430 0.0192 -0.2375    1.0004     0.6980
  C3 2023–2025        30                    NaN     0.3829      0.1252   2.654    4.531 0.0167 -0.1109    0.9654     0.8093
  C4 2010–2014         0 aucun mois de décision        NaN         NaN     NaN      NaN    NaN     NaN       NaN        NaN
  C4 2015–2019        18                    NaN     0.0479      0.0786   0.634    1.068 0.0107 -0.0658    0.9851     0.5713
  C4 2020–2022        36                    NaN    -0.0640      0.1195  -0.493   -0.669 0.0196 -0.2412    0.9911     0.7029
  C4 2023–2025        30                    NaN     0.3559      0.1250   2.500    4.310 0.0165 -0.1186    0.9804     0.8103

## Sensibilité (C4)
                           Strategie  n_months  Rendement_total  Rendement  Volatilite  Sharpe  Sortino   CVaR  Max_DD  Turnover  Liquidite  n_positions_moy  pct_mois_positifs  ret_mensuel_moyen  wealth_final_100  duree_DD_jours
                       A_knee_utopie        84         0.813191     0.0943      0.1146   0.844    1.282 0.0167 -0.2434    0.9860     0.7131            31.52           0.595238           0.007780        181.319109             510
                         B_max_alpha        84         0.941425     0.1057      0.1197   0.899    1.408 0.0174 -0.2493    1.0358     0.6550            31.10           0.595238           0.008682        194.142532             577
                     C_max_liquidite        84         1.234044     0.1295      0.1209   1.068    1.614 0.0182 -0.2563    1.0041     0.8351            31.98           0.583333           0.010338        223.404353             479
                          D_min_cvar        84         0.889369     0.1011      0.0981   1.032    1.576 0.0139 -0.2155    0.9240     0.5205            26.42           0.571429           0.008145        188.936940             510
E_composite_equilibre_identique_knee        84         0.813191     0.0943      0.1146   0.844    1.282 0.0167 -0.2434    0.9860     0.7131            31.52           0.595238           0.007780        181.319109             510

## 30. Limites
Échantillon Stage 1 dès 2018-07 seulement ; volumes 2023+ encore partiellement NULL ; NSGA stochastique (graine fixe) ; turnover élevé ; EW déjà très compétitif ; 2020–2022 difficile.

## 31–32. Conclusion et modèle final
Meilleure configuration selon Sharpe sur la fenêtre testable : **C2 Ridge + régime** (Sharpe=1.035, rend. ann.=0.1126).
C4 : rend. 0.0943, Sharpe 0.844, CVaR 0.0167, MaxDD -0.2434, L 0.7131.
MASI : 0.0769 / Sharpe 0.661. EW C4 : 0.1195 / Sharpe 1.132.
Le modèle opérationnel recommandé reste **C4 (Hybrid + régime)** comme spécification principale du projet, sous réserve des tests (voir JSON).

Livrable : `experiments/factorial_hybrid_adapt/outputs/final_experiment_2010_2025/`
