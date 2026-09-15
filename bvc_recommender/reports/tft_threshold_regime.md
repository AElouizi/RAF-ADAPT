# TFT + régime à seuils — Optuna + réentraînement

- **Généré** : 2026-08-11T22:08:12.789864+00:00
- **Régime** : is_bull / is_neutral / is_bear (seuils both-AND)
- **Seuils** : {'p33_momentum': -0.016704, 'p67_momentum': 0.042896, 'p33_breadth': 0.392857, 'p67_breadth': 0.596491}
- **Cible** : alpha_ajuste_risque (excès vs MASI / vol. baissière — **pas** alpha de Jensen)
- **Splits** : train ≤ 2020-12-31 | val ≤ 2022-12-31 | test ≥ 2023-01-01

## Meilleurs hyperparamètres (Optuna / val_loss)

- Trials : 8
- Best val_loss : 0.38502436876296997
- Params : `{'hidden_size': 16, 'attention_head_size': 4, 'dropout': 0.2, 'learning_rate': 0.00024970737145052745}`

## Métriques TFT (seuils)

| Split | IC | Hit Ratio | Sharpe L/S |
|-------|-----|-----------|------------|
| val | -0.0455 | 45.6% | -1.94 |
| test | -0.0777 | 47.8% | -1.81 |

## Comparaison vs HMM (référence)

| Split | IC seuils | IC HMM |
|-------|-----------|--------|
| val | -0.0455 | -0.0080 |
| test | -0.0777 | -0.1290 |

**Verdict** : IC test (-0.0777) meilleur que HMM (-0.1290) mais encore négatif — ne pas lancer l'ablation step5.
