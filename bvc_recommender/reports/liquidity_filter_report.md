# Sous-module C — Filtre de liquidité

- **Généré** : 2026-07-21T22:51:32.956580+00:00
- **Formule** : `score_final = score_TFT × sigmoid(VMQ_20j, seuil=500,000 MAD)`
- **Date as-of** : 2025-08-13
- **Univers** : 64 tickers
- **Source score** : tft_q50

## Labels (terciles)

- **TOP** : 22
- **NEUTRE** : 21
- **BOTTOM** : 21

## Top 15 (score_final décroissant)

| Rang | Ticker | Score TFT | VMQ 20j | Facteur liq. | Score final | Label |
|------|--------|-----------|---------|--------------|-------------|-------|
| 1 | HPS | 1.1656 | 13,876,194 | 1.000 | 1.1656 | TOP |
| 2 | SNP | 0.4803 | 615,975 | 0.997 | 0.4790 | TOP |
| 3 | DWY | 0.3041 | 481,097 | 0.976 | 0.2967 | TOP |
| 4 | LBV | 0.2904 | 7,091,015 | 1.000 | 0.2904 | TOP |
| 5 | STR | 0.2295 | 1,024,627 | 1.000 | 0.2295 | TOP |
| 6 | TQM | 0.1850 | 5,244,055 | 1.000 | 0.1850 | TOP |
| 7 | RDS | 0.1820 | 13,015,792 | 1.000 | 0.1820 | TOP |
| 8 | JET | 0.1642 | 3,527,692 | 1.000 | 0.1642 | TOP |
| 9 | GAZ | 0.1568 | 739,440 | 1.000 | 0.1568 | TOP |
| 10 | CSR | 0.1460 | 2,846,226 | 1.000 | 0.1460 | TOP |
| 11 | MNG | 0.1426 | 41,261,663 | 1.000 | 0.1426 | TOP |
| 12 | SID | 0.0949 | 2,869,904 | 1.000 | 0.0949 | TOP |
| 13 | DHO | 0.0834 | 887,244 | 1.000 | 0.0834 | TOP |
| 14 | AFM | 0.1037 | 315,860 | 0.741 | 0.0769 | TOP |
| 15 | MIC | 0.4470 | 134,026 | 0.135 | 0.0604 | TOP |

# Sous-module D — Optimisation NSGA-III

- **Candidats TOP** : 22
- **Solutions Pareto** : 24
- **CVaR MASI** : 0.018216
- **Limite CVaR cible (×0.9)** : 0.016394
- **Limite CVaR effective** : 0.016394
- **CVaR min réalisable** : 0.016152

## P_agressif — max alpha

- Positions : 22
- Alpha attendu : 0.184285
- CVaR 95 % : 0.016373
- Score liquidité : 0.1177
- CVaR / limite : 0.9987
- Contrainte CVaR respectée : True

| Ticker | Poids % |
|--------|---------|
| DRI | 19.67 |
| AFM | 18.81 |
| HPS | 7.89 |
| EQD | 7.35 |
| DWY | 5.74 |
| CMT | 4.96 |
| MNG | 4.32 |
| M2M | 3.16 |
| LBV | 2.09 |
| SNP | 2.00 |
| TQM | 2.00 |
| STR | 2.00 |
| RDS | 2.00 |
| JET | 2.00 |
| DHO | 2.00 |
| SID | 2.00 |
| GAZ | 2.00 |
| CSR | 2.00 |
| SOT | 2.00 |
| MUT | 2.00 |
| MIC | 2.00 |
| SMI | 2.00 |

## P_equilibre — point genou

- Positions : 22
- Alpha attendu : 0.163571
- CVaR 95 % : 0.016289
- Score liquidité : 0.1193
- CVaR / limite : 0.9936
- Contrainte CVaR respectée : True

| Ticker | Poids % |
|--------|---------|
| DRI | 19.99 |
| AFM | 19.45 |
| EQD | 7.45 |
| HPS | 6.02 |
| DWY | 5.60 |
| MNG | 5.09 |
| CMT | 5.05 |
| M2M | 3.34 |
| LBV | 2.00 |
| SNP | 2.00 |
| TQM | 2.00 |
| STR | 2.00 |
| RDS | 2.00 |
| JET | 2.00 |
| DHO | 2.00 |
| SID | 2.00 |
| GAZ | 2.00 |
| CSR | 2.00 |
| SOT | 2.00 |
| MUT | 2.00 |
| MIC | 2.00 |
| SMI | 2.00 |

## P_defensif — min CVaR

- Positions : 22
- Alpha attendu : 0.128329
- CVaR 95 % : 0.016155
- Score liquidité : 0.1025
- CVaR / limite : 0.9854
- Contrainte CVaR respectée : True

| Ticker | Poids % |
|--------|---------|
| DRI | 21.11 |
| AFM | 20.09 |
| EQD | 7.63 |
| DWY | 6.00 |
| CMT | 5.16 |
| MNG | 4.33 |
| SID | 3.44 |
| M2M | 3.43 |
| HPS | 2.79 |
| LBV | 2.00 |
| TQM | 2.00 |
| STR | 2.00 |
| SNP | 2.00 |
| RDS | 2.00 |
| DHO | 2.00 |
| JET | 2.00 |
| GAZ | 2.00 |
| CSR | 2.00 |
| SOT | 2.00 |
| MUT | 2.00 |
| MIC | 2.00 |
| SMI | 2.00 |
