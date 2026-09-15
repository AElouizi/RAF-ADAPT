"""Constantes partagées pour le benchmarking RAF-ADAPT."""

from __future__ import annotations

from bvc_recommender.config import (
    RANDOM_STATE,
    SPLIT_TEST_START,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
)

# Aligné sur le backtest RAF-ADAPT / Étape 8
TRAIN_END = SPLIT_TRAIN_END  # "2020-12-31"
VAL_END = SPLIT_VAL_END  # "2022-12-31"
TEST_START = SPLIT_TEST_START  # "2023-01-01"
TEST_END = "2025-12-31"

TRANSACTION_COST = 0.003  # 0.3 % par trade (turnover unilatéral × coût)
RISK_FREE_RATE = 0.0  # taux sans risque annualisé (identique pour tous)
RANDOM_SEED = RANDOM_STATE  # 42

# Hit ratio : agrégation mensuelle (rendement du mois > 0)
HIT_RATIO_FREQ = "ME"
