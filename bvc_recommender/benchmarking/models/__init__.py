"""Registre des modèles de benchmarking."""

from __future__ import annotations

from bvc_recommender.benchmarking.base import BenchmarkModel
from bvc_recommender.benchmarking.models.benchmark_01_equal_weight_masi20 import EqualWeightMASI20
from bvc_recommender.benchmarking.models.benchmark_02_buyhold_masi import BuyHoldMASI
from bvc_recommender.benchmarking.models.benchmark_03_equal_weight_top5 import EqualWeightTop5
from bvc_recommender.benchmarking.models.benchmark_04_ridge import RidgeBenchmark
from bvc_recommender.benchmarking.models.benchmark_05_lightgbm import LightGBMBenchmark
from bvc_recommender.benchmarking.models.benchmark_06_lstm import LSTMBenchmark
from bvc_recommender.benchmarking.models.benchmark_07_tft_no_regime import TFTNoRegimeBenchmark
from bvc_recommender.benchmarking.models.benchmark_08_tft_fuzzy import TFTFuzzyBenchmark
from bvc_recommender.benchmarking.models.benchmark_09_tft_markowitz import TFTMarkowitzBenchmark
from bvc_recommender.benchmarking.models.benchmark_10_raf_adapt_full import RAFAdaptFullBenchmark

LEVEL1_MODELS: list[type[BenchmarkModel]] = [
    EqualWeightMASI20,
    BuyHoldMASI,
    EqualWeightTop5,
]

LEVEL2_MODELS: list[type[BenchmarkModel]] = [
    RidgeBenchmark,
    LightGBMBenchmark,
    LSTMBenchmark,
]

LEVEL3_MODELS: list[type[BenchmarkModel]] = [
    TFTNoRegimeBenchmark,
    TFTFuzzyBenchmark,
    TFTMarkowitzBenchmark,
    RAFAdaptFullBenchmark,
]

ALL_MODELS: list[type[BenchmarkModel]] = [
    *LEVEL1_MODELS,
    *LEVEL2_MODELS,
    *LEVEL3_MODELS,
]


def get_models(*, levels: set[int] | None = None) -> list[BenchmarkModel]:
    models = [cls() for cls in ALL_MODELS]
    if levels is not None:
        models = [m for m in models if m.level in levels]
    return models
