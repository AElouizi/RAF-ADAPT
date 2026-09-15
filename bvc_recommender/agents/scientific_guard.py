"""Gardes scientifiques : aucun overwrite du livrable figé, params lus (pas recopiés)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from bvc_recommender.benchmarking.config import TRANSACTION_COST
from bvc_recommender.config import FEATURES_DIR, RANDOM_STATE
from bvc_recommender.models.portfolio_optimizer import PORTFOLIO_MAX_WEIGHT
from experiments.factorial_hybrid_adapt.factorial_cells import (
    CELLS,
    HYBRID_WEIGHT_RF,
    HYBRID_WEIGHT_RIDGE,
)
from experiments.factorial_hybrid_adapt.paths import EXPERIMENT_ROOT, REPORTS_DIR
from experiments.factorial_hybrid_adapt.recommendations import RecommendationConfig
from experiments.factorial_hybrid_adapt.stage2_allocation import (
    EVAL_END_MONTH,
    EVAL_START_MONTH,
    NSGA_GEN_STAGE2,
    NSGA_POP_STAGE2,
    Stage2Config,
)

FROZEN_START = "2018-07"
FROZEN_END = "2025-06"
FINAL_DIR = EXPERIMENT_ROOT / "outputs" / "final_experiment_2010_2025"
MART_DIR = EXPERIMENT_ROOT / "outputs" / "platform_mart"
AGENT_RUNS_DIR = EXPERIMENT_ROOT / "outputs" / "agent_runs"

PROTECTED_DIRS = (FINAL_DIR, MART_DIR)

RIDGE_ALPHA = 1.0  # Ridge(alpha=1.0) dans factorial_cells._fit_ridge — ne pas changer le modèle


def validated_stage2_config() -> Stage2Config:
    """Protocole Stage 2 validé (pareto 3 obj, knee, w_i ≤ 10 %). Lecture, pas duplication de NSGA."""
    return Stage2Config(
        sell_mode="exclude",
        liquidity_mode="pareto",
        apply_vmq_filter=False,
        w_max=float(PORTFOLIO_MAX_WEIGHT),
        selection_rule="knee",
        pop_size=NSGA_POP_STAGE2,
        n_gen=NSGA_GEN_STAGE2,
        eval_start_month=EVAL_START_MONTH,
        eval_end_month=EVAL_END_MONTH,
    )


def scientific_parameters() -> dict[str, Any]:
    rec_cfg = RecommendationConfig()
    s2 = validated_stage2_config()
    return {
        "random_state": RANDOM_STATE,
        "ridge_alpha": RIDGE_ALPHA,
        "hybrid_weight_ridge": HYBRID_WEIGHT_RIDGE,
        "hybrid_weight_rf": HYBRID_WEIGHT_RF,
        "cells": [
            {"cell_id": c.cell_id, "name": c.name, "model": c.model, "regime": c.regime}
            for c in CELLS
        ],
        "recommendation": {
            "benchmark_mode": rec_cfg.benchmark_mode,
            "tau_method": rec_cfg.tau_method,
            "tau_quantile": rec_cfg.tau_quantile,
        },
        "stage2": {
            "liquidity_mode": s2.liquidity_mode,
            "selection_rule": s2.selection_rule,
            "sell_mode": s2.sell_mode,
            "w_max": s2.w_max,
            "apply_vmq_filter": s2.apply_vmq_filter,
            "nsga_pop": s2.pop_size,
            "nsga_gen": s2.n_gen,
            "objectives": ["alpha_max", "cvar_min", "liquidity_max"],
        },
        "transaction_cost": TRANSACTION_COST,
        "frozen_window": {"start": FROZEN_START, "end": FROZEN_END},
        "ml_eval_window": {"start": EVAL_START_MONTH, "end": EVAL_END_MONTH},
    }


def assert_no_historical_overwrite(path: Path) -> None:
    resolved = path.resolve()
    for protected in PROTECTED_DIRS:
        try:
            resolved.relative_to(protected.resolve())
        except ValueError:
            continue
        raise PermissionError(
            f"Écriture interdite dans le livrable scientifique figé : {path}"
        )


def period_overlaps_frozen(period_start: str, period_end: str) -> bool:
    return not (period_end < FROZEN_START or period_start > FROZEN_END)


def refuse_recompute_frozen(*, mode: str, allow_overwrite: bool, period_start: str, period_end: str) -> None:
    if mode != "compute_new":
        return
    if allow_overwrite:
        raise PermissionError(
            "allow_overwrite_historical est interdit : les résultats 2018-07→2025-06 restent figés."
        )
    if period_overlaps_frozen(period_start, period_end):
        raise PermissionError(
            "COMPUTE_NEW refusé sur la fenêtre figée 2018-07→2025-06. "
            "Utiliser ingest_frozen, ou une période strictement postérieure à 2025-06."
        )


def sha256_file(path: Path, *, max_bytes: int | None = 32_000_000) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as f:
        if max_bytes is not None and size > max_bytes:
            h.update(f.read(1_048_576))
            f.seek(max(0, size - 1_048_576))
            h.update(f.read())
            h.update(str(size).encode())
        else:
            for chunk in iter(lambda: f.read(1_048_576), b""):
                h.update(chunk)
    return h.hexdigest()


def frozen_artifact_manifest() -> dict[str, str]:
    files = [
        MART_DIR / "kpi_C1C4.csv",
        MART_DIR / "fact_recommendations_C4.csv",
        MART_DIR / "meta_platform.json",
        FINAL_DIR / "04_final_portfolios" / "C2_holdings_knee.csv",
        FINAL_DIR / "05_backtest" / "C2_monthly.csv",
        FINAL_DIR / "05_backtest" / "C2_daily.csv",
    ]
    out = {}
    for p in files:
        if p.is_file():
            out[str(p.as_posix())] = sha256_file(p)
    return out


def feature_local_paths() -> dict[str, Path]:
    names = ("features_fondamentales", "features_techniques", "features_indices")
    found: dict[str, Path] = {}
    for name in names:
        for ext in (".parquet", ".csv"):
            p = FEATURES_DIR / f"{name}{ext}"
            if p.is_file():
                found[name] = p
                break
    return found


def stage1_reco_path() -> Path | None:
    for p in (
        REPORTS_DIR / "stage1_recommendations_14blocks.parquet",
        FINAL_DIR / "02_stage1_recommendations" / "stage1_recommendations_14blocks.parquet",
        REPORTS_DIR / "stage2_inputs_14blocks.parquet",
        FINAL_DIR / "02_stage1_recommendations" / "stage2_inputs_14blocks.parquet",
    ):
        if p.is_file():
            return p
    return None
