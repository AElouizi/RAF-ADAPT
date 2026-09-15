"""
Étape 5 — Allocation NSGA-III sur les top-25 de chaque cellule.

L'optimiseur (objectifs, contraintes, bornes) n'est pas modifié :
seule la liste de titres + score_final change selon la cellule.
Params pop/gen alignés sur le walk-forward projet (fast NSGA).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bvc_recommender.config import DATA_PROCESSED_DIR, FEATURES_DIR
from bvc_recommender.models.liquidity_filter import LABEL_TOP
from bvc_recommender.models.portfolio_optimizer import (
    NSGA_GENERATIONS,
    NSGA_POP_SIZE,
    optimize_portfolios,
)

logger = logging.getLogger(__name__)

# Aligné sur pipeline_acd / backtest walk-forward (identique pour les 4 cellules)
NSGA_POP_WALKFORWARD = 36
NSGA_GEN_WALKFORWARD = 30
PORTFOLIO_KEYS = ("P_agressif", "P_equilibre", "P_defensif")


def _load_parquet(name: str, folder: Path) -> pd.DataFrame:
    path = folder / f"{name}.parquet"
    if not path.is_file():
        path = folder / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{name} introuvable sous {folder}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_market_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cours = _load_parquet("market_data_cours_historique", DATA_PROCESSED_DIR)
    indices = _load_parquet("market_data_indices_historique", DATA_PROCESSED_DIR)
    technical = _load_parquet("features_techniques", FEATURES_DIR)
    return cours, indices, technical


def _vmq_asof(technical: pd.DataFrame, tickers: list[str], as_of: pd.Timestamp) -> pd.Series:
    tech = technical.copy()
    tech["date_cours"] = pd.to_datetime(tech["date_cours"], errors="coerce")
    tech = tech[tech["ticker"].isin(tickers) & (tech["date_cours"] <= as_of)]
    if tech.empty or "vmq_20j" not in tech.columns:
        return pd.Series({t: np.nan for t in tickers}, dtype=float)
    snap = (
        tech.sort_values(["ticker", "date_cours"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .set_index("ticker")["vmq_20j"]
    )
    return pd.to_numeric(snap.reindex(tickers), errors="coerce")


def build_ranked_from_selection(
    month_sel: pd.DataFrame,
    technical: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Construit l'entrée NSGA : 25 titres labellisés TOP.
    score_final = score de la cellule (pas de re-tercile / pas de filtre C).
    """
    sel = month_sel.copy()
    as_of = pd.Timestamp(as_of or sel["date_cours"].max())
    tickers = sel["ticker"].astype(str).tolist()
    ranked = pd.DataFrame(
        {
            "ticker": tickers,
            "score_final": pd.to_numeric(sel["score"], errors="coerce").values,
            "label": LABEL_TOP,
            "date_cours": as_of,
        }
    )
    ranked["vmq_20j"] = _vmq_asof(technical, tickers, as_of).values
    return ranked


def _filter_asof(df: pd.DataFrame, date_col: str, as_of: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out[out[date_col] <= pd.Timestamp(as_of)]


def optimize_month(
    ranked: pd.DataFrame,
    cours: pd.DataFrame,
    indices: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    pop_size: int = NSGA_POP_WALKFORWARD,
    n_gen: int = NSGA_GEN_WALKFORWARD,
) -> dict[str, Any]:
    idx_date_col = "date_index" if "date_index" in indices.columns else "date"
    cours_asof = _filter_asof(cours, "date_cours", as_of)
    indices_asof = _filter_asof(indices, idx_date_col, as_of)
    return optimize_portfolios(
        ranked,
        cours_asof,
        indices_asof,
        label_filter=LABEL_TOP,
        pop_size=pop_size,
        n_gen=n_gen,
    )


def _weights_frame(
    portfolios: dict[str, Any],
    *,
    cell_id: int,
    mois: str,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in PORTFOLIO_KEYS:
        p = portfolios.get(key) or {}
        tickers = p.get("tickers") or []
        weights = p.get("weights") or []
        for t, w in zip(tickers, weights):
            rows.append(
                {
                    "cell_id": cell_id,
                    "mois": mois,
                    "date_rebalance": as_of,
                    "portfolio": key,
                    "ticker": t,
                    "weight": float(w),
                    "n_positions": p.get("n_positions"),
                    "expected_alpha": p.get("expected_alpha"),
                    "cvar": p.get("cvar_95", p.get("cvar")),
                    "liquidity": p.get("liquidity_score", p.get("liquidity")),
                }
            )
    return pd.DataFrame(rows)


def run_allocation_for_cell(
    selections: pd.DataFrame,
    cours: pd.DataFrame,
    indices: pd.DataFrame,
    technical: pd.DataFrame,
    *,
    cell_id: int,
    split: str = "test",
    pop_size: int = NSGA_POP_WALKFORWARD,
    n_gen: int = NSGA_GEN_WALKFORWARD,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Walk-forward mensuel NSGA-III pour une cellule (split test par défaut)."""
    sel = selections[(selections["cell_id"] == cell_id) & (selections["split"] == split)].copy()
    if sel.empty:
        raise ValueError(f"Aucune sélection pour cell={cell_id} split={split}")

    # Réduire le bruit des loggers NSGA / liquidité
    opt_logger = logging.getLogger("bvc_recommender.models.portfolio_optimizer")
    prev_level = opt_logger.level
    opt_logger.setLevel(logging.WARNING)

    weight_frames: list[pd.DataFrame] = []
    month_meta: list[dict[str, Any]] = []
    months = sorted(sel["mois"].unique())

    try:
        for i, mois in enumerate(months, start=1):
            month_sel = sel[sel["mois"] == mois]
            as_of = pd.Timestamp(month_sel["date_cours"].max())
            ranked = build_ranked_from_selection(month_sel, technical, as_of=as_of)
            status = "ok"
            err = None
            try:
                portfolios = optimize_month(
                    ranked, cours, indices, as_of, pop_size=pop_size, n_gen=n_gen
                )
            except Exception as exc:  # noqa: BLE001 — fallback equal-weight documenté
                status = "fallback_equal"
                err = str(exc)
                logger.warning(
                    "[C%s | %s] NSGA échec (%s) — equal-weight sur 25 titres",
                    cell_id,
                    mois,
                    exc,
                )
                n = len(ranked)
                w = 1.0 / n if n else 0.0
                portfolios = {
                    key: {
                        "name": key,
                        "tickers": ranked["ticker"].tolist(),
                        "weights": [w] * n,
                        "n_positions": n,
                        "expected_alpha": float(
                            np.nansum(ranked["score_final"].values * w)
                        ),
                        "cvar": None,
                        "liquidity": None,
                    }
                    for key in PORTFOLIO_KEYS
                }

            weight_frames.append(
                _weights_frame(portfolios, cell_id=cell_id, mois=mois, as_of=as_of)
            )
            peq = portfolios.get("P_equilibre") or {}
            month_meta.append(
                {
                    "cell_id": cell_id,
                    "mois": mois,
                    "date_rebalance": str(as_of.date()),
                    "status": status,
                    "error": err,
                    "n_candidates": int(len(ranked)),
                    "n_positions_equilibre": peq.get("n_positions"),
                    "sum_weights_equilibre": float(sum(peq.get("weights") or [])),
                    "expected_alpha_equilibre": peq.get("expected_alpha"),
                    "cvar_equilibre": peq.get("cvar_95", peq.get("cvar")),
                    "liquidity_equilibre": peq.get(
                        "liquidity_score", peq.get("liquidity")
                    ),
                }
            )
            if i % 4 == 0 or i == len(months):
                logger.info(
                    "C%s allocation %s/%s (%s) status=%s",
                    cell_id,
                    i,
                    len(months),
                    mois,
                    status,
                )
    finally:
        opt_logger.setLevel(prev_level)

    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    return weights, month_meta


def run_allocation_all_cells(
    selections: pd.DataFrame,
    *,
    cell_ids: tuple[int, ...] = (1, 2, 3, 4),
    split: str = "test",
    pop_size: int = NSGA_POP_WALKFORWARD,
    n_gen: int = NSGA_GEN_WALKFORWARD,
) -> dict[str, Any]:
    cours, indices, technical = load_market_panels()
    all_weights: list[pd.DataFrame] = []
    all_meta: list[dict[str, Any]] = []

    logger.info(
        "NSGA-III walk-forward | pop=%s gen=%s | split=%s | cells=%s "
        "(defaults module: pop=%s gen=%s — non utilisés ici, mode walk-forward projet)",
        pop_size,
        n_gen,
        split,
        cell_ids,
        NSGA_POP_SIZE,
        NSGA_GENERATIONS,
    )

    for cell_id in cell_ids:
        logger.info("=== Allocation cellule %s ===", cell_id)
        w, meta = run_allocation_for_cell(
            selections,
            cours,
            indices,
            technical,
            cell_id=cell_id,
            split=split,
            pop_size=pop_size,
            n_gen=n_gen,
        )
        all_weights.append(w)
        all_meta.extend(meta)

    weights = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    meta_df = pd.DataFrame(all_meta)
    return {
        "weights": weights,
        "month_meta": meta_df,
        "nsga_params": {"pop_size": pop_size, "n_gen": n_gen, "split": split},
    }


def save_allocation_artifacts(
    results: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "portfolios_weights.parquet"
    meta_path = output_dir / "portfolios_month_meta.csv"
    json_path = output_dir / "portfolios_etape5_summary.json"

    results["weights"].to_parquet(weights_path, index=False)
    results["month_meta"].to_csv(meta_path, index=False)

    meta = results["month_meta"]
    summary: dict[str, Any] = {
        "nsga_params": results["nsga_params"],
        "cells": {},
    }
    for cell_id, grp in meta.groupby("cell_id"):
        summary["cells"][str(cell_id)] = {
            "n_months": int(len(grp)),
            "n_ok": int((grp["status"] == "ok").sum()),
            "n_fallback": int((grp["status"] != "ok").sum()),
            "mean_n_positions_equilibre": float(
                pd.to_numeric(grp["n_positions_equilibre"], errors="coerce").mean()
            ),
            "mean_sum_weights": float(
                pd.to_numeric(grp["sum_weights_equilibre"], errors="coerce").mean()
            ),
        }

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV lisible P_equilibre seulement
    eq = results["weights"][results["weights"]["portfolio"] == "P_equilibre"]
    eq_csv = output_dir / "portfolios_P_equilibre.csv"
    eq.to_csv(eq_csv, index=False)

    return {
        "weights": weights_path,
        "meta": meta_path,
        "summary": json_path,
        "equilibre_csv": eq_csv,
    }
