"""
Bloc D généralisé — NSGA-III en boucle sur les modèles Bloc B × blocs indépendants.

Réutilise ``run_stage2_model`` / ``optimize_universe`` sans dupliquer NSGA-III.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bvc_recommender.backtest.metrics import compute_backtest_metrics
from bvc_recommender.benchmarking.config import TRANSACTION_COST
from bvc_recommender.benchmarking.simulation import (
    masi20_equal_weights,
    portfolio_daily_returns,
    turnover_cost,
)
from bvc_recommender.config import LIQUIDITY_VMQ_THRESHOLD_MAD
from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.bloc_b_metrics import run_omnibus_test
from experiments.factorial_hybrid_adapt.bloc_b_models import MODEL_GROUP
from experiments.factorial_hybrid_adapt.stage2_allocation import (
    DEFAULT_L_MIN,
    Stage2Config,
    run_stage2_model,
)
from experiments.factorial_hybrid_adapt.stage2_bloc_b_loader import (
    BLOC_D_MODEL_IDS,
    DEFAULT_PREDS_DIR,
    block_fold_ids_from_summary,
    build_stage2_inputs_for_model_fold,
    load_all_fold_predictions,
)
from experiments.factorial_hybrid_adapt.stage2_metrics import (
    _liq_asof,
    _masi_returns,
    _price_panel,
    _weights_dict,
)
from experiments.factorial_hybrid_adapt.walk_forward_folds import Fold, generate_nonoverlapping_blocks

logger = logging.getLogger(__name__)

PORTFOLIO = "P_selected"
FINANCIAL_METRICS = (
    "ann_return",
    "sharpe",
    "sortino",
    "cvar_95_realized",
    "max_drawdown",
    "turnover_mean",
    "liquidity_mean",
    "total_return_block",
)


def _blocks_from_fold_ids(fold_ids: list[int], all_folds: list[Fold]) -> list[Fold]:
    by_id = {f.fold_id: f for f in all_folds}
    return [by_id[fid] for fid in fold_ids if fid in by_id]


def simulate_block_portfolio(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    technical: pd.DataFrame,
    masi: pd.Series,
    *,
    model_id: str,
    fold_id: int,
    portfolio: str = PORTFOLIO,
) -> dict[str, Any]:
    """Simule rendements réalisés sur un bloc (P_selected)."""
    w = weights[
        (weights["model_id"] == model_id)
        & (weights["fold_id"] == fold_id)
        & (weights["portfolio"] == portfolio)
    ].copy()
    if w.empty:
        return {"daily_returns": pd.Series(dtype=float), "monthly": pd.DataFrame()}

    w["date_rebalance"] = pd.to_datetime(w["date_rebalance"], errors="coerce")
    months = sorted(w["mois"].unique())
    parts: list[pd.Series] = []
    turnovers: list[float] = []
    liqs: list[float] = []
    monthly: list[dict[str, Any]] = []
    prev_w: dict[str, float] = {}

    for i, mois in enumerate(months):
        month_w = w[w["mois"] == mois]
        reb = pd.Timestamp(month_w["date_rebalance"].iloc[0])
        if i + 1 < len(months):
            nxt_mois = months[i + 1]
            nxt = pd.Timestamp(w.loc[w["mois"] == nxt_mois, "date_rebalance"].iloc[0])
        else:
            nxt = prices.index[prices.index >= reb].max()

        wdict = _weights_dict(month_w)
        if not wdict:
            continue

        if prev_w:
            to = float(
                sum(abs(wdict.get(t, 0.0) - prev_w.get(t, 0.0)) for t in set(wdict) | set(prev_w))
            )
        else:
            to = float(sum(wdict.values()))
        turnovers.append(to)

        tickers = list(wdict.keys())
        liq_vals = _liq_asof(technical, tickers, reb)
        liq_w = 0.0
        w_avail = 0.0
        for t in tickers:
            v = liq_vals.get(t, np.nan)
            if pd.notna(v):
                liq_w += wdict[t] * float(v)
                w_avail += wdict[t]
        if w_avail > 0:
            liq_w = liq_w / w_avail * sum(wdict.values())
        liqs.append(float(liq_w))

        daily = portfolio_daily_returns(wdict, prices, reb, nxt)
        cost = turnover_cost(prev_w, wdict, TRANSACTION_COST)
        if not daily.empty and cost > 0:
            daily = daily.copy()
            daily.iloc[0] -= cost
        if not daily.empty:
            parts.append(daily)
            monthly.append(
                {
                    "fold_id": fold_id,
                    "model_id": model_id,
                    "mois": mois,
                    "ret_month": float((1 + daily).prod() - 1),
                    "turnover": to,
                    "liquidite_ponderee": float(liq_w),
                }
            )
        prev_w = wdict

    daily_rets = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    bench = masi.reindex(daily_rets.index).fillna(0.0) if not daily_rets.empty else masi
    perf = (
        compute_backtest_metrics(daily_rets, bench, name=model_id)
        if not daily_rets.empty
        else {}
    )
    return {
        "daily_returns": daily_rets,
        "monthly": pd.DataFrame(monthly),
        "turnover_mean": float(np.mean(turnovers)) if turnovers else np.nan,
        "liquidity_mean": float(np.mean(liqs)) if liqs else np.nan,
        "perf": perf,
        "total_return_block": float((1 + daily_rets).prod() - 1) if not daily_rets.empty else np.nan,
    }


def simulate_benchmark_block(
    fold: Fold,
    prices: pd.DataFrame,
    cours: pd.DataFrame,
    masi: pd.Series,
    *,
    kind: str,
) -> dict[str, Any]:
    """MASI buy-hold ou Equal-Weight MASI20 sur la fenêtre test du bloc."""
    t0 = pd.Timestamp(fold.test_start)
    t1 = pd.Timestamp(fold.test_end)
    reb_dates = prices.index[(prices.index >= t0) & (prices.index <= t1)]
    if len(reb_dates) == 0:
        return {"daily_returns": pd.Series(dtype=float), "perf": {}}

    # Rebalance mensuel fin de mois dans le bloc
    months = pd.period_range(fold.test_start.to_period("M"), fold.test_end.to_period("M"), freq="M")
    parts: list[pd.Series] = []
    prev_w: dict[str, float] = {}

    for i, per in enumerate(months):
        month_days = prices.index[prices.index.to_period("M") == per]
        if len(month_days) == 0:
            continue
        reb = month_days[-1]
        if i + 1 < len(months):
            nxt_per = months[i + 1]
            nxt_days = prices.index[prices.index.to_period("M") == nxt_per]
            nxt = nxt_days[-1] if len(nxt_days) else prices.index.max()
        else:
            nxt = t1

        if kind == "masi":
            wdict = {}  # bench via masi series directly below
        else:
            wdict = masi20_equal_weights(cours, reb, n=20)

        if kind == "masi":
            daily = masi.reindex(prices.loc[reb:nxt].index).fillna(0.0)
        else:
            daily = portfolio_daily_returns(wdict, prices, reb, nxt)
            cost = turnover_cost(prev_w, wdict, TRANSACTION_COST)
            if not daily.empty and cost > 0:
                daily = daily.copy()
                daily.iloc[0] -= cost
            prev_w = wdict

        if not daily.empty:
            parts.append(daily)

    daily_rets = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    bench = masi.reindex(daily_rets.index).fillna(0.0) if not daily_rets.empty else masi
    perf = (
        compute_backtest_metrics(daily_rets, bench, name=kind)
        if not daily_rets.empty
        else {}
    )
    return {"daily_returns": daily_rets, "perf": perf}


def block_financial_row(
    sim: dict[str, Any],
    *,
    fold_id: int,
    model_id: str,
    groupe: str,
) -> dict[str, Any]:
    perf = sim.get("perf") or {}
    return {
        "fold_id": fold_id,
        "model_id": model_id,
        "groupe": groupe,
        "ann_return": perf.get("annualized_return"),
        "sharpe": perf.get("sharpe"),
        "sortino": perf.get("sortino"),
        "cvar_95_realized": perf.get("cvar_95"),
        "max_drawdown": perf.get("max_drawdown"),
        "turnover_mean": sim.get("turnover_mean"),
        "liquidity_mean": sim.get("liquidity_mean"),
        "total_return_block": sim.get("total_return_block"),
        "n_days": perf.get("days"),
    }


def metrics_to_long_financial(block_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in block_rows.iterrows():
        for met in FINANCIAL_METRICS:
            if met not in r.index:
                continue
            rows.append(
                {
                    "fold_id": int(r["fold_id"]),
                    "model_id": r["model_id"],
                    "groupe": r["groupe"],
                    "metrique": met,
                    "valeur": float(r[met]) if pd.notna(r[met]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def aggregate_summary_by_model(block_rows: pd.DataFrame) -> pd.DataFrame:
    """Moyenne des métriques sur les 14 blocs + vs benchmarks."""
    agg = (
        block_rows.groupby(["model_id", "groupe"], as_index=False)
        .agg(
            ann_return=("ann_return", "mean"),
            sharpe=("sharpe", "mean"),
            sortino=("sortino", "mean"),
            cvar_95_realized=("cvar_95_realized", "mean"),
            max_drawdown=("max_drawdown", "mean"),
            turnover_mean=("turnover_mean", "mean"),
            liquidity_mean=("liquidity_mean", "mean"),
            total_return_block=("total_return_block", "mean"),
            n_blocks=("fold_id", "count"),
        )
        .sort_values("sharpe", ascending=False)
    )
    return agg


def run_portfolio_optimization_block(
    model_id: str,
    fold: Fold,
    *,
    cfg: Stage2Config,
    preds_dir: Path | None = None,
    all_preds_cache: dict[str, pd.DataFrame] | None = None,
    cours: pd.DataFrame | None = None,
    indices: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """NSGA-III pour un (model_id, bloc) — scores isolés, pas de fuite."""
    if cours is None or indices is None:
        cours, indices, _ = load_market_panels()

    all_preds = None
    if all_preds_cache is not None:
        all_preds = all_preds_cache.get(model_id)
    if all_preds is None:
        all_preds = load_all_fold_predictions(model_id, preds_dir=preds_dir)

    stage2 = build_stage2_inputs_for_model_fold(
        model_id,
        fold,
        preds_dir=preds_dir,
        all_preds_for_tau=all_preds,
    )

    block_cfg = replace(
        cfg,
        eval_start_month=fold.test_start_month,
        eval_end_month=fold.test_end_month,
    )

    logger.info(
        "[BlocD] START model=%s groupe=%s fold=%s test=%s→%s",
        model_id,
        MODEL_GROUP.get(model_id),
        fold.fold_id,
        fold.test_start_month,
        fold.test_end_month,
    )

    weights, objectives, meta, pareto_o, pareto_w = run_stage2_model(
        stage2,
        cours,
        indices,
        model_id=model_id,
        cfg=block_cfg,
    )

    meta_df = pd.DataFrame(meta) if isinstance(meta, list) else meta
    n_ok = int((meta_df["status"] == "ok").sum()) if not meta_df.empty else 0
    n_fb = int((meta_df["status"] != "ok").sum()) if not meta_df.empty else 0
    logger.info(
        "[BlocD] DONE model=%s fold=%s | NSGA ok=%s fallback=%s",
        model_id,
        fold.fold_id,
        n_ok,
        n_fb,
    )

    return {
        "model_id": model_id,
        "groupe": MODEL_GROUP.get(model_id, "unknown"),
        "fold": fold,
        "weights": weights,
        "objectives": objectives,
        "meta": meta_df,
        "pareto_objectives": pareto_o,
        "pareto_weights": pareto_w,
    }


def run_portfolio_optimization_all_models(
    model_ids: Sequence[str],
    blocks: list[Fold],
    *,
    cfg: Stage2Config | None = None,
    preds_dir: Path | None = None,
    parts_dir: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """
    Boucle modèle × bloc. Chaque run est indépendant (scores + prev_weights isolés).
    """
    cfg = cfg or Stage2Config(
        liquidity_mode="pareto",
        apply_vmq_filter=False,
        min_vmq=float(LIQUIDITY_VMQ_THRESHOLD_MAD),
        l_min=DEFAULT_L_MIN,
    )
    preds_dir = preds_dir or DEFAULT_PREDS_DIR
    parts_dir = parts_dir or (Path(__file__).resolve().parent / "outputs" / "bloc_d_runs")
    parts_dir.mkdir(parents=True, exist_ok=True)

    cours, indices, technical = load_market_panels()
    prices = _price_panel(cours)
    masi = _masi_returns(indices)

    # Cache preds par modèle (tau expanding — lecture seule, pas de fuite OOS)
    preds_cache = {mid: load_all_fold_predictions(mid, preds_dir=preds_dir) for mid in model_ids}

    all_weights: list[pd.DataFrame] = []
    all_meta: list[pd.DataFrame] = []
    block_fin_rows: list[dict[str, Any]] = []
    bench_rows: list[dict[str, Any]] = []
    long_rows: list[pd.DataFrame] = []

    for fold in blocks:
        # Benchmarks une fois par bloc (indépendants du modèle)
        for bkind in ("masi", "ew_masi20"):
            bsim = simulate_benchmark_block(fold, prices, cours, masi, kind=bkind)
            perf = bsim.get("perf") or {}
            bench_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "benchmark": bkind,
                    "ann_return": perf.get("annualized_return"),
                    "sharpe": perf.get("sharpe"),
                    "total_return_block": float(
                        (1 + bsim["daily_returns"]).prod() - 1
                    )
                    if not bsim["daily_returns"].empty
                    else np.nan,
                }
            )

        for model_id in model_ids:
            ckpt = parts_dir / f"fold_{fold.fold_id:04d}_{model_id}_weights.parquet"
            if resume and ckpt.is_file():
                w = pd.read_parquet(ckpt)
                all_weights.append(w)
                meta_ck = parts_dir / f"fold_{fold.fold_id:04d}_{model_id}_meta.csv"
                if meta_ck.is_file():
                    all_meta.append(pd.read_csv(meta_ck))
                fin_ck = parts_dir / f"fold_{fold.fold_id:04d}_{model_id}_financial.json"
                if fin_ck.is_file():
                    block_fin_rows.append(json.loads(fin_ck.read_text(encoding="utf-8")))
                continue

            try:
                out = run_portfolio_optimization_block(
                    model_id,
                    fold,
                    cfg=cfg,
                    preds_dir=preds_dir,
                    all_preds_cache=preds_cache,
                    cours=cours,
                    indices=indices,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[BlocD] FAIL model=%s fold=%s : %s",
                    model_id,
                    fold.fold_id,
                    exc,
                )
                continue

            w = out["weights"]
            w.to_parquet(ckpt, index=False)
            out["meta"].to_csv(
                parts_dir / f"fold_{fold.fold_id:04d}_{model_id}_meta.csv",
                index=False,
            )
            all_weights.append(w)
            all_meta.append(out["meta"])

            sim = simulate_block_portfolio(
                w,
                prices,
                technical,
                masi,
                model_id=model_id,
                fold_id=fold.fold_id,
            )
            fin = block_financial_row(
                sim,
                fold_id=fold.fold_id,
                model_id=model_id,
                groupe=out["groupe"],
            )
            block_fin_rows.append(fin)
            (parts_dir / f"fold_{fold.fold_id:04d}_{model_id}_financial.json").write_text(
                json.dumps(fin, indent=2, default=str),
                encoding="utf-8",
            )

            # Long : poids + métriques bloc
            sel = w[(w["portfolio"] == PORTFOLIO)].copy()
            for _, row in sel.iterrows():
                long_rows.append(
                    pd.DataFrame(
                        [
                            {
                                "fold_id": fold.fold_id,
                                "model_id": model_id,
                                "groupe": out["groupe"],
                                "metrique": "weight",
                                "ticker": row["ticker"],
                                "mois": row["mois"],
                                "valeur": float(row["weight"]),
                            }
                        ]
                    )
                )

    block_df = pd.DataFrame(block_fin_rows)
    weights_all = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    meta_all = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    results_long = metrics_to_long_financial(block_df)
    summary = aggregate_summary_by_model(block_df) if not block_df.empty else pd.DataFrame()

    # Omnibus financier par groupe (moyenne intra-groupe par bloc)
    omnibus = {
        m: run_omnibus_test(results_long, m) for m in ("sharpe", "ann_return", "cvar_95_realized", "turnover_mean")
    }

    bench_df = pd.DataFrame(bench_rows)
    bench_summary: dict[str, float] = {}
    if not bench_df.empty:
        for bkind in ("masi", "ew_masi20"):
            sub = bench_df[bench_df["benchmark"] == bkind]
            if not sub.empty:
                bench_summary[f"{bkind}_sharpe_mean"] = float(sub["sharpe"].mean())
                bench_summary[f"{bkind}_ann_return_mean"] = float(sub["ann_return"].mean())

    return {
        "weights": weights_all,
        "meta": meta_all,
        "block_financials": block_df,
        "results_long": results_long,
        "summary": summary,
        "benchmarks_by_block": bench_df,
        "benchmark_summary": bench_summary,
        "omnibus_financial": omnibus,
        "parts_dir": parts_dir,
        "model_ids": list(model_ids),
        "n_blocks": len(blocks),
    }
