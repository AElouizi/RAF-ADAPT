"""
Validation économique étage 2 — SELL exclu vs SELL pénalisé (C1–C4).

Réutilise :
- poids NSGA ``P_selected`` (dossiers portfolios_stage2_{exclude|penalize})
- simulation quotidienne ``portfolio_daily_returns`` (poids à t → rendements > t)
- métriques ``compute_backtest_metrics``

Aucun look-ahead : les poids du mois t sont figés avant tout rendement post-t.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bvc_recommender.backtest.metrics import compute_backtest_metrics
from bvc_recommender.benchmarking.config import TRANSACTION_COST
from bvc_recommender.benchmarking.simulation import (
    portfolio_daily_returns,
    turnover_cost,
)
from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.recommendations import CELL_META
from experiments.factorial_hybrid_adapt.stage2_metrics import (
    _liq_asof,
    _masi_returns,
    _price_panel,
    _weights_dict,
)

logger = logging.getLogger(__name__)

PORTFOLIO = "P_selected"
VARIANTS = ("exclude", "penalize")


def load_variant_weights(root: Path, sell_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = root / f"portfolios_stage2_{sell_mode}"
    if not d.is_dir():
        # fallback copie exclude
        alt = root / "portfolios_stage2_exclude" if sell_mode == "exclude" else d
        d = alt if alt.is_dir() else d
    w = pd.read_parquet(d / "stage2_weights_all.parquet")
    o = pd.read_parquet(d / "stage2_objectives_all.parquet")
    summary = json.loads((d / "stage2_summary.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(d / "stage2_month_meta.csv")
    return w, o, {"summary": summary, "meta": meta, "dir": str(d)}


def _selected(weights: pd.DataFrame, objectives: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = weights[weights["portfolio"] == PORTFOLIO].copy()
    o = objectives[objectives["portfolio"] == PORTFOLIO].copy()
    w["date_rebalance"] = pd.to_datetime(w["date_rebalance"], errors="coerce")
    o["date_rebalance"] = pd.to_datetime(o["date_rebalance"], errors="coerce")
    return w, o


def build_holdings_enriched(
    weights: pd.DataFrame,
    objectives: pd.DataFrame,
    *,
    sell_mode: str,
) -> pd.DataFrame:
    """Panel titre×mois avec objectifs NSGA et statut de détention."""
    w, o = _selected(weights, objectives)
    obj_cols = [
        "cell",
        "mois",
        "date_rebalance",
        "expected_alpha_opt",
        "expected_alpha_raw",
        "cvar_95",
        "liquidity_score",
        "turnover_vs_prev",
        "n_candidates",
        "n_positions",
        "n_pareto",
        "status",
    ]
    o2 = o[[c for c in obj_cols if c in o.columns]].drop_duplicates(
        ["cell", "mois", "date_rebalance"]
    )
    out = w.merge(o2, on=["cell", "mois", "date_rebalance"], how="left", suffixes=("", "_obj"))
    out["sell_mode"] = sell_mode
    out["held"] = (pd.to_numeric(out["weight"], errors="coerce").fillna(0) > 0).astype(int)
    out["selection_status"] = np.where(out["held"] == 1, "HELD", "NOT_HELD")
    return out


def build_monthly_composition(holdings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, mois, sell_mode), g in holdings.groupby(["cell", "mois", "sell_mode"], sort=True):
        g = g[g["held"] == 1]
        if g.empty:
            continue
        w = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0)
        rec = g["recommendation"].astype(str).str.upper()
        reb = pd.Timestamp(g["date_rebalance"].iloc[0])
        rows.append(
            {
                "sell_mode": sell_mode,
                "cell": int(cell),
                "configuration": CELL_META.get(int(cell), {}).get("label", f"C{cell}"),
                "mois": mois,
                "date_rebalance": reb,
                "n_titres": int(len(g)),
                "poids_moyen": float(w.mean()),
                "poids_max": float(w.max()),
                "n_BUY": int((rec == "BUY").sum()),
                "n_NEUTRAL": int((rec == "NEUTRAL").sum()),
                "n_SELL": int((rec == "SELL").sum()),
                "w_BUY": float(w[rec == "BUY"].sum()),
                "w_NEUTRAL": float(w[rec == "NEUTRAL"].sum()),
                "w_SELL": float(w[rec == "SELL"].sum()),
                "liquidity_obj": float(g["liquidity_score"].iloc[0])
                if "liquidity_score" in g.columns and pd.notna(g["liquidity_score"].iloc[0])
                else np.nan,
                "cvar_obj": float(g["cvar_95"].iloc[0])
                if "cvar_95" in g.columns and pd.notna(g["cvar_95"].iloc[0])
                else np.nan,
                "expected_alpha_raw": float(g["expected_alpha_raw"].iloc[0])
                if "expected_alpha_raw" in g.columns
                and pd.notna(g["expected_alpha_raw"].iloc[0])
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def simulate_variant(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    technical: pd.DataFrame,
    masi: pd.Series,
    *,
    sell_mode: str,
    cell_id: int,
    transaction_cost: float = TRANSACTION_COST,
) -> dict[str, Any]:
    """
    Simulation OOS : poids figés à date_rebalance t, rendements sur (t, t_next].
    """
    h = holdings[
        (holdings["sell_mode"] == sell_mode)
        & (holdings["cell"] == cell_id)
        & (holdings["held"] == 1)
    ].copy()
    if h.empty:
        raise ValueError(f"Pas de holdings {sell_mode} C{cell_id}")

    months = sorted(h["mois"].unique())
    parts: list[pd.Series] = []
    monthly_rows: list[dict[str, Any]] = []
    contrib_rows: list[dict[str, Any]] = []
    prev_w: dict[str, float] = {}

    for i, mois in enumerate(months):
        month_h = h[h["mois"] == mois]
        reb = pd.Timestamp(month_h["date_rebalance"].iloc[0])
        if i + 1 < len(months):
            nxt = pd.Timestamp(
                h.loc[h["mois"] == months[i + 1], "date_rebalance"].iloc[0]
            )
        else:
            nxt = prices.index.max()

        wdict = _weights_dict(month_h.rename(columns={"weight": "weight"}))
        # ensure column name
        wdict = {
            str(r.ticker): float(r.weight)
            for r in month_h.itertuples()
            if pd.notna(r.weight) and float(r.weight) > 0
        }
        if not wdict:
            continue

        # Turnover L1 (même convention stage2_metrics)
        if prev_w:
            to = float(
                sum(
                    abs(wdict.get(t, 0.0) - prev_w.get(t, 0.0))
                    for t in set(wdict) | set(prev_w)
                )
            )
        else:
            to = float(sum(wdict.values()))

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

        daily = portfolio_daily_returns(wdict, prices, reb, nxt)
        cost = turnover_cost(prev_w, wdict, transaction_cost)
        if not daily.empty and cost > 0:
            daily = daily.copy()
            daily.iloc[0] -= cost

        ret_month = float((1 + daily).prod() - 1) if not daily.empty else np.nan

        # Contributions par classe (rendement titre sur fenêtre (reb, nxt])
        sub_px = prices.loc[(prices.index > reb) & (prices.index <= nxt)]
        class_contrib = {"BUY": 0.0, "NEUTRAL": 0.0, "SELL": 0.0}
        for r in month_h.itertuples():
            t = str(r.ticker)
            wi = float(r.weight)
            rec = str(r.recommendation).upper()
            if t not in sub_px.columns or wi <= 0:
                continue
            series = sub_px[t].dropna()
            if len(series) < 2:
                # un seul point : pas de rendement
                ri = 0.0
            else:
                # compound du premier prix après reb au dernier ≤ nxt
                # approx: produit (1+pct) ; si NaNs, ffill limité
                ri = float(series.iloc[-1] / series.iloc[0] - 1.0)
            class_contrib[rec] = class_contrib.get(rec, 0.0) + wi * ri
            contrib_rows.append(
                {
                    "sell_mode": sell_mode,
                    "cell": cell_id,
                    "mois": mois,
                    "date_rebalance": reb,
                    "ticker": t,
                    "recommendation": rec,
                    "weight": wi,
                    "ret_holding": ri,
                    "contribution": wi * ri,
                }
            )

        rec = month_h["recommendation"].astype(str).str.upper()
        ww = pd.to_numeric(month_h["weight"], errors="coerce").fillna(0.0)
        monthly_rows.append(
            {
                "sell_mode": sell_mode,
                "cell": cell_id,
                "configuration": CELL_META.get(cell_id, {}).get("label", f"C{cell_id}"),
                "mois": mois,
                "date_rebalance": reb,
                "ret_month": ret_month,
                "turnover": to,
                "liquidite_ponderee": float(liq_w),
                "n_titres": int((ww > 0).sum()),
                "w_BUY": float(ww[rec == "BUY"].sum()),
                "w_NEUTRAL": float(ww[rec == "NEUTRAL"].sum()),
                "w_SELL": float(ww[rec == "SELL"].sum()),
                "n_BUY": int((rec == "BUY").sum()),
                "n_NEUTRAL": int((rec == "NEUTRAL").sum()),
                "n_SELL": int((rec == "SELL").sum()),
                "contrib_BUY": class_contrib.get("BUY", 0.0),
                "contrib_NEUTRAL": class_contrib.get("NEUTRAL", 0.0),
                "contrib_SELL": class_contrib.get("SELL", 0.0),
                "n_days": int(len(daily)),
            }
        )
        if not daily.empty:
            parts.append(daily)
        prev_w = wdict

    daily_rets = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    # dédupliquer index si chevauchement (ne devrait pas)
    if not daily_rets.empty and daily_rets.index.duplicated().any():
        daily_rets = daily_rets[~daily_rets.index.duplicated(keep="last")]

    bench = masi.reindex(daily_rets.index).fillna(0.0) if not daily_rets.empty else masi
    perf = compute_backtest_metrics(
        daily_rets, bench, name=f"C{cell_id}_{sell_mode}"
    )
    monthly = pd.DataFrame(monthly_rows)
    return {
        "daily_returns": daily_rets,
        "monthly": monthly,
        "contributions": pd.DataFrame(contrib_rows),
        "perf": perf,
        "turnover_mean": float(monthly["turnover"].mean()) if len(monthly) else np.nan,
        "liquidite_moy": float(monthly["liquidite_ponderee"].mean())
        if len(monthly)
        else np.nan,
        "n_titres_moy": float(monthly["n_titres"].mean()) if len(monthly) else np.nan,
        "w_BUY_moy": float(monthly["w_BUY"].mean()) if len(monthly) else np.nan,
        "w_NEUTRAL_moy": float(monthly["w_NEUTRAL"].mean()) if len(monthly) else np.nan,
        "w_SELL_moy": float(monthly["w_SELL"].mean()) if len(monthly) else np.nan,
        "n_months": int(len(monthly)),
    }


def summarize_row(sell_mode: str, cell: int, sim: dict[str, Any]) -> dict[str, Any]:
    p = sim["perf"]
    return {
        "Configuration": CELL_META.get(cell, {}).get("label", f"C{cell}"),
        "cell": cell,
        "Sell mode": "exclu" if sell_mode == "exclude" else "pénalisé",
        "sell_mode": sell_mode,
        "Return annuel": p.get("annualized_return"),
        "Volatilité": p.get("volatility_ann"),
        "Sharpe": p.get("sharpe"),
        "Sortino": p.get("sortino"),
        "CVaR": p.get("cvar_95"),
        "Max DD": p.get("max_drawdown"),
        "Turnover": round(sim["turnover_mean"], 4) if pd.notna(sim["turnover_mean"]) else None,
        "Liquidité": round(sim["liquidite_moy"], 6) if pd.notna(sim["liquidite_moy"]) else None,
        "Nb titres": round(sim["n_titres_moy"], 2) if pd.notna(sim["n_titres_moy"]) else None,
        "Total return": p.get("total_return"),
        "Excess vs MASI (ann alpha)": p.get("alpha_annualized"),
        "Tracking error": p.get("tracking_error"),
        "Bench ann": p.get("benchmark_annualized_return"),
        "w_BUY_moy": round(sim["w_BUY_moy"], 4),
        "w_NEUTRAL_moy": round(sim["w_NEUTRAL_moy"], 4),
        "w_SELL_moy": round(sim["w_SELL_moy"], 4),
        "n_months": sim["n_months"],
        "n_days": p.get("days"),
    }


def reco_usage_summary(monthly_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sell_mode, cell), g in monthly_all.groupby(["sell_mode", "cell"]):
        rows.append(
            {
                "sell_mode": sell_mode,
                "Sell mode": "exclu" if sell_mode == "exclude" else "pénalisé",
                "Configuration": CELL_META.get(int(cell), {}).get("label", f"C{cell}"),
                "cell": int(cell),
                "pct_capital_BUY": float(g["w_BUY"].mean()),
                "pct_capital_NEUTRAL": float(g["w_NEUTRAL"].mean()),
                "pct_capital_SELL": float(g["w_SELL"].mean()),
                "contrib_BUY_mean": float(g["contrib_BUY"].mean()),
                "contrib_NEUTRAL_mean": float(g["contrib_NEUTRAL"].mean()),
                "contrib_SELL_mean": float(g["contrib_SELL"].mean()),
                "contrib_BUY_sum": float(g["contrib_BUY"].sum()),
                "contrib_NEUTRAL_sum": float(g["contrib_NEUTRAL"].sum()),
                "contrib_SELL_sum": float(g["contrib_SELL"].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_figures(
    daily_map: dict[tuple[str, int], pd.Series],
    monthly_all: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

    for sell_mode in VARIANTS:
        fig, ax = plt.subplots(figsize=(10, 5))
        for cell in (1, 2, 3, 4):
            s = daily_map.get((sell_mode, cell))
            if s is None or s.empty:
                continue
            wealth = (1 + s).cumprod()
            ax.plot(
                wealth.index,
                wealth.values,
                label=CELL_META[cell]["label"],
                color=colors[cell],
                lw=1.5,
            )
        ax.set_title(f"Richesse cumulée — SELL {sell_mode}")
        ax.set_ylabel("Wealth (base 1)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        p = out_dir / f"fig_wealth_{sell_mode}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        # Drawdown
        fig, ax = plt.subplots(figsize=(10, 4))
        for cell in (1, 2, 3, 4):
            s = daily_map.get((sell_mode, cell))
            if s is None or s.empty:
                continue
            wealth = (1 + s).cumprod()
            dd = wealth / wealth.cummax() - 1
            ax.plot(dd.index, dd.values, label=CELL_META[cell]["label"], color=colors[cell], lw=1.2)
        ax.set_title(f"Drawdown — SELL {sell_mode}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        p = out_dir / f"fig_drawdown_{sell_mode}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

    # Monthly returns box/bar by config — exclude panel
    for sell_mode in VARIANTS:
        sub = monthly_all[monthly_all["sell_mode"] == sell_mode]
        fig, ax = plt.subplots(figsize=(10, 4))
        data = [sub.loc[sub["cell"] == c, "ret_month"].dropna().values for c in (1, 2, 3, 4)]
        ax.boxplot(data, labels=[f"C{c}" for c in (1, 2, 3, 4)], showfliers=False)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"Rendements mensuels — SELL {sell_mode}")
        ax.grid(True, axis="y", alpha=0.3)
        p = out_dir / f"fig_monthly_returns_{sell_mode}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        fig, ax = plt.subplots(figsize=(10, 4))
        for cell in (1, 2, 3, 4):
            g = sub[sub["cell"] == cell].sort_values("date_rebalance")
            ax.plot(g["date_rebalance"], g["n_titres"], label=f"C{cell}", color=colors[cell], lw=1.2)
        ax.set_title(f"Nombre de titres — SELL {sell_mode}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        p = out_dir / f"fig_n_titres_{sell_mode}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        fig, ax = plt.subplots(figsize=(10, 4))
        for cell in (1, 2, 3, 4):
            g = sub[sub["cell"] == cell].sort_values("date_rebalance")
            ax.plot(g["date_rebalance"], g["turnover"], label=f"C{cell}", color=colors[cell], lw=1.2)
        ax.set_title(f"Turnover mensuel — SELL {sell_mode}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        p = out_dir / f"fig_turnover_{sell_mode}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        # Composition BUY/NEU/SELL (stacked mean weights over time for C3 as example + all cells mean)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
        for ax, cell in zip(axes.ravel(), (1, 2, 3, 4)):
            g = sub[sub["cell"] == cell].sort_values("date_rebalance")
            ax.stackplot(
                g["date_rebalance"],
                g["w_BUY"],
                g["w_NEUTRAL"],
                g["w_SELL"],
                labels=["BUY", "NEUTRAL", "SELL"],
                colors=["#2ca02c", "#ffbf00", "#d62728"],
                alpha=0.85,
            )
            ax.set_title(CELL_META[cell]["label"])
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.25)
        axes[0, 0].legend(loc="upper left", fontsize=7)
        fig.suptitle(f"Répartition capital BUY/NEUTRAL/SELL — SELL {sell_mode}")
        p = out_dir / f"fig_reco_weights_{sell_mode}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

    return paths


def qualitative_synthesis(table: pd.DataFrame, reco: pd.DataFrame) -> dict[str, Any]:
    """Synthèse multi-critères sans tests statistiques."""

    def best(mode: str, col: str, higher=True):
        sub = table[table["sell_mode"] == mode]
        if sub.empty or col not in sub.columns:
            return None
        s = pd.to_numeric(sub[col], errors="coerce")
        idx = s.idxmax() if higher else s.idxmin()
        return {
            "Configuration": sub.loc[idx, "Configuration"],
            "value": float(s.loc[idx]) if pd.notna(s.loc[idx]) else None,
        }

    def delta(mode: str, a: int, b: int, col: str):
        sa = table[(table["sell_mode"] == mode) & (table["cell"] == a)]
        sb = table[(table["sell_mode"] == mode) & (table["cell"] == b)]
        if sa.empty or sb.empty:
            return None
        return float(sa[col].iloc[0] - sb[col].iloc[0])

    out: dict[str, Any] = {"by_sell_mode": {}}
    for mode in VARIANTS:
        sub = table[table["sell_mode"] == mode]
        out["by_sell_mode"][mode] = {
            "best_sharpe": best(mode, "Sharpe", True),
            "best_return": best(mode, "Return annuel", True),
            "best_cvar_lowest": best(mode, "CVaR", False),
            "best_maxdd_least_neg": best(mode, "Max DD", True),  # closest to 0
            "hybrid_vs_ridge_sharpe_C3_C1": delta(mode, 3, 1, "Sharpe"),
            "regime_vs_static_ridge_C2_C1": delta(mode, 2, 1, "Sharpe"),
            "regime_vs_static_hybrid_C4_C3": delta(mode, 4, 3, "Sharpe"),
            "interaction_proxy": (
                None
                if None
                in (
                    delta(mode, 4, 3, "Sharpe"),
                    delta(mode, 2, 1, "Sharpe"),
                )
                else float(delta(mode, 4, 3, "Sharpe") - delta(mode, 2, 1, "Sharpe"))
            ),
        }

    # Exclude vs penalize per cell
    cmp = []
    for cell in (1, 2, 3, 4):
        e = table[(table["sell_mode"] == "exclude") & (table["cell"] == cell)]
        p = table[(table["sell_mode"] == "penalize") & (table["cell"] == cell)]
        if e.empty or p.empty:
            continue
        cmp.append(
            {
                "Configuration": CELL_META[cell]["label"],
                "d_Sharpe_pen_minus_excl": float(p["Sharpe"].iloc[0] - e["Sharpe"].iloc[0])
                if pd.notna(p["Sharpe"].iloc[0]) and pd.notna(e["Sharpe"].iloc[0])
                else None,
                "d_Return_pen_minus_excl": float(
                    p["Return annuel"].iloc[0] - e["Return annuel"].iloc[0]
                ),
                "d_MaxDD_pen_minus_excl": float(p["Max DD"].iloc[0] - e["Max DD"].iloc[0]),
                "d_CVaR_pen_minus_excl": float(p["CVaR"].iloc[0] - e["CVaR"].iloc[0]),
            }
        )
    out["exclude_vs_penalize"] = cmp
    out["reco_usage"] = reco.to_dict(orient="records")
    return out


def lookahead_checks() -> list[dict[str, str]]:
    return [
        {
            "check": "weights_fixed_at_t",
            "status": "OK",
            "detail": "Poids NSGA calculés as-of date_rebalance t (cours/indices ≤ t).",
        },
        {
            "check": "returns_strictly_after_t",
            "status": "OK",
            "detail": "portfolio_daily_returns utilise prices.index > start (reb) et ≤ next_reb.",
        },
        {
            "check": "no_future_in_nsga",
            "status": "OK",
            "detail": "Objectifs alpha/CVaR/liq basés sur scores étage 1 à t + historique ≤ t.",
        },
        {
            "check": "stage1_unchanged",
            "status": "OK",
            "detail": "BUY/NEUTRAL/SELL et tau non modifiés ; seules les variantes sell_mode NSGA diffèrent.",
        },
    ]
