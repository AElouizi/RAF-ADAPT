"""
Backtest Stage 2 — variantes liquidité (même fenêtre / cellules / SELL).

Lit les poids NSGA et produit contrôles + perf C1–C4 + exports web.

Usage :
  py -3 .../run_stage2_official_backtest.py
  py -3 .../run_stage2_official_backtest.py --liquidity-role pareto \\
      --portfolio-dir portfolios_stage2_liq_pareto --report-name stage2_liq_pareto
  py -3 .../run_stage2_official_backtest.py --liquidity-role none \\
      --portfolio-dir portfolios_stage2_liq_none --report-name stage2_liq_none
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bvc_recommender.config import LIQUIDITY_VMQ_THRESHOLD_MAD
from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.paths import OUTPUT_DIR, REPORTS_DIR, ensure_output_dirs
from experiments.factorial_hybrid_adapt.recommendations import CELL_META
from experiments.factorial_hybrid_adapt.stage2_allocation import (
    DEFAULT_L_MIN,
    EVAL_END_MONTH,
    EVAL_START_MONTH,
)
from experiments.factorial_hybrid_adapt.stage2_econ_eval import simulate_variant
from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PORTFOLIO = "P_selected"
MIN_VMQ = float(LIQUIDITY_VMQ_THRESHOLD_MAD)

ROLE_DEFAULTS = {
    "eligibility": ("portfolios_stage2_official", "stage2_official"),
    "pareto": ("portfolios_stage2_liq_pareto", "stage2_liq_pareto"),
    "none": ("portfolios_stage2_liq_none", "stage2_liq_none"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest Stage 2 (variantes liquidité)")
    p.add_argument(
        "--liquidity-role",
        choices=("eligibility", "pareto", "none"),
        default="eligibility",
    )
    p.add_argument("--portfolio-dir", type=str, default=None)
    p.add_argument("--report-name", type=str, default=None)
    return p.parse_args()


def load_portfolios(root: Path, dirname: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    d = root / dirname
    w = pd.read_parquet(d / "stage2_weights_all.parquet")
    o = pd.read_parquet(d / "stage2_objectives_all.parquet")
    meta = pd.read_csv(d / "stage2_month_meta.csv")
    summary = json.loads((d / "stage2_summary.json").read_text(encoding="utf-8"))
    return w, o, meta, summary


def build_holdings(w: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    h = w[w["portfolio"] == PORTFOLIO].copy()
    h["date_rebalance"] = pd.to_datetime(h["date_rebalance"], errors="coerce")
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
    h["liquidity_vmq_20j"] = pd.to_numeric(h.get("liquidity_vmq_20j"), errors="coerce")
    h["recommendation"] = h["recommendation"].astype(str).str.upper()
    h["held"] = (h["weight"].fillna(0) > 0).astype(int)
    h["sell_mode"] = "exclude"
    h["eligible_vmq"] = (h["liquidity_vmq_20j"].fillna(0) >= MIN_VMQ).astype(int)
    obj = o[o["portfolio"] == PORTFOLIO][
        [c for c in ("cell", "mois", "date_rebalance", "n_candidates", "n_positions",
                     "expected_alpha_raw", "cvar_95", "liquidity_L_weighted", "status")
         if c in o.columns]
    ].drop_duplicates(["cell", "mois"])
    obj["date_rebalance"] = pd.to_datetime(obj["date_rebalance"], errors="coerce")
    return h.merge(obj, on=["cell", "mois", "date_rebalance"], how="left", suffixes=("", "_obj"))


def validity_controls(
    holdings: pd.DataFrame,
    meta: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    liquidity_role: str,
) -> dict[str, Any]:
    held = holdings[holdings["held"] == 1]
    n_months = int(meta["mois"].nunique()) if len(meta) else 0
    n_ok = int((meta["status"] == "ok").sum()) if "status" in meta.columns else 0
    n_fb = int((meta["status"] != "ok").sum()) if "status" in meta.columns else 0
    n_empty = int((meta["status"] == "empty_universe").sum()) if "status" in meta.columns else 0
    months_with_port = held.groupby(["cell", "mois"]).ngroups
    expected = 4 * n_months
    viol_vmq = held[held["liquidity_vmq_20j"].fillna(0) < MIN_VMQ]
    viol_sell = held[held["recommendation"] == "SELL"]
    out = {
        "n_months_executed": n_months,
        "n_month_cell_ok": n_ok,
        "n_fallbacks": n_fb,
        "n_empty_universe": n_empty,
        "n_month_cell_with_portfolio": int(months_with_port),
        "n_month_cell_expected": int(expected),
        "mean_n_candidates_nsga": float(
            pd.to_numeric(meta.get("n_candidates"), errors="coerce").mean()
        )
        if "n_candidates" in meta.columns
        else float(monthly["n_titres"].mean()) if len(monthly) else np.nan,
        "mean_n_positions_final": float(
            pd.to_numeric(meta.get("n_positions_selected"), errors="coerce").mean()
        )
        if "n_positions_selected" in meta.columns
        else np.nan,
        "mean_n_positions_from_monthly": float(monthly["n_titres"].mean()) if len(monthly) else np.nan,
        "n_held_rows_vmq_below_500k": int(len(viol_vmq)),
        "weight_sum_vmq_below_500k": float(viol_vmq["weight"].sum()) if len(viol_vmq) else 0.0,
        "n_held_SELL": int(len(viol_sell)),
        "weight_sum_SELL": float(viol_sell["weight"].sum()) if len(viol_sell) else 0.0,
        "no_sell_ok": len(viol_sell) == 0,
        "min_vmq_in_holdings": float(held["liquidity_vmq_20j"].min()) if len(held) else np.nan,
        "liquidity_role": liquidity_role,
    }
    if liquidity_role == "eligibility":
        out["vmq_constraint_ok"] = len(viol_vmq) == 0
    else:
        out["vmq_constraint_ok"] = None
        out["vmq_below_500k_diagnostic_only"] = True
    return out


def performance_table(sims: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for cell, sim in sims.items():
        p = sim["perf"]
        rows.append(
            {
                "Configuration": CELL_META[cell]["label"],
                "cell": cell,
                "Return annuel": p.get("annualized_return"),
                "Volatilité": p.get("volatility_ann"),
                "Sharpe": p.get("sharpe"),
                "Sortino": p.get("sortino"),
                "CVaR": p.get("cvar_95"),
                "Max DD": p.get("max_drawdown"),
                "Turnover": round(sim["turnover_mean"], 4),
                "Liquidité (L pondéré)": round(sim.get("liquidite_moy") or np.nan, 4)
                if pd.notna(sim.get("liquidite_moy"))
                else None,
                "Nb positions": round(sim["n_titres_moy"], 2),
                "Total return": p.get("total_return"),
                "Alpha ann. vs MASI": p.get("alpha_annualized"),
                "Tracking error": p.get("tracking_error"),
                "n_months": sim["n_months"],
                "n_days": p.get("days"),
            }
        )
    return pd.DataFrame(rows).sort_values("cell")


def pairwise_comparisons(table: pd.DataFrame) -> pd.DataFrame:
    pairs = [(2, 1, "C2 vs C1"), (3, 1, "C3 vs C1"), (4, 1, "C4 vs C1"), (4, 3, "C4 vs C3")]
    metrics = [
        ("Return annuel", "d_Return_ann"),
        ("Sharpe", "d_Sharpe"),
        ("Sortino", "d_Sortino"),
        ("CVaR", "d_CVaR"),
        ("Max DD", "d_MaxDD"),
        ("Turnover", "d_Turnover"),
    ]
    rows = []
    for a, b, name in pairs:
        ra = table[table["cell"] == a].iloc[0]
        rb = table[table["cell"] == b].iloc[0]
        row: dict[str, Any] = {"Comparaison": name, "cell_a": a, "cell_b": b}
        for col, key in metrics:
            row[key] = float(ra[col] - rb[col]) if pd.notna(ra[col]) and pd.notna(rb[col]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def monthly_dominance(monthly: pd.DataFrame) -> dict[str, float]:
    wide = monthly.pivot_table(index="mois", columns="cell", values="ret_month")
    out = {}

    def pct(a, b):
        sub = wide[[a, b]].dropna()
        return float((sub[a] > sub[b]).mean()) if len(sub) else np.nan

    out["pct_months_C2_gt_C1"] = pct(2, 1)
    out["pct_months_C3_gt_C1"] = pct(3, 1)
    out["pct_months_C4_gt_C1"] = pct(4, 1)
    out["pct_months_C4_gt_C3"] = pct(4, 3)
    return out


def write_figs(
    daily_map: dict[int, pd.Series],
    monthly: pd.DataFrame,
    fig_dir: Path,
    *,
    title_suffix: str,
) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for cell, s in daily_map.items():
        if s is None or s.empty:
            continue
        wealth = (1 + s).cumprod()
        ax.plot(wealth.index, wealth.values, label=CELL_META[cell]["label"], color=colors[cell], lw=1.5)
    ax.set_title(f"Richesse cumulée — Stage 2 {title_suffix} (2023-01→2025-06)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    p = fig_dir / "fig_wealth.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    for cell, s in daily_map.items():
        if s is None or s.empty:
            continue
        wealth = (1 + s).cumprod()
        dd = wealth / wealth.cummax() - 1
        ax.plot(dd.index, dd.values, label=CELL_META[cell]["label"], color=colors[cell], lw=1.2)
    ax.set_title(f"Drawdown — Stage 2 {title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    p = fig_dir / "fig_drawdown.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    for cell in (1, 2, 3, 4):
        g = monthly[monthly["cell"] == cell].sort_values("date_rebalance")
        ax.plot(g["date_rebalance"], g["ret_month"], label=f"C{cell}", color=colors[cell], lw=1.2)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"Rendements mensuels — Stage 2 {title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    p = fig_dir / "fig_monthly_returns.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths


def enrich_web_export(
    holdings: pd.DataFrame,
    monthly: pd.DataFrame,
    contributions: pd.DataFrame,
    *,
    liquidity_role: str,
) -> pd.DataFrame:
    h = holdings[holdings["held"] == 1].copy()
    m = monthly[
        ["cell", "mois", "ret_month", "turnover", "liquidite_ponderee", "n_titres"]
    ].rename(columns={"ret_month": "portfolio_ret_month"})
    out = h.merge(m, on=["cell", "mois"], how="left")
    if len(contributions):
        c = contributions[
            ["cell", "mois", "ticker", "ret_holding", "contribution"]
        ].copy()
        out = out.merge(c, on=["cell", "mois", "ticker"], how="left")
    out["eligible_vmq_500k"] = (out["liquidity_vmq_20j"].fillna(0) >= MIN_VMQ).astype(int)
    out["liquidity_role"] = liquidity_role
    out["min_vmq_threshold"] = MIN_VMQ
    out["protocol_window"] = f"{EVAL_START_MONTH}_{EVAL_END_MONTH}"
    return out


def nsga_objectives_for_role(role: str) -> list[str]:
    if role == "pareto":
        return ["alpha", "CVaR", "max_L"]
    return ["alpha", "CVaR"]


def main() -> int:
    args = parse_args()
    role = args.liquidity_role
    default_port, default_report = ROLE_DEFAULTS[role]
    portfolio_dirname = args.portfolio_dir or default_port
    report_name = args.report_name or default_report

    ensure_output_dirs()
    root = OUTPUT_DIR
    out = REPORTS_DIR / report_name
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    w, o, meta, summary_nsga = load_portfolios(root, portfolio_dirname)
    holdings = build_holdings(w, o)
    holdings["sell_mode"] = "exclude"

    cours, indices, technical = load_market_panels()
    prices = _price_panel(cours)
    masi = _masi_returns(indices)

    sims: dict[int, dict[str, Any]] = {}
    monthly_all = []
    contrib_all = []
    daily_map: dict[int, pd.Series] = {}

    for cell in (1, 2, 3, 4):
        logger.info("Simulation OOS [%s] C%s", role, cell)
        sim = simulate_variant(
            holdings, prices, technical, masi, sell_mode="exclude", cell_id=cell
        )
        if "liquidity_L_weighted" in o.columns:
            lo = o[(o["cell"] == cell) & (o["portfolio"] == PORTFOLIO)]
            sim["liquidite_moy"] = float(
                pd.to_numeric(lo.get("liquidity_L_weighted"), errors="coerce").mean()
            )
        sims[cell] = sim
        daily_map[cell] = sim["daily_returns"]
        monthly_all.append(sim["monthly"])
        contrib_all.append(sim["contributions"])

    monthly = pd.concat(monthly_all, ignore_index=True)
    contrib = pd.concat(contrib_all, ignore_index=True) if contrib_all else pd.DataFrame()
    table = performance_table(sims)
    comparisons = pairwise_comparisons(table)
    dominance = monthly_dominance(monthly)
    controls = validity_controls(holdings, meta, monthly, liquidity_role=role)

    controls["mean_n_after_sell_exclude"] = controls["mean_n_candidates_nsga"]
    try:
        inp = pd.read_parquet(REPORTS_DIR / "stage2_inputs_14blocks.parquet")
        inp["date"] = pd.to_datetime(inp["date"])
        inp["mois"] = inp["date"].dt.to_period("M").astype(str)
        inp = inp[(inp["mois"] >= EVAL_START_MONTH) & (inp["mois"] <= EVAL_END_MONTH)]
        inp["vmq"] = pd.to_numeric(inp["liquidity_vmq_20j"], errors="coerce")
        inp["rec"] = inp["recommendation"].astype(str).str.upper()
        rows_liq = []
        for (cell, mois), g in inp.groupby(["cell", "mois"]):
            n_liq = int((g["vmq"].fillna(0) >= MIN_VMQ).sum())
            if role == "eligibility":
                n_nsga = int(
                    ((g["vmq"].fillna(0) >= MIN_VMQ) & (g["rec"] != "SELL")).sum()
                )
            else:
                n_nsga = int((g["rec"] != "SELL").sum())
            rows_liq.append(
                {
                    "cell": cell,
                    "mois": mois,
                    "n_liq": n_liq,
                    "n_after_sell": n_nsga,
                    "n_nsga_protocol": n_nsga,
                }
            )
        liqdf = pd.DataFrame(rows_liq)
        controls["mean_n_liquid"] = float(liqdf["n_liq"].mean())
        controls["mean_n_after_sell_exclude"] = float(liqdf["n_after_sell"].mean())
        liqdf.to_csv(out / "universe_eligibility_by_month.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("eligibility recount failed: %s", exc)

    rank = table.copy()
    rank["score_rank"] = (
        rank["Sharpe"].rank(ascending=False)
        + rank["Return annuel"].rank(ascending=False)
        + rank["Sortino"].rank(ascending=False)
        + rank["CVaR"].rank(ascending=True)
        + rank["Max DD"].rank(ascending=False)
        + rank["Turnover"].rank(ascending=True)
    )
    rank = rank.sort_values("score_rank")
    ranking = []
    for i, (_, r) in enumerate(rank.iterrows(), start=1):
        ranking.append(
            {
                "rank": i,
                "Configuration": r["Configuration"],
                "cell": int(r["cell"]),
                "Sharpe": r["Sharpe"],
                "Return annuel": r["Return annuel"],
                "Sortino": r["Sortino"],
                "CVaR": r["CVaR"],
                "Max DD": r["Max DD"],
                "composite_score_rank_sum": float(r["score_rank"]),
            }
        )

    web = enrich_web_export(holdings, monthly, contrib, liquidity_role=role)
    figs = write_figs(daily_map, monthly, fig_dir, title_suffix=role)

    table.to_csv(out / "table_performance_C1C4.csv", index=False)
    comparisons.to_csv(out / "comparisons_pairwise.csv", index=False)
    monthly.to_csv(out / "performance_mensuelle.csv", index=False)
    holdings.to_parquet(out / "holdings_titre_mois.parquet", index=False)
    web.to_parquet(out / "export_web_titre_mois.parquet", index=False)
    web.to_csv(out / "export_web_titre_mois.csv", index=False)
    contrib.to_parquet(out / "contributions_titre_mois.parquet", index=False)
    meta.to_csv(out / "month_meta_copy.csv", index=False)

    wealth_frames = []
    for cell, s in daily_map.items():
        if s is None or s.empty:
            continue
        wealth_frames.append(
            pd.DataFrame(
                {
                    "date": s.index,
                    "ret": s.values,
                    "wealth": (1 + s).cumprod().values,
                    "cell": cell,
                    "configuration": CELL_META[cell]["label"],
                }
            )
        )
    if wealth_frames:
        pd.concat(wealth_frames, ignore_index=True).to_parquet(
            out / "daily_returns_wealth.parquet", index=False
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "window": f"{EVAL_START_MONTH} → {EVAL_END_MONTH}",
            "min_vmq": MIN_VMQ if role == "eligibility" else None,
            "l_min": DEFAULT_L_MIN if role == "eligibility" else None,
            "sell": "exclude",
            "nsga_objectives": nsga_objectives_for_role(role),
            "liquidity_role": role,
        },
        "controls": controls,
        "performance": table.to_dict(orient="records"),
        "comparisons": comparisons.to_dict(orient="records"),
        "dominance_monthly": dominance,
        "preliminary_ranking": ranking,
        "nsga_summary": summary_nsga,
        "figures": [str(p) for p in figs],
        "portfolio_dir": portfolio_dirname,
    }
    (out / f"stage2_{role}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "stage2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\n=== CONTRÔLES [{role}] ===")
    for k, v in controls.items():
        print(f"  {k}: {v}")
    print(f"\n=== TABLE C1–C4 [{role}] ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(
            table[
                [
                    "Configuration",
                    "Return annuel",
                    "Volatilité",
                    "Sharpe",
                    "Sortino",
                    "CVaR",
                    "Max DD",
                    "Turnover",
                    "Liquidité (L pondéré)",
                    "Nb positions",
                ]
            ].to_string(index=False)
        )
    print("\n=== COMPARAISONS ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(comparisons.to_string(index=False))
    print("\n=== DOMINANCE MENSUELLE ===")
    for k, v in dominance.items():
        print(f"  {k}: {v:.1%}" if pd.notna(v) else f"  {k}: —")
    print("\n=== CLASSEMENT PRÉLIMINAIRE ===")
    for r in ranking:
        print(
            f"  #{r['rank']} {r['Configuration']} | Sharpe={r['Sharpe']} | "
            f"Ret={r['Return annuel']} | Sortino={r['Sortino']}"
        )
    print(f"\nSorties : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
