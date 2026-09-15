"""
Backtest FINAL C4 — portefeuilles figés (knee), convention projet.

Ne modifie PAS Stage 1 / Stage 2 / poids / hyperparamètres / règle knee.

Convention de date (anti look-ahead) :
  - À la date_rebalance t (dernier jour de bourse du mois de décision),
    les poids C4 sont déjà déterminés avec données ≤ t uniquement.
  - Ces poids sont appliqués aux rendements réalisés sur l'intervalle ouvert
    (t , t_next ] où t_next = date_rebalance du mois suivant
    (dernier mois : jusqu'au dernier prix disponible).
  - Les frais de transaction sont débités le 1er jour de la fenêtre de détention
    (convention ``turnover_cost`` du projet).

Coût de transaction (projet) :
  TRANSACTION_COST = 0.003
  cost = Σ_t |w_new(t) − w_old(t)| × 0.003
  (turnover L1 bilatéral ; entrée initiale : Σ w = 1 → cost = 0.3 %).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bvc_recommender.backtest.metrics import compute_backtest_metrics
from bvc_recommender.benchmarking.config import TRANSACTION_COST
from bvc_recommender.benchmarking.simulation import (
    portfolio_daily_returns,
    turnover_cost,
)
from experiments.factorial_hybrid_adapt.stage2_allocation import (
    EVAL_END_MONTH,
    EVAL_START_MONTH,
)
from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel

BASE = Path(__file__).resolve().parent
HOLDINGS_PATH = BASE / "outputs" / "portfolios_C4_final" / "C4_final_holdings_titre_mois.parquet"
OUT = BASE / "outputs" / "reports" / "stage2_C4_final_backtest"
FIG = OUT / "figures"


def max_dd_duration_days(wealth: pd.Series) -> int:
    peak = wealth.cummax()
    under = wealth < peak
    if not under.any():
        return 0
    groups = (under != under.shift(fill_value=False)).cumsum()
    lengths = under.groupby(groups).sum()
    under_groups = under.groupby(groups).first()
    lengths = lengths[under_groups]
    return int(lengths.max()) if len(lengths) else 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    holdings = pd.read_parquet(HOLDINGS_PATH)
    holdings["date_rebalance"] = pd.to_datetime(holdings["date_rebalance"], errors="coerce")
    holdings["weight"] = pd.to_numeric(holdings["weight"], errors="coerce")
    holdings["month"] = holdings["month"].astype(str)
    holdings = holdings[
        (holdings["month"] >= EVAL_START_MONTH) & (holdings["month"] <= EVAL_END_MONTH)
    ].copy()

    cours, indices, _technical = load_market_panels()
    prices = _price_panel(cours)
    masi = _masi_returns(indices)

    months = sorted(holdings["month"].unique())
    assert months[0] == EVAL_START_MONTH and months[-1] == EVAL_END_MONTH

    # map month -> rebalance date (frozen in artefacts)
    reb_by_month = (
        holdings.groupby("month")["date_rebalance"].first().sort_index().to_dict()
    )

    monthly_rows: list[dict] = []
    holding_ret_rows: list[dict] = []
    daily_net_parts: list[pd.Series] = []
    daily_gross_parts: list[pd.Series] = []
    prev_w: dict[str, float] = {}

    lookahead_checks: list[dict] = []

    for i, mois in enumerate(months):
        g = holdings[holdings["month"] == mois].copy()
        reb = pd.Timestamp(reb_by_month[mois])
        if i + 1 < len(months):
            nxt = pd.Timestamp(reb_by_month[months[i + 1]])
            holding_end_month = months[i + 1]
        else:
            nxt = pd.Timestamp(prices.index.max())
            holding_end_month = nxt.strftime("%Y-%m")

        # Anti look-ahead : fenêtre de rendement strictement après reb
        assert prices.index.is_monotonic_increasing
        px_window = prices.loc[(prices.index > reb) & (prices.index <= nxt)]
        if not px_window.empty:
            assert px_window.index.min() > reb
            assert px_window.index.max() <= nxt

        wdict = {
            str(r.ticker): float(r.weight)
            for r in g.itertuples()
            if pd.notna(r.weight) and float(r.weight) > 0
        }
        # poids figés : ne pas recalculer
        w_sum = sum(wdict.values())
        assert abs(w_sum - 1.0) < 1e-8

        if prev_w:
            turnover = float(
                sum(
                    abs(wdict.get(t, 0.0) - prev_w.get(t, 0.0))
                    for t in set(wdict) | set(prev_w)
                )
            )
        else:
            turnover = float(sum(wdict.values()))  # entrée initiale = 1.0

        cost = float(turnover_cost(prev_w, wdict, TRANSACTION_COST))
        # cohérence cost = turnover * TRANSACTION_COST
        assert abs(cost - turnover * TRANSACTION_COST) < 1e-12

        daily_gross = portfolio_daily_returns(wdict, prices, reb, nxt)
        daily_net = daily_gross.copy()
        if not daily_net.empty and cost > 0:
            daily_net.iloc[0] = daily_net.iloc[0] - cost

        gross_return = float((1 + daily_gross).prod() - 1) if not daily_gross.empty else np.nan
        net_return = float((1 + daily_net).prod() - 1) if not daily_net.empty else np.nan

        portfolio_alpha = float(g["portfolio_alpha"].iloc[0])
        portfolio_cvar = float(g["portfolio_cvar"].iloc[0])
        portfolio_liquidity = float(g["portfolio_liquidity"].iloc[0])
        n_positions = int(len(g))

        # contributions titre (gross holding-period return)
        for r in g.itertuples():
            t = str(r.ticker)
            wi = float(r.weight)
            if t not in px_window.columns or wi <= 0:
                ri = np.nan
                contrib = np.nan
            else:
                series = px_window[t].dropna()
                if len(series) < 2:
                    ri = 0.0
                else:
                    ri = float(series.iloc[-1] / series.iloc[0] - 1.0)
                contrib = wi * ri
            holding_ret_rows.append(
                {
                    "month": mois,
                    "decision_date": reb.date().isoformat(),
                    "holding_start_exclusive": reb.date().isoformat(),
                    "holding_end_inclusive": (
                        px_window.index.max().date().isoformat()
                        if not px_window.empty
                        else None
                    ),
                    "holding_period_label": f"({mois} rebalance] → {holding_end_month}",
                    "cell": "C4",
                    "ticker": t,
                    "recommendation": str(r.recommendation).upper(),
                    "weight": wi,
                    "alpha": float(r.alpha) if pd.notna(r.alpha) else np.nan,
                    "liquidity_L": float(r.liquidity_L) if pd.notna(r.liquidity_L) else np.nan,
                    "vmq": float(r.vmq) if pd.notna(r.vmq) else np.nan,
                    "realized_return_holding": ri,
                    "contribution_gross": contrib,
                    "portfolio_alpha": portfolio_alpha,
                    "portfolio_cvar": portfolio_cvar,
                    "portfolio_liquidity": portfolio_liquidity,
                    "turnover": turnover,
                    "transaction_cost": cost,
                    "gross_return": gross_return,
                    "net_return": net_return,
                }
            )

        monthly_rows.append(
            {
                "month": mois,
                "decision_date": reb.date().isoformat(),
                "holding_end_date": (
                    px_window.index.max().date().isoformat()
                    if not px_window.empty
                    else None
                ),
                "holding_period": f"(after {reb.date()}] → {nxt.date() if i+1 < len(months) else 'last price'}",
                "gross_return": gross_return,
                "transaction_cost": cost,
                "net_return": net_return,
                "turnover": turnover,
                "portfolio_alpha": portfolio_alpha,
                "portfolio_cvar": portfolio_cvar,
                "portfolio_liquidity": portfolio_liquidity,
                "n_positions": n_positions,
                "n_days": int(len(daily_net)),
                "n_SELL": int((g["recommendation"].astype(str).str.upper() == "SELL").sum()),
                "sum_weights": w_sum,
            }
        )

        lookahead_checks.append(
            {
                "month": mois,
                "decision_date": str(reb.date()),
                "first_return_date": (
                    str(daily_gross.index.min().date()) if not daily_gross.empty else None
                ),
                "last_return_date": (
                    str(daily_gross.index.max().date()) if not daily_gross.empty else None
                ),
                "first_return_strictly_after_decision": (
                    bool(daily_gross.index.min() > reb) if not daily_gross.empty else None
                ),
                "weights_frozen_from_artefact": True,
                "weights_not_recomputed_with_tplus1": True,
            }
        )

        if not daily_net.empty:
            daily_net_parts.append(daily_net)
            daily_gross_parts.append(daily_gross)
        prev_w = wdict

    monthly = pd.DataFrame(monthly_rows)
    holdings_ret = pd.DataFrame(holding_ret_rows)
    la = pd.DataFrame(lookahead_checks)

    daily_net = pd.concat(daily_net_parts).sort_index()
    daily_gross = pd.concat(daily_gross_parts).sort_index()
    if daily_net.index.duplicated().any():
        daily_net = daily_net[~daily_net.index.duplicated(keep="last")]
    if daily_gross.index.duplicated().any():
        daily_gross = daily_gross[~daily_gross.index.duplicated(keep="last")]

    bench = masi.reindex(daily_net.index).fillna(0.0)
    perf_net = compute_backtest_metrics(daily_net, bench, name="C4_final_net")
    perf_gross = compute_backtest_metrics(daily_gross, bench, name="C4_final_gross")

    wealth_net = (1 + daily_net).cumprod()
    wealth100 = 100 * wealth_net
    dd = wealth_net / wealth_net.cummax() - 1
    dd_days = max_dd_duration_days(wealth_net)

    # métriques demandées (net = convention investisseur)
    r_m = monthly["net_return"].astype(float)
    metrics = {
        "cell": "C4",
        "configuration": "Hybrid + régime",
        "selection": "knee / utopia distance (frozen)",
        "window_decision_months": f"{EVAL_START_MONTH} → {EVAL_END_MONTH}",
        "transaction_cost_rate": TRANSACTION_COST,
        "turnover_definition": "L1 bilateral Σ|Δw| (project convention stage2_metrics)",
        "cost_formula": "turnover × 0.003, charged on first holding day",
        "n_months": int(len(monthly)),
        "total_return_net": float(wealth_net.iloc[-1] - 1),
        "total_return_gross": float((1 + daily_gross).prod() - 1),
        "cumulative_wealth_base100_net": float(wealth100.iloc[-1]),
        "annualized_return_net": perf_net.get("annualized_return"),
        "annualized_return_gross": perf_gross.get("annualized_return"),
        "volatility_ann_net": perf_net.get("volatility_ann"),
        "sharpe_net": perf_net.get("sharpe"),
        "sortino_net": perf_net.get("sortino"),
        "cvar_95_net": perf_net.get("cvar_95"),
        "max_drawdown_net": perf_net.get("max_drawdown"),
        "max_drawdown_duration_days": dd_days,
        "turnover_mean": float(monthly["turnover"].mean()),
        "transaction_cost_mean": float(monthly["transaction_cost"].mean()),
        "portfolio_liquidity_mean_exante": float(monthly["portfolio_liquidity"].mean()),
        "n_positions_mean": float(monthly["n_positions"].mean()),
        "pct_positive_months_net": float((r_m > 0).mean()),
        "alpha_ann_vs_masi_net": perf_net.get("alpha_annualized"),
        "days": perf_net.get("days"),
    }

    # contrôles
    controls = {
        "n_months": int(len(monthly)),
        "all_sum_weights_1": bool((abs(monthly["sum_weights"] - 1.0) < 1e-8).all()),
        "n_SELL_total": int(monthly["n_SELL"].sum()),
        "all_first_return_after_decision": bool(
            la["first_return_strictly_after_decision"].dropna().all()
        ),
        "weights_frozen_from_C4_final_artefact": True,
        "no_model_retrain": True,
        "no_weight_reoptimization": True,
        "transaction_cost_rate": TRANSACTION_COST,
        "rebalance_convention": (
            "Weights decided at month-end date_rebalance t (as-of ≤ t). "
            "Applied to realized returns on trading days strictly after t "
            "up to and including next month-end rebalance (exclusive of t)."
        ),
    }
    assert controls["all_sum_weights_1"]
    assert controls["n_SELL_total"] == 0
    assert controls["all_first_return_after_decision"]

    # exports
    monthly.to_csv(OUT / "C4_backtest_monthly_returns.csv", index=False)
    monthly.to_parquet(OUT / "C4_backtest_monthly_returns.parquet", index=False)
    holdings_ret.to_csv(OUT / "C4_backtest_holdings_realized.csv", index=False)
    holdings_ret.to_parquet(OUT / "C4_backtest_holdings_realized.parquet", index=False)
    la.to_csv(OUT / "C4_backtest_lookahead_checks.csv", index=False)

    wealth_df = pd.DataFrame(
        {
            "date": wealth100.index,
            "ret_net": daily_net.values,
            "ret_gross": daily_gross.reindex(daily_net.index).values,
            "wealth100_net": wealth100.values,
            "drawdown_net": dd.values,
        }
    )
    wealth_df.to_csv(OUT / "C4_backtest_wealth_drawdown_daily.csv", index=False)
    wealth_df.to_parquet(OUT / "C4_backtest_wealth_drawdown_daily.parquet", index=False)

    # monthly wealth for convenience
    mw = monthly.copy()
    mw["wealth100_net"] = 100 * (1 + mw["net_return"]).cumprod()
    peak = mw["wealth100_net"].cummax()
    mw["drawdown_net"] = mw["wealth100_net"] / peak - 1
    mw.to_csv(OUT / "C4_backtest_wealth_monthly.csv", index=False)

    pd.Series(metrics).to_csv(OUT / "C4_backtest_metrics.csv", header=["value"])
    (OUT / "C4_backtest_metrics.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "controls": controls,
                "perf_net_detail": perf_net,
                "perf_gross_detail": perf_gross,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    (OUT / "C4_backtest_controls.json").write_text(
        json.dumps(controls, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # figures
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wealth100.index, wealth100.values, color="#2ca02c", lw=1.6, label="C4 net (base 100)")
    ax.axhline(100, color="grey", lw=0.8, ls="--")
    ax.set_title("C4 final — richesse cumulée nette (base 100)")
    ax.set_ylabel("Richesse")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_wealth_C4_final.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dd.index, dd.values, color="#d62728", lw=1.3)
    ax.set_title("C4 final — drawdown net")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_drawdown_C4_final.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly["month"], monthly["net_return"], color="#1f77b4", width=0.8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("C4 final — rendements mensuels nets")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig_monthly_net_C4_final.png", dpi=140)
    plt.close()

    print("=== CONTROLES ANTI LOOK-AHEAD ===")
    for k, v in controls.items():
        print(f"  {k}: {v}")
    print()
    print("=== METRIQUES FINALES (NET) ===")
    for k in (
        "n_months",
        "total_return_net",
        "cumulative_wealth_base100_net",
        "annualized_return_net",
        "volatility_ann_net",
        "sharpe_net",
        "sortino_net",
        "cvar_95_net",
        "max_drawdown_net",
        "max_drawdown_duration_days",
        "turnover_mean",
        "transaction_cost_mean",
        "portfolio_liquidity_mean_exante",
        "n_positions_mean",
        "pct_positive_months_net",
    ):
        print(f"  {k}: {metrics[k]}")
    print()
    print("OUT:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
