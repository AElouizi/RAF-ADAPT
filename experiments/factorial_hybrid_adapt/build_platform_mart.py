"""
Construit le mart lecture seule pour la plateforme (principal = C2).

Source : livrable expérimental FINAL
  outputs/final_experiment_2010_2025/
Fenêtre testable : 2018-07 → 2025-06 (84 mois). Pas de filtre VMQ ≥ 500k.
Ne recalcule pas Stage 1 / NSGA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bvc_recommender.config import DATA_PROCESSED_DIR, FEATURES_DIR
from bvc_recommender.models.liquidity_filter import sigmoid_liquidity_factor
from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.assemble_final_experiment import (
    _period_end,
    ew_backtest,
    masi_on_calendar,
)
from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel

BASE = Path(__file__).resolve().parent
FINAL = BASE / "outputs" / "final_experiment_2010_2025"
OUT = BASE / "outputs" / "platform_mart"
WINDOW_START = "2015-06"
WINDOW_END = "2025-06"
PRINCIPAL_CELL = 2
PRINCIPAL_LABEL = "C2 Ridge + régime"


def _ticker_names() -> pd.DataFrame:
    h = pd.read_parquet(DATA_PROCESSED_DIR / "histo_const_ind.parquet")
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    names = (
        h.sort_values("date")
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "libelle"]]
        .rename(columns={"libelle": "name"})
    )
    names["ticker"] = names["ticker"].astype(str)
    names["name"] = names["name"].astype(str)
    return names


def _regime_by_month() -> pd.DataFrame:
    fi = pd.read_parquet(FEATURES_DIR / "features_indices.parquet")
    fi["date_cours"] = pd.to_datetime(fi["date_cours"], errors="coerce")
    fi = fi.dropna(subset=["date_cours"]).sort_values("date_cours")
    fi["month"] = fi["date_cours"].dt.to_period("M").astype(str)

    snap = fi.groupby("month", as_index=False).tail(1).copy()

    def label(r) -> str:
        if bool(r.get("is_bull")):
            return "bull"
        if bool(r.get("is_bear")):
            return "bear"
        if bool(r.get("is_neutral")) or bool(r.get("is_sideways")):
            return "neutral"
        return "unknown"

    snap["regime_market"] = snap.apply(label, axis=1)
    return snap[["month", "regime_market", "date_cours"]].rename(
        columns={"date_cours": "regime_asof"}
    )


def _daily_wealth_from_csv(path: Path, col: str) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["date"])
    return d[["date", "wealth100"]].rename(columns={"wealth100": col})


def main() -> int:
    hold_path = FINAL / "04_final_portfolios" / f"C{PRINCIPAL_CELL}_holdings_knee.parquet"
    hold_csv = FINAL / "04_final_portfolios" / f"C{PRINCIPAL_CELL}_holdings_knee.csv"
    if not hold_path.is_file() and not hold_csv.is_file():
        raise FileNotFoundError(f"Livrable FINAL manquant : {hold_path}")

    OUT.mkdir(parents=True, exist_ok=True)
    names = _ticker_names()
    regime = _regime_by_month()

    rec = pd.read_parquet(FINAL / "02_stage1_recommendations" / "stage2_inputs_14blocks.parquet")
    rec["date"] = pd.to_datetime(rec["date"], errors="coerce")
    rec["month"] = rec["date"].dt.to_period("M").astype(str)
    rec = rec[
        (rec["cell"] == PRINCIPAL_CELL)
        & (rec["month"] >= WINDOW_START)
        & (rec["month"] <= WINDOW_END)
    ].copy()
    rec["ticker"] = rec["ticker"].astype(str)
    rec["recommendation"] = rec["recommendation"].astype(str).str.upper()
    rec["alpha"] = pd.to_numeric(rec["predicted_return"], errors="coerce")
    rec["conviction_score"] = pd.to_numeric(rec["conviction_score"], errors="coerce")
    rec["vmq"] = pd.to_numeric(rec["liquidity_vmq_20j"], errors="coerce")
    rec["liquidity_L"] = sigmoid_liquidity_factor(rec["vmq"])
    rec = rec.merge(names, on="ticker", how="left")
    rec = rec.merge(regime, on="month", how="left")
    rec["name"] = rec["name"].fillna(rec["ticker"])
    keep_rec = [
        c
        for c in (
            "month",
            "date",
            "ticker",
            "name",
            "recommendation",
            "alpha",
            "conviction_score",
            "regime_market",
            "liquidity_L",
            "vmq",
            "realized_alpha",
            "cell",
            "cell_name",
            "model",
            "tau",
        )
        if c in rec.columns
    ]
    fact_rec = rec[keep_rec].sort_values(
        ["month", "recommendation", "alpha"], ascending=[True, True, False]
    )

    hold = pd.read_parquet(hold_path) if hold_path.is_file() else pd.read_csv(hold_csv)
    hold["month"] = hold["month"].astype(str)
    hold["ticker"] = hold["ticker"].astype(str)
    hold["weight"] = pd.to_numeric(hold["weight"], errors="coerce")
    hold["alpha"] = pd.to_numeric(hold["predicted_return"], errors="coerce")
    hold["liquidity_L"] = pd.to_numeric(hold["L_sigmoid"], errors="coerce")
    hold["vmq"] = pd.to_numeric(hold["liquidity_vmq_20j"], errors="coerce")
    hold["recommendation"] = hold["recommendation"].astype(str).str.upper()
    hold = hold.merge(names, on="ticker", how="left")
    hold["name"] = hold["name"].fillna(hold["ticker"])
    hold = hold.merge(regime[["month", "regime_market"]], on="month", how="left")
    hold["in_portfolio"] = 1
    hold["weight_pct"] = hold["weight"] * 100

    monthly_bt = pd.read_csv(FINAL / "05_backtest" / f"C{PRINCIPAL_CELL}_monthly.csv")
    monthly_bt["month"] = monthly_bt["month"].astype(str)
    obj = (
        hold.groupby("month", as_index=False)
        .agg(
            portfolio_alpha=("expected_alpha_opt", "first"),
            portfolio_cvar=("cvar_95", "first"),
            portfolio_liquidity=("liquidity_L_weighted", "first"),
            date_rebalance=("date_rebalance", "first"),
        )
    )
    monthly = monthly_bt.merge(obj, on="month", how="left")
    monthly = monthly.merge(regime[["month", "regime_market"]], on="month", how="left")
    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["wealth100_net"] = 100 * (1 + monthly["net_return"].fillna(0)).cumprod()
    monthly["cell"] = f"C{PRINCIPAL_CELL}"
    monthly["selection_method"] = "knee_utopia_distance"
    monthly["n_positions"] = monthly["n_positions"].astype(int)

    # Realized holding returns (as-of holding window, no look-ahead in weights)
    cours, indices, _ = load_market_panels()
    prices = _price_panel(cours)
    months = sorted(hold["month"].unique())
    reb = hold.groupby("month")["date_rebalance"].first()
    real_rows = []
    for i, mois in enumerate(months):
        t0 = pd.Timestamp(reb[mois])
        t1 = _period_end(months, i, reb, prices, WINDOW_END)
        g = hold[hold["month"] == mois]
        px = prices.loc[(prices.index > t0) & (prices.index <= t1)]
        for r in g.itertuples():
            tkr = str(r.ticker)
            ret = np.nan
            if tkr in px.columns and len(px) and px[tkr].notna().sum() >= 2:
                s = px[tkr].dropna()
                if len(s) >= 2:
                    ret = float(s.iloc[-1] / s.iloc[0] - 1)
            w = float(r.weight) if pd.notna(r.weight) else 0.0
            real_rows.append(
                {
                    "month": mois,
                    "ticker": tkr,
                    "realized_return_holding": ret,
                    "contribution_gross": (w * ret) if pd.notna(ret) else np.nan,
                }
            )
    real = pd.DataFrame(real_rows)
    hold = hold.merge(real, on=["month", "ticker"], how="left")
    fact_hold = hold.sort_values(["month", "weight"], ascending=[True, False])

    port_keys = fact_hold[["month", "ticker", "weight", "weight_pct", "in_portfolio"]].copy()
    unified = fact_rec.merge(port_keys, on=["month", "ticker"], how="left")
    unified["in_portfolio"] = unified["in_portfolio"].fillna(0).astype(int)
    unified["weight"] = pd.to_numeric(unified["weight"], errors="coerce").fillna(0.0)
    unified["weight_pct"] = pd.to_numeric(unified["weight_pct"], errors="coerce").fillna(0.0)

    w4 = _daily_wealth_from_csv(FINAL / "05_backtest" / "C4_daily.csv", "C4")
    w1 = _daily_wealth_from_csv(FINAL / "05_backtest" / "C1_daily.csv", "C1")
    w2 = _daily_wealth_from_csv(FINAL / "05_backtest" / "C2_daily.csv", "C2")
    w3 = _daily_wealth_from_csv(FINAL / "05_backtest" / "C3_daily.csv", "C3")
    wealth = w4.merge(w1, on="date", how="outer").merge(w2, on="date", how="outer").merge(w3, on="date", how="outer")
    wealth = wealth.sort_values("date")
    principal_col = f"C{PRINCIPAL_CELL}"
    if principal_col in wealth.columns:
        wealth["Principal"] = wealth[principal_col]

    inp = pd.read_parquet(FINAL / "02_stage1_recommendations" / "stage2_inputs_14blocks.parquet")
    inp["date"] = pd.to_datetime(inp["date"])
    inp["mois"] = inp["date"].dt.to_period("M").astype(str)
    inp["recommendation"] = inp["recommendation"].astype(str).str.upper()
    inp["vmq"] = pd.to_numeric(inp["liquidity_vmq_20j"], errors="coerce")
    inp["L"] = sigmoid_liquidity_factor(inp["vmq"])
    sub = inp[inp["cell"] == PRINCIPAL_CELL].copy()
    months_h = sorted(hold["month"].unique())
    reb_d = {m: pd.Timestamp(reb[m]) for m in months_h}
    masi_all = _masi_returns(indices)
    ew_m, ew_d = ew_backtest(sub, prices, masi_all, months_h, reb_d, WINDOW_END)
    masi_m, masi_d = masi_on_calendar(masi_all, months_h, reb_d, prices, WINDOW_END)

    def _series_wealth(s: pd.Series, name: str) -> pd.DataFrame:
        s = s.sort_index()
        if s.index.duplicated().any():
            s = s[~s.index.duplicated(keep="last")]
        return pd.DataFrame({"date": s.index, name: 100 * (1 + s).cumprod()})

    wealth = wealth.merge(_series_wealth(masi_d, "MASI"), on="date", how="left")
    wealth = wealth.merge(_series_wealth(ew_d, "EqualWeight"), on="date", how="left")
    wealth = wealth.sort_values("date")
    for col in ("C4", "MASI", "EqualWeight", "C1", "C2", "C3", "Principal"):
        if col in wealth.columns:
            wealth[col] = wealth[col].ffill()
            wealth[f"dd_{col}"] = wealth[col] / wealth[col].cummax() - 1
    cut = pd.Timestamp("2025-07-31")
    wealth = wealth[pd.to_datetime(wealth["date"]) <= cut].copy()

    table = pd.read_csv(FINAL / "06_benchmarks" / "table_C1C4_MASI_EW.csv")
    principal_name = PRINCIPAL_LABEL
    kpi = table[
        table["Strategie"].isin([principal_name, "MASI", f"EqualWeight C{PRINCIPAL_CELL}"])
    ].copy()
    kpi["Strategie"] = kpi["Strategie"].replace(
        {f"EqualWeight C{PRINCIPAL_CELL}": "Equal Weight", principal_name: PRINCIPAL_LABEL}
    )
    tests = pd.read_csv(FINAL / "07_statistical_validation" / "tests_principaux.csv")
    subp = pd.read_csv(FINAL / "08_robustness" / "subperiods_C1C4.csv")
    liq = pd.read_csv(FINAL / "08_robustness" / "robustness_liquidity_objective.csv")
    sens = pd.read_csv(FINAL / "08_robustness" / "sensitivity_selection_C4.csv")

    monthly_bench = pd.DataFrame(
        {
            "month": monthly["month"],
            f"C{PRINCIPAL_CELL}": monthly["net_return"],
            "Principal": monthly["net_return"],
        }
    )
    ew_mm = ew_m.rename(columns={"net_return": "EqualWeight"})[["month", "EqualWeight"]]
    masi_mm = masi_m.rename(columns={"net_return": "MASI"})[["month", "MASI"]]
    monthly_bench = monthly_bench.merge(ew_mm, on="month", how="left").merge(
        masi_mm, on="month", how="left"
    )

    names.to_parquet(OUT / "dim_ticker.parquet", index=False)
    names.to_csv(OUT / "dim_ticker.csv", index=False)
    fact_rec.to_parquet(OUT / "fact_recommendations_C4.parquet", index=False)
    fact_rec.to_csv(OUT / "fact_recommendations_C4.csv", index=False)
    fact_hold.to_parquet(OUT / "fact_portfolio_holdings_C4.parquet", index=False)
    fact_hold.to_csv(OUT / "fact_portfolio_holdings_C4.csv", index=False)
    unified.to_parquet(OUT / "fact_titles_month_C4.parquet", index=False)
    unified.to_csv(OUT / "fact_titles_month_C4.csv", index=False)
    monthly.to_parquet(OUT / "fact_portfolio_monthly_C4.parquet", index=False)
    monthly.to_csv(OUT / "fact_portfolio_monthly_C4.csv", index=False)
    wealth.to_parquet(OUT / "fact_performance_daily.parquet", index=False)
    wealth.to_csv(OUT / "fact_performance_daily.csv", index=False)
    kpi.to_csv(OUT / "kpi_benchmarks.csv", index=False)
    table.to_csv(OUT / "kpi_C1C4.csv", index=False)
    tests.to_csv(OUT / "tests_statistiques.csv", index=False)
    subp.to_csv(OUT / "subperiods_C1C4.csv", index=False)
    liq.to_csv(OUT / "robustness_liquidity.csv", index=False)
    sens.to_csv(OUT / "sensitivity_selection.csv", index=False)
    monthly_bench.to_csv(OUT / "fact_benchmarks_monthly.csv", index=False)

    principal = table[table["Strategie"].str.startswith(f"C{PRINCIPAL_CELL}")].iloc[0]
    months_out = sorted(fact_rec["month"].unique().tolist())
    meta = {
        "model": PRINCIPAL_LABEL,
        "principal_cell": PRINCIPAL_CELL,
        "stage2": "NSGA-III (Alpha, CVaR, Liquidité) + knee / utopia distance",
        "w_max": 0.10,
        "window_requested": "2010-01 → 2025-06",
        "window": f"{WINDOW_START} → {WINDOW_END}",
        "window_note": "2010-01 → 2018-06 impossible sans réentraîner Stage 1",
        "vmq_eligibility_filter": False,
        "liquidity_role": "Pareto objective only",
        "months": months_out,
        "n_months": len(months_out),
        "n_recommendation_rows": int(len(fact_rec)),
        "n_holding_rows": int(len(fact_hold)),
        "read_only": True,
        "source": "outputs/final_experiment_2010_2025/",
        "metrics_net": {
            "annualized_return_net": float(principal["Rendement"]),
            "sharpe_net": float(principal["Sharpe"]),
            "sortino_net": float(principal["Sortino"]),
            "max_drawdown_net": float(principal["Max_DD"]),
            "cvar_95_net": float(principal["CVaR"]),
            "liquidity_mean": float(principal["Liquidite"]),
            "turnover_mean": float(principal["Turnover"]),
            "wealth_final_100": float(principal["wealth_final_100"]),
            "n_months": int(principal["n_months"]),
        },
    }
    (OUT / "meta_platform.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("Mart écrit dans", OUT)
    print("mois:", len(months_out), months_out[0], "→", months_out[-1])
    print("recos:", len(fact_rec), "| holdings:", len(fact_hold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
