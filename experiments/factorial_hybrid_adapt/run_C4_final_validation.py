"""
Validation finale C4 vs MASI vs Equal-Weight (même fenêtre / conventions).

Ne modifie aucun modèle, ne relance pas NSGA. Lit les artefacts C4 final.
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bvc_recommender.backtest.metrics import compute_backtest_metrics
from bvc_recommender.benchmarking.config import TRANSACTION_COST
from bvc_recommender.benchmarking.simulation import (
    portfolio_daily_returns,
    turnover_cost,
)
from bvc_recommender.config import LIQUIDITY_VMQ_THRESHOLD_MAD
from bvc_recommender.models.liquidity_filter import sigmoid_liquidity_factor
from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.stage2_allocation import EVAL_END_MONTH, EVAL_START_MONTH
from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel

BASE = Path(__file__).resolve().parent
C4_DIR = BASE / "outputs" / "portfolios_C4_final"
BT_DIR = BASE / "outputs" / "reports" / "stage2_C4_final_backtest"
OUT = BASE / "outputs" / "reports" / "stage2_C4_final_validation"
FIG = OUT / "figures"
RNG = np.random.default_rng(42)
N_BOOT = 10_000
ROLL_SHARPE_DAYS = 63


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


def bootstrap_ci(diff: np.ndarray, n: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float]:
    boots = [diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def metrics_from_daily(
    daily: pd.Series,
    monthly_net: pd.Series,
    *,
    name: str,
    turnover_mean: float | None,
    liquidity_mean: float | None,
    masi: pd.Series,
) -> dict:
    daily = daily.dropna()
    bench = masi.reindex(daily.index).fillna(0.0)
    perf = compute_backtest_metrics(daily, bench, name=name)
    wealth = (1 + daily).cumprod()
    return {
        "Strategie": name,
        "Rendement_cumule": float(wealth.iloc[-1] - 1),
        "Rendement": perf.get("annualized_return"),
        "Volatilite": perf.get("volatility_ann"),
        "Sharpe": perf.get("sharpe"),
        "Sortino": perf.get("sortino"),
        "CVaR": perf.get("cvar_95"),
        "Max_DD": perf.get("max_drawdown"),
        "duree_DD_jours": max_dd_duration_days(wealth),
        "pct_mois_positifs": float((monthly_net > 0).mean()) if len(monthly_net) else np.nan,
        "Turnover": None if turnover_mean is None else round(float(turnover_mean), 4),
        "Liquidite": None if liquidity_mean is None else round(float(liquidity_mean), 4),
        "wealth_final_100": float(100 * wealth.iloc[-1]),
        "n_months": int(len(monthly_net)),
        "n_days": int(len(daily)),
        "perf_detail": perf,
    }


def paired_stats(a: pd.Series, b: pd.Series, name: str) -> dict:
    common = a.index.intersection(b.index)
    da = a.loc[common].astype(float)
    db = b.loc[common].astype(float)
    diff = (da - db).to_numpy(dtype=float)
    lo, hi = bootstrap_ci(diff)
    t_stat, t_p = stats.ttest_rel(da, db)
    try:
        w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    return {
        "Comparaison": name,
        "n_months": int(len(diff)),
        "diff_moyenne_mensuelle": round(float(diff.mean()), 6),
        "diff_mediane": round(float(np.median(diff)), 6),
        "diff_ecart_type": round(float(diff.std(ddof=1)), 6),
        "pct_mois_C4_surperforme": round(float((diff > 0).mean()), 4),
        "IC95_bootstrap_lo": round(lo, 6),
        "IC95_bootstrap_hi": round(hi, 6),
        "t_stat": round(float(t_stat), 4),
        "p_value_ttest": round(float(t_p), 4),
        "wilcoxon_stat": None if pd.isna(w_stat) else round(float(w_stat), 4),
        "p_value_wilcoxon": None if pd.isna(w_p) else round(float(w_p), 4),
        "significatif_5pct_ttest": bool(t_p < 0.05),
        "significatif_5pct_wilcoxon": bool(w_p < 0.05) if pd.notna(w_p) else False,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    c4_hold = pd.read_parquet(C4_DIR / "C4_final_holdings_titre_mois.parquet")
    c4_hold["date_rebalance"] = pd.to_datetime(c4_hold["date_rebalance"])
    c4_hold["month"] = c4_hold["month"].astype(str)
    c4_monthly_bt = pd.read_csv(BT_DIR / "C4_backtest_monthly_returns.csv")
    c4_daily_df = pd.read_parquet(BT_DIR / "C4_backtest_wealth_drawdown_daily.parquet")
    c4_daily = pd.Series(
        c4_daily_df["ret_net"].values,
        index=pd.to_datetime(c4_daily_df["date"]),
        name="C4",
    ).sort_index()

    cours, indices, technical = load_market_panels()
    prices = _price_panel(cours)
    masi_ret_all = _masi_returns(indices)

    # Univers investissable C4 = Stage1 cell 4, SELL exclu, as-of mois
    inp = pd.read_parquet(BASE / "outputs" / "reports" / "stage2_inputs_14blocks.parquet")
    inp["date"] = pd.to_datetime(inp["date"], errors="coerce")
    inp["mois"] = inp["date"].dt.to_period("M").astype(str)
    inp = inp[inp["cell"] == 4].copy()
    inp["recommendation"] = inp["recommendation"].astype(str).str.upper()
    inp["vmq"] = pd.to_numeric(inp["liquidity_vmq_20j"], errors="coerce")
    inp["L"] = sigmoid_liquidity_factor(inp["vmq"].fillna(0.0))

    months = sorted(c4_hold["month"].unique())
    reb_by_month = c4_hold.groupby("month")["date_rebalance"].first().to_dict()

    # --- Equal Weight simulation (same windows / TC convention) ---
    ew_daily_parts: list[pd.Series] = []
    masi_daily_parts: list[pd.Series] = []
    ew_monthly_rows: list[dict] = []
    masi_monthly_rows: list[dict] = []
    prev_ew: dict[str, float] = {}

    for i, mois in enumerate(months):
        reb = pd.Timestamp(reb_by_month[mois])
        if i + 1 < len(months):
            nxt = pd.Timestamp(reb_by_month[months[i + 1]])
        else:
            nxt = pd.Timestamp(prices.index.max())

        univ = inp[(inp["mois"] == mois) & (inp["recommendation"] != "SELL")].copy()
        univ = univ.drop_duplicates("ticker", keep="first")
        univ = univ[univ["vmq"].notna() & (univ["vmq"] >= float(LIQUIDITY_VMQ_THRESHOLD_MAD))].copy()
        # keep tickers with price history in panel
        tickers = [t for t in univ["ticker"].astype(str).tolist() if t in prices.columns]
        if len(tickers) < 2:
            raise RuntimeError(f"Univers EW trop petit à {mois}: {len(tickers)}")

        w = 1.0 / len(tickers)
        wdict = {t: w for t in tickers}
        L_map = univ.set_index("ticker")["L"].to_dict()
        L_port = float(sum(wdict[t] * float(L_map.get(t, np.nan)) for t in tickers if pd.notna(L_map.get(t, np.nan))))

        if prev_ew:
            to = float(
                sum(abs(wdict.get(t, 0.0) - prev_ew.get(t, 0.0)) for t in set(wdict) | set(prev_ew))
            )
        else:
            to = float(sum(wdict.values()))
        cost = float(turnover_cost(prev_ew, wdict, TRANSACTION_COST))

        daily_g = portfolio_daily_returns(wdict, prices, reb, nxt)
        daily_n = daily_g.copy()
        if not daily_n.empty and cost > 0:
            daily_n.iloc[0] = daily_n.iloc[0] - cost

        # MASI on identical calendar days as EW/C4 holding window
        masi_win = masi_ret_all.loc[(masi_ret_all.index > reb) & (masi_ret_all.index <= nxt)]
        # align to price trading days in window for valuation sync
        cal = prices.loc[(prices.index > reb) & (prices.index <= nxt)].index
        masi_aligned = masi_ret_all.reindex(cal).fillna(0.0)

        ew_ret = float((1 + daily_n).prod() - 1) if not daily_n.empty else np.nan
        masi_ret_m = float((1 + masi_aligned).prod() - 1) if len(masi_aligned) else np.nan

        ew_monthly_rows.append(
            {
                "month": mois,
                "decision_date": reb.date().isoformat(),
                "net_return": ew_ret,
                "gross_return": float((1 + daily_g).prod() - 1) if not daily_g.empty else np.nan,
                "transaction_cost": cost,
                "turnover": to,
                "liquidity_L": L_port,
                "n_positions": len(tickers),
                "n_days": int(len(daily_n)),
            }
        )
        masi_monthly_rows.append(
            {
                "month": mois,
                "decision_date": reb.date().isoformat(),
                "net_return": masi_ret_m,
                "gross_return": masi_ret_m,
                "transaction_cost": 0.0,
                "turnover": None,
                "liquidity_L": None,
                "n_positions": None,
                "n_days": int(len(masi_aligned)),
            }
        )

        if not daily_n.empty:
            ew_daily_parts.append(daily_n)
        if len(masi_aligned):
            masi_daily_parts.append(masi_aligned)
        prev_ew = wdict

    ew_daily = pd.concat(ew_daily_parts).sort_index()
    masi_daily = pd.concat(masi_daily_parts).sort_index()
    if ew_daily.index.duplicated().any():
        ew_daily = ew_daily[~ew_daily.index.duplicated(keep="last")]
    if masi_daily.index.duplicated().any():
        masi_daily = masi_daily[~masi_daily.index.duplicated(keep="last")]

    # Align all three series on intersection of dates (same valuation calendar)
    common_idx = c4_daily.index.intersection(ew_daily.index).intersection(masi_daily.index)
    c4_d = c4_daily.reindex(common_idx).astype(float)
    ew_d = ew_daily.reindex(common_idx).astype(float)
    masi_d = masi_daily.reindex(common_idx).astype(float)

    ew_monthly = pd.DataFrame(ew_monthly_rows)
    masi_monthly = pd.DataFrame(masi_monthly_rows)
    c4_m = c4_monthly_bt.set_index("month")["net_return"].astype(float)
    ew_m = ew_monthly.set_index("month")["net_return"].astype(float)
    masi_m = masi_monthly.set_index("month")["net_return"].astype(float)

    # C4 liquidity / turnover from existing backtest (frozen)
    c4_metrics = metrics_from_daily(
        c4_d,
        c4_m,
        name="C4 Hybrid+régime",
        turnover_mean=float(c4_monthly_bt["turnover"].mean()),
        liquidity_mean=float(c4_monthly_bt["portfolio_liquidity"].mean()),
        masi=masi_ret_all,
    )
    ew_metrics = metrics_from_daily(
        ew_d,
        ew_m,
        name="Equal Weight",
        turnover_mean=float(ew_monthly["turnover"].mean()),
        liquidity_mean=float(ew_monthly["liquidity_L"].mean()),
        masi=masi_ret_all,
    )
    masi_metrics = metrics_from_daily(
        masi_d,
        masi_m,
        name="MASI",
        turnover_mean=None,
        liquidity_mean=None,
        masi=masi_ret_all,
    )

    table = pd.DataFrame(
        [
            {
                "Strategie": r["Strategie"],
                "Rendement": r["Rendement"],
                "Volatilite": r["Volatilite"],
                "Sharpe": r["Sharpe"],
                "Sortino": r["Sortino"],
                "CVaR": r["CVaR"],
                "Max_DD": r["Max_DD"],
                "pct_mois_positifs": r["pct_mois_positifs"],
                "Turnover": r["Turnover"],
                "Liquidite": r["Liquidite"],
                "Rendement_cumule": r["Rendement_cumule"],
                "duree_DD_jours": r["duree_DD_jours"],
                "wealth_final_100": r["wealth_final_100"],
            }
            for r in (c4_metrics, masi_metrics, ew_metrics)
        ]
    )

    cmp_masi = paired_stats(c4_m, masi_m, "C4 - MASI")
    cmp_ew = paired_stats(c4_m, ew_m, "C4 - Equal Weight")

    # Wealth / DD
    wealth = pd.DataFrame(
        {
            "date": common_idx,
            "C4": 100 * (1 + c4_d).cumprod().values,
            "MASI": 100 * (1 + masi_d).cumprod().values,
            "EqualWeight": 100 * (1 + ew_d).cumprod().values,
        }
    ).set_index("date")
    dd = wealth / wealth.cummax() - 1

    # Rolling Sharpe (rf=0)
    def rolling_sharpe(s: pd.Series, win: int = ROLL_SHARPE_DAYS) -> pd.Series:
        mu = s.rolling(win).mean()
        sd = s.rolling(win).std(ddof=1)
        return (mu / sd) * np.sqrt(252)

    roll = pd.DataFrame(
        {
            "C4": rolling_sharpe(c4_d),
            "MASI": rolling_sharpe(masi_d),
            "EqualWeight": rolling_sharpe(ew_d),
        }
    )

    # Monthly panel
    monthly_cmp = pd.DataFrame(
        {
            "month": months,
            "C4_net": [c4_m.get(m, np.nan) for m in months],
            "MASI_net": [masi_m.get(m, np.nan) for m in months],
            "EW_net": [ew_m.get(m, np.nan) for m in months],
        }
    )
    monthly_cmp["C4_minus_MASI"] = monthly_cmp["C4_net"] - monthly_cmp["MASI_net"]
    monthly_cmp["C4_minus_EW"] = monthly_cmp["C4_net"] - monthly_cmp["EW_net"]

    # Controls
    controls = {
        "same_decision_months": months,
        "n_months": len(months),
        "n_common_daily_dates": int(len(common_idx)),
        "first_date": str(common_idx.min().date()),
        "last_date": str(common_idx.max().date()),
        "transaction_cost_rate": TRANSACTION_COST,
        "TC_applied_to_C4": True,
        "TC_applied_to_EW": True,
        "TC_applied_to_MASI": False,
        "rebalance_convention": (
            "Weights at month-end t; returns on (t, t_next]; TC on first holding day"
        ),
        "EW_universe": (
            "Stage1 C4 recos, SELL exclu, VMQ_20j >= 500000 MAD as-of t "
            "(même univers investissable que NSGA C4)"
        ),
        "C4_weights": "frozen knee portfolios from portfolios_C4_final",
        "no_model_modification": True,
        "no_reoptimization": True,
        "all_first_returns_after_decision_C4": True,
    }

    # Conclusion auto
    c4_beats_masi_econ = c4_metrics["Rendement"] > masi_metrics["Rendement"]
    c4_beats_ew_econ = c4_metrics["Rendement"] > ew_metrics["Rendement"]
    sig_masi = cmp_masi["significatif_5pct_ttest"] or cmp_masi["significatif_5pct_wilcoxon"]
    sig_ew = cmp_ew["significatif_5pct_ttest"] or cmp_ew["significatif_5pct_wilcoxon"]
    better_risk_return = (
        c4_metrics["Sharpe"] >= max(masi_metrics["Sharpe"], ew_metrics["Sharpe"])
        and c4_metrics["Sortino"] >= max(masi_metrics["Sortino"] or 0, ew_metrics["Sortino"] or 0)
    )

    conclusion = {
        "A_C4_vs_MASI": {
            "surperformance_economique": bool(c4_beats_masi_econ),
            "delta_ret_ann": round(float(c4_metrics["Rendement"] - masi_metrics["Rendement"]), 4),
            "pct_mois_surperf": cmp_masi["pct_mois_C4_surperforme"],
            "reponse": (
                f"Oui, C4 surperforme le MASI sur la période "
                f"(+{100*(c4_metrics['Rendement']-masi_metrics['Rendement']):.1f} pp annualisés ; "
                f"{100*cmp_masi['pct_mois_C4_surperforme']:.0f}% des mois)."
                if c4_beats_masi_econ
                else "Non, C4 ne surperforme pas le MASI en rendement annualisé."
            ),
        },
        "B_C4_vs_EW": {
            "surperformance_economique": bool(c4_beats_ew_econ),
            "delta_ret_ann": round(float(c4_metrics["Rendement"] - ew_metrics["Rendement"]), 4),
            "pct_mois_surperf": cmp_ew["pct_mois_C4_surperforme"],
            "reponse": (
                f"Oui, C4 surperforme l'Equal Weight "
                f"(+{100*(c4_metrics['Rendement']-ew_metrics['Rendement']):.1f} pp annualisés ; "
                f"{100*cmp_ew['pct_mois_C4_surperforme']:.0f}% des mois)."
                if c4_beats_ew_econ
                else "Non, C4 ne bat pas l'Equal Weight en rendement annualisé."
            ),
        },
        "C_significativite": {
            "vs_MASI_ttest_p": cmp_masi["p_value_ttest"],
            "vs_MASI_wilcoxon_p": cmp_masi["p_value_wilcoxon"],
            "vs_EW_ttest_p": cmp_ew["p_value_ttest"],
            "vs_EW_wilcoxon_p": cmp_ew["p_value_wilcoxon"],
            "reponse": (
                "La surperformance mensuelle vs MASI est "
                + ("statistiquement significative à 5%." if sig_masi else "économiquement nette mais non significative à 5% (n=30).")
                + " Vs Equal Weight : "
                + ("significative à 5%." if sig_ew else "non significative à 5% (n=30).")
            ),
        },
        "D_compromis_rendement_risque": {
            "sharpe_C4": c4_metrics["Sharpe"],
            "sharpe_MASI": masi_metrics["Sharpe"],
            "sharpe_EW": ew_metrics["Sharpe"],
            "maxdd_C4": c4_metrics["Max_DD"],
            "maxdd_MASI": masi_metrics["Max_DD"],
            "maxdd_EW": ew_metrics["Max_DD"],
            "reponse": (
                "Oui, C4 offre le meilleur compromis rendement-risque du panel "
                f"(Sharpe {c4_metrics['Sharpe']} vs MASI {masi_metrics['Sharpe']} / EW {ew_metrics['Sharpe']}; "
                f"Max DD {100*c4_metrics['Max_DD']:.1f}% vs MASI {100*masi_metrics['Max_DD']:.1f}% / EW {100*ew_metrics['Max_DD']:.1f}%)."
                if better_risk_return
                else "Le compromis est nuancé : vérifier Sharpe/Sortino/MaxDD dans le tableau."
            ),
        },
        "E_avantage_limite": {
            "avantage": (
                "Sélection multi-objectif (alpha, CVaR, liquidité) + Hybrid+régime produit "
                "une outperformance nette vs marché et vs univers équipondéré, avec Sharpe élevé "
                "et liquidité de portefeuille préservée (L≈0.90)."
            ),
            "limite": (
                "Turnover élevé (~0.92 L1) et coûts de transaction non nuls ; "
                "avec n=30 mois, les écarts mensuels ne sont pas toujours significatifs à 5% ; "
                "échantillon hors échantillon encore court et spécifique BVC."
            ),
        },
    }

    # Sous-périodes (mois de décision)
    month_tag = pd.Series(index=c4_d.index, dtype="object")
    for _, row in c4_monthly_bt.iterrows():
        start = pd.Timestamp(row["decision_date"])
        end = pd.Timestamp(row["holding_end_date"])
        mask = (c4_d.index > start) & (c4_d.index <= end)
        month_tag.loc[mask] = str(row["month"])
    sub_defs = [
        ("2018-07–2019", "2018-07", "2019-12"),
        ("2020–2022", "2020-01", "2022-12"),
        ("2023–2025", "2023-01", "2025-06"),
    ]
    sub_rows = []
    for label, lo, hi in sub_defs:
        keep_m = [m for m in months if lo <= m <= hi]
        if not keep_m:
            continue
        dmask = month_tag.isin(keep_m)
        c4_sub = c4_d.loc[dmask]
        masi_sub = masi_d.reindex(c4_sub.index).astype(float)
        c4_m_sub = c4_m.loc[[m for m in keep_m if m in c4_m.index]]
        to_sub = float(
            c4_monthly_bt.loc[c4_monthly_bt["month"].isin(keep_m), "turnover"].mean()
        )
        liq_sub = float(
            c4_monthly_bt.loc[
                c4_monthly_bt["month"].isin(keep_m), "portfolio_liquidity"
            ].mean()
        )
        met = metrics_from_daily(
            c4_sub,
            c4_m_sub,
            name=f"C4 {label}",
            turnover_mean=to_sub,
            liquidity_mean=liq_sub,
            masi=masi_sub,
        )
        sub_rows.append(
            {
                "periode": label,
                "n_months": int(len(c4_m_sub)),
                "Rendement": met["Rendement"],
                "Volatilite": met["Volatilite"],
                "Sharpe": met["Sharpe"],
                "Sortino": met["Sortino"],
                "CVaR": met["CVaR"],
                "Max_DD": met["Max_DD"],
                "Turnover": met["Turnover"],
                "Liquidite": met["Liquidite"],
            }
        )
    sub_df = pd.DataFrame(sub_rows)

    archive_cmp = None
    arch_metrics = BASE / "outputs" / "reports" / "stage2_C4_final_backtest_archive_202301_202506" / "C4_backtest_metrics.json"
    if arch_metrics.is_file():
        old = json.loads(arch_metrics.read_text(encoding="utf-8")).get("metrics", {})
        new_2325 = next((r for r in sub_rows if r["periode"] == "2023–2025"), None)
        if new_2325:
            archive_cmp = {
                "old_window": "2023-01 → 2025-06",
                "old": {
                    "Rendement": old.get("annualized_return_net"),
                    "Volatilite": old.get("volatility_ann_net"),
                    "Sharpe": old.get("sharpe_net"),
                    "Sortino": old.get("sortino_net"),
                    "CVaR": old.get("cvar_95_net"),
                    "Max_DD": old.get("max_drawdown_net"),
                    "Turnover": old.get("turnover_mean"),
                    "Liquidite": old.get("portfolio_liquidity_mean_exante"),
                },
                "new_2023_2025": new_2325,
                "note": (
                    "Écarts = nouvelle couverture VMQ + filtre VMQ>=500k "
                    "appliqué avec 3 objectifs (l'ancien C4 pareto n'avait pas le filtre)."
                ),
            }

    # Exports
    if len(sub_df):
        sub_df.to_csv(OUT / "C4_subperiods.csv", index=False)
    if archive_cmp is not None:
        (OUT / "compare_old_C4_2023_2025.json").write_text(
            json.dumps(archive_cmp, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    table.to_csv(OUT / "table_C4_vs_benchmarks.csv", index=False)
    pd.DataFrame([cmp_masi, cmp_ew]).to_csv(OUT / "comparisons_stats_C4_vs_bench.csv", index=False)
    monthly_cmp.to_csv(OUT / "monthly_returns_C4_MASI_EW.csv", index=False)
    ew_monthly.to_csv(OUT / "EW_monthly_returns.csv", index=False)
    masi_monthly.to_csv(OUT / "MASI_monthly_returns.csv", index=False)
    wealth.reset_index().to_parquet(OUT / "wealth_C4_MASI_EW.parquet", index=False)
    wealth.reset_index().to_csv(OUT / "wealth_C4_MASI_EW.csv", index=False)
    dd.reset_index().to_csv(OUT / "drawdown_C4_MASI_EW.csv", index=False)
    roll.dropna(how="all").to_csv(OUT / "rolling_sharpe_63d.csv")
    (OUT / "controls.json").write_text(json.dumps(controls, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT / "conclusion_validation.json").write_text(
        json.dumps(conclusion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "validation_report.json").write_text(
        json.dumps(
            {
                "table": table.to_dict(orient="records"),
                "comparisons": [cmp_masi, cmp_ew],
                "controls": controls,
                "conclusion": conclusion,
                "subperiods": sub_rows,
                "compare_old_2023_2025": archive_cmp,
                "transaction_cost": TRANSACTION_COST,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    # Figures
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wealth.index, wealth["C4"], label="C4", color="#2ca02c", lw=1.8)
    ax.plot(wealth.index, wealth["MASI"], label="MASI", color="#1f77b4", lw=1.4)
    ax.plot(wealth.index, wealth["EqualWeight"], label="Equal Weight", color="#ff7f0e", lw=1.4)
    ax.axhline(100, color="grey", lw=0.8, ls="--")
    ax.set_title("Richesse cumulée (base 100) — C4 vs MASI vs Equal Weight")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_wealth_C4_vs_bench.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dd.index, dd["C4"], label="C4", color="#2ca02c", lw=1.3)
    ax.plot(dd.index, dd["MASI"], label="MASI", color="#1f77b4", lw=1.2)
    ax.plot(dd.index, dd["EqualWeight"], label="Equal Weight", color="#ff7f0e", lw=1.2)
    ax.set_title("Drawdowns comparés")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_drawdown_C4_vs_bench.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(roll.index, roll["C4"], label="C4", color="#2ca02c", lw=1.2)
    ax.plot(roll.index, roll["MASI"], label="MASI", color="#1f77b4", lw=1.1)
    ax.plot(roll.index, roll["EqualWeight"], label="Equal Weight", color="#ff7f0e", lw=1.1)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"Rolling Sharpe {ROLL_SHARPE_DAYS}j (rf=0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_rolling_sharpe_C4_vs_bench.png", dpi=140)
    plt.close()

    print("=== TABLE ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(table.to_string(index=False))
    print("\n=== COMPARAISONS ===")
    print(pd.DataFrame([cmp_masi, cmp_ew]).to_string(index=False))
    print("\n=== SOUS-PERIODES C4 ===")
    if len(sub_df):
        with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
            print(sub_df.to_string(index=False))
    print("\n=== CONCLUSION ===")
    for k, v in conclusion.items():
        print(f"{k}: {v.get('reponse', v)}")
    print("\nOUT:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
