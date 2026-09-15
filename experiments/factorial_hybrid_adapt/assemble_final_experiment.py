"""Assemble the FINAL experimental deliverable (no web platform).

Requires Stage2 artefacts:
  outputs/portfolios_stage2_wmax10/         (3 obj, w_max=10%, no VMQ filter)
  outputs/portfolios_stage2_none_wmax10/    (2 obj, w_max=10%, no VMQ filter)
"""
from __future__ import annotations

import json
import shutil
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
from bvc_recommender.benchmarking.simulation import portfolio_daily_returns, turnover_cost
from bvc_recommender.config import DATA_PROCESSED_DIR, FEATURES_DIR, LIQUIDITY_VMQ_THRESHOLD_MAD
from bvc_recommender.models.liquidity_filter import sigmoid_liquidity_factor
from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR
from experiments.factorial_hybrid_adapt.recommendations import CELL_META
from experiments.factorial_hybrid_adapt.stage2_allocation import EVAL_END_MONTH, EVAL_START_MONTH
from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel

BASE = Path(__file__).resolve().parent
FINAL = BASE / "outputs" / "final_experiment_2010_2025"
PARETO_DIR = BASE / "outputs" / "portfolios_stage2_wmax10"
NONE_DIR = BASE / "outputs" / "portfolios_stage2_none_wmax10"
RNG = np.random.default_rng(42)
N_BOOT = 10_000
STUDY_START = "2010-01"
STUDY_END = "2025-06"
FEASIBLE_START = EVAL_START_MONTH
FEASIBLE_END = EVAL_END_MONTH
SUBPERIODS = [
    ("2010–2014", "2010-01", "2014-12"),
    ("2015–2019", "2015-01", "2019-12"),
    ("2020–2022", "2020-01", "2022-12"),
    ("2023–2025", "2023-01", "2025-06"),
]


def _dirs() -> dict[str, Path]:
    names = [
        "01_protocol",
        "02_stage1_recommendations",
        "03_stage2_pareto",
        "04_final_portfolios",
        "05_backtest",
        "06_benchmarks",
        "07_statistical_validation",
        "08_robustness",
        "09_figures",
        "10_reports",
    ]
    out = {}
    FINAL.mkdir(parents=True, exist_ok=True)
    for n in names:
        p = FINAL / n
        p.mkdir(parents=True, exist_ok=True)
        out[n[:2]] = p
    return out


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
        "pct_mois_A_gt_B": round(float((diff > 0).mean()), 4),
        "IC95_lo": round(lo, 6),
        "IC95_hi": round(hi, 6),
        "t_stat": round(float(t_stat), 4),
        "p_ttest": round(float(t_p), 4),
        "p_wilcoxon": None if pd.isna(w_p) else round(float(w_p), 4),
        "significatif_5pct": bool((t_p < 0.05) or (pd.notna(w_p) and w_p < 0.05)),
        "significatif_5pct_ttest": bool(t_p < 0.05),
        "significatif_5pct_wilcoxon": bool(w_p < 0.05) if pd.notna(w_p) else False,
    }


def _period_end(months, i, reb, prices, eval_end: str) -> pd.Timestamp:
    """Last holding window = one month after the last decision, not the last price in the DB."""
    if i + 1 < len(months):
        return pd.Timestamp(reb[months[i + 1]])
    cap = pd.Timestamp(f"{eval_end}-01") + pd.offsets.MonthEnd(2)
    return min(pd.Timestamp(prices.index.max()), pd.Timestamp(cap))


def backtest_holdings(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    h = holdings.copy()
    h["month"] = h["mois"].astype(str) if "mois" in h.columns else h["month"].astype(str)
    h["date_rebalance"] = pd.to_datetime(h["date_rebalance"], errors="coerce")
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
    h = h[(h["month"] >= start) & (h["month"] <= end)].copy()
    months = sorted(h["month"].unique())
    reb = h.groupby("month")["date_rebalance"].first().to_dict()
    monthly_rows = []
    daily_parts = []
    prev_w: dict[str, float] = {}
    for i, mois in enumerate(months):
        g = h[h["month"] == mois]
        t0 = pd.Timestamp(reb[mois])
        t1 = _period_end(months, i, reb, prices, end)
        wdict = {
            str(r.ticker): float(r.weight)
            for r in g.itertuples()
            if pd.notna(r.weight) and float(r.weight) > 0
        }
        s = sum(wdict.values())
        if s > 0:
            wdict = {k: v / s for k, v in wdict.items()}
        if prev_w:
            to = float(sum(abs(wdict.get(t, 0.0) - prev_w.get(t, 0.0)) for t in set(wdict) | set(prev_w)))
        else:
            to = float(sum(wdict.values()))
        cost = float(turnover_cost(prev_w, wdict, TRANSACTION_COST))
        dg = portfolio_daily_returns(wdict, prices, t0, t1)
        dn = dg.copy()
        if not dn.empty and cost > 0:
            dn.iloc[0] = dn.iloc[0] - cost
        liq = np.nan
        if "L_sigmoid" in g.columns:
            ww = g.set_index("ticker")["weight"]
            ll = pd.to_numeric(g.set_index("ticker")["L_sigmoid"], errors="coerce")
            liq = float((ww * ll).sum() / ww.sum()) if ww.sum() else np.nan
        elif "liquidity_L_weighted" in g.columns:
            liq = float(pd.to_numeric(g["liquidity_L_weighted"], errors="coerce").iloc[0])
        monthly_rows.append(
            {
                "month": mois,
                "decision_date": t0.date().isoformat(),
                "net_return": float((1 + dn).prod() - 1) if not dn.empty else np.nan,
                "gross_return": float((1 + dg).prod() - 1) if not dg.empty else np.nan,
                "turnover": to,
                "transaction_cost": cost,
                "n_positions": int((pd.Series(wdict) > 0).sum()),
                "liquidity_L": liq,
                "n_days": int(len(dn)),
            }
        )
        if not dn.empty:
            daily_parts.append(dn)
        prev_w = wdict
    monthly = pd.DataFrame(monthly_rows)
    daily = pd.concat(daily_parts).sort_index() if daily_parts else pd.Series(dtype=float)
    if not daily.empty and daily.index.duplicated().any():
        daily = daily[~daily.index.duplicated(keep="last")]
    wealth = (1 + daily).cumprod() if not daily.empty else pd.Series(dtype=float)
    return monthly, daily, wealth


def metrics_bundle(daily: pd.Series, monthly: pd.DataFrame, name: str, masi: pd.Series) -> dict:
    if daily is None or daily.empty:
        return {"Strategie": name, "error": "empty"}
    bench = masi.reindex(daily.index).fillna(0.0)
    perf = compute_backtest_metrics(daily, bench, name=name)
    wealth = (1 + daily).cumprod()
    r_m = monthly["net_return"].astype(float)
    return {
        "Strategie": name,
        "n_months": int(len(monthly)),
        "Rendement_total": float(wealth.iloc[-1] - 1),
        "Rendement": perf.get("annualized_return"),
        "Volatilite": perf.get("volatility_ann"),
        "Sharpe": perf.get("sharpe"),
        "Sortino": perf.get("sortino"),
        "CVaR": perf.get("cvar_95"),
        "Max_DD": perf.get("max_drawdown"),
        "Turnover": round(float(monthly["turnover"].mean()), 4) if "turnover" in monthly else None,
        "Liquidite": round(float(monthly["liquidity_L"].mean()), 4)
        if "liquidity_L" in monthly and monthly["liquidity_L"].notna().any()
        else None,
        "n_positions_moy": round(float(monthly["n_positions"].mean()), 2)
        if "n_positions" in monthly
        else None,
        "pct_mois_positifs": float((r_m > 0).mean()),
        "ret_mensuel_moyen": float(r_m.mean()),
        "wealth_final_100": float(100 * wealth.iloc[-1]),
        "duree_DD_jours": max_dd_duration_days(wealth),
    }


def load_selected_weights(port_dir: Path, cell: int, portfolio: str = "P_selected") -> pd.DataFrame:
    w = pd.read_parquet(port_dir / f"stage2_weights_C{cell}.parquet")
    w = w[w["portfolio"] == portfolio].copy()
    w["month"] = w["mois"].astype(str)
    o = pd.read_parquet(port_dir / f"stage2_objectives_C{cell}.parquet")
    o = o[o["portfolio"] == portfolio][["mois", "liquidity_L_weighted", "cvar_95", "expected_alpha_opt"]].copy()
    o["mois"] = o["mois"].astype(str)
    w = w.merge(o, left_on="month", right_on="mois", how="left", suffixes=("", "_obj"))
    return w


def ew_backtest(inp_cell: pd.DataFrame, prices, masi_all, months, reb, eval_end: str) -> tuple[pd.DataFrame, pd.Series]:
    parts = []
    rows = []
    prev = {}
    for i, mois in enumerate(months):
        t0 = pd.Timestamp(reb[mois])
        t1 = _period_end(months, i, reb, prices, eval_end)
        univ = inp_cell[(inp_cell["mois"] == mois) & (inp_cell["recommendation"] != "SELL")]
        univ = univ.drop_duplicates("ticker")
        tickers = [t for t in univ["ticker"].astype(str) if t in prices.columns]
        if len(tickers) < 2:
            continue
        w = 1.0 / len(tickers)
        wdict = {t: w for t in tickers}
        Lmap = univ.set_index("ticker")["L"] if "L" in univ.columns else pd.Series(dtype=float)
        Lport = float(sum(wdict[t] * float(Lmap.get(t, np.nan)) for t in tickers if pd.notna(Lmap.get(t, np.nan)))) if len(Lmap) else np.nan
        to = float(sum(abs(wdict.get(t, 0) - prev.get(t, 0)) for t in set(wdict) | set(prev))) if prev else 1.0
        cost = float(turnover_cost(prev, wdict, TRANSACTION_COST))
        dg = portfolio_daily_returns(wdict, prices, t0, t1)
        dn = dg.copy()
        if not dn.empty and cost > 0:
            dn.iloc[0] -= cost
        rows.append(
            {
                "month": mois,
                "net_return": float((1 + dn).prod() - 1) if not dn.empty else np.nan,
                "turnover": to,
                "liquidity_L": Lport,
                "n_positions": len(tickers),
            }
        )
        if not dn.empty:
            parts.append(dn)
        prev = wdict
    monthly = pd.DataFrame(rows)
    daily = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    if not daily.empty and daily.index.duplicated().any():
        daily = daily[~daily.index.duplicated(keep="last")]
    return monthly, daily


def masi_on_calendar(masi_all: pd.Series, months, reb, prices, eval_end: str) -> tuple[pd.DataFrame, pd.Series]:
    parts = []
    rows = []
    for i, mois in enumerate(months):
        t0 = pd.Timestamp(reb[mois])
        t1 = _period_end(months, i, reb, prices, eval_end)
        cal = prices.loc[(prices.index > t0) & (prices.index <= t1)].index
        m = masi_all.reindex(cal).fillna(0.0)
        rows.append({"month": mois, "net_return": float((1 + m).prod() - 1) if len(m) else np.nan})
        if len(m):
            parts.append(m)
    monthly = pd.DataFrame(rows)
    daily = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    if not daily.empty and daily.index.duplicated().any():
        daily = daily[~daily.index.duplicated(keep="last")]
    return monthly, daily


def write_protocol(d: Path) -> None:
    proto = {
        "study_window_requested": f"{STUDY_START} → {STUDY_END}",
        "feasible_window_frozen_before_backtest": f"{FEASIBLE_START} → {FEASIBLE_END}",
        "impossible_months": "2010-01 → 2018-06",
        "reason_impossible": (
            "Stage 1 walk-forward figé : train 36 mois depuis ml_dataset "
            "(début 2015-07, INDICATORS_MIN_DATE=2015-06-03). "
            "Premier bloc de test = 2018-07. Aucun réentraînement C1–C4."
        ),
        "n_months_requested": 186,
        "n_months_feasible": 84,
        "vmq_eligibility_filter": False,
        "liquidity_role": "NSGA-III objective only (max L)",
        "sell": "excluded from Stage 2 universe",
        "nsga_objectives": ["max Alpha (score_opt)", "min CVaR 95%", "max L=sigmoid(VMQ)"],
        "selection_principal": "knee / distance to utopia (min-max normalized F)",
        "w_max": 0.10,
        "w_min": 0.0,
        "transaction_cost": TRANSACTION_COST,
        "no_lookahead": True,
        "models_unchanged": True,
        "hyperparameters_unchanged": True,
    }
    (d / "protocol.json").write_text(json.dumps(proto, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "impossible_months.csv").write_text(
        "month_start,month_end,status,reason\n"
        "2010-01,2018-06,impossible,Pas de recommandations Stage 1 (WF 36m train, ml_dataset dès 2015-07)\n"
        f"{FEASIBLE_START},{FEASIBLE_END},feasible,Reco C1-C4 existantes + VMQ as-of t\n",
        encoding="utf-8",
    )


def liquidity_coverage(d: Path) -> pd.DataFrame:
    tech = pd.read_parquet(FEATURES_DIR / "features_techniques.parquet")
    tech["date_cours"] = pd.to_datetime(tech["date_cours"], errors="coerce")
    tech["year"] = tech["date_cours"].dt.year
    vmq = pd.to_numeric(tech["vmq_20j"], errors="coerce")
    rows = []
    for y, g in tech.groupby("year"):
        v = pd.to_numeric(g["vmq_20j"], errors="coerce")
        rows.append(
            {
                "annee": int(y),
                "n_obs": int(len(g)),
                "pct_vmq_dispo": round(100 * v.notna().mean(), 2),
                "vmq_median": float(v.median()) if v.notna().any() else np.nan,
                "pct_vmq_ge_500k": round(100 * (v.fillna(-1) >= LIQUIDITY_VMQ_THRESHOLD_MAD).mean(), 2),
                "n_titres": int(g["ticker"].nunique()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(d / "liquidity_coverage_by_year.csv", index=False)
    demo = pd.Series([0, 1e5, 2.5e5, 5e5, 1e6, 5e6])
    sig = sigmoid_liquidity_factor(demo)
    pd.DataFrame({"VMQ": demo, "L": sig}).to_csv(d / "sigmoid_reference.csv", index=False)
    note = {
        "vmq_formula": "rolling_mean_20(prix_cloture * titres_echanges.fillna(0))",
        "fillna0_on_volume": (
            "Formule technique existante : volume NaN traité comme 0 AVANT le rolling. "
            "Ce n'est pas une imputation de VMQ par une valeur future. "
            "Les VMQ NaN restants = fenêtres < 20 séances."
        ),
        "no_future": True,
        "features_min_date": str(tech["date_cours"].min().date()),
        "features_max_date": str(tech["date_cours"].max().date()),
        "INDICATORS_MIN_DATE": "2015-06-03",
        "volume_null_live_after_2010_fill": "voir audit volumes ; 2023+ encore des NULL (histo_volume s'arrête fin 2022)",
    }
    (d / "liquidity_notes.json").write_text(json.dumps(note, indent=2, ensure_ascii=False), encoding="utf-8")
    return df


def style_fig():
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 10})


def main() -> int:
    if not (PARETO_DIR / "stage2_weights_C4.parquet").is_file():
        raise FileNotFoundError(f"NSGA pareto manquant : {PARETO_DIR}")
    has_none = (NONE_DIR / "stage2_weights_C4.parquet").is_file()
    D = _dirs()
    style_fig()
    write_protocol(D["01"])
    liquidity_coverage(D["01"])

    rec = REPORTS_DIR / "stage1_recommendations_14blocks.parquet"
    shutil.copy2(rec, D["02"] / rec.name)
    s2 = REPORTS_DIR / "stage2_inputs_14blocks.parquet"
    shutil.copy2(s2, D["02"] / s2.name)
    inp = pd.read_parquet(s2)
    inp["date"] = pd.to_datetime(inp["date"])
    inp["mois"] = inp["date"].dt.to_period("M").astype(str)
    inp["recommendation"] = inp["recommendation"].astype(str).str.upper()
    inp["vmq"] = pd.to_numeric(inp["liquidity_vmq_20j"], errors="coerce")
    inp["L"] = sigmoid_liquidity_factor(inp["vmq"])  # NaN stay NaN in series? fillna in factor
    # sigmoid fillna 0 internally - document; for EW we use L as-is after factor

    for fn in PARETO_DIR.glob("stage2_*"):
        shutil.copy2(fn, D["03"] / fn.name)

    cours, indices, _ = load_market_panels()
    prices = _price_panel(cours)
    masi_all = _masi_returns(indices)

    cells = (1, 2, 3, 4)
    daily_map = {}
    monthly_map = {}
    metric_rows = []

    # Principal: 3-obj knee
    for cell in cells:
        w = load_selected_weights(PARETO_DIR, cell, "P_selected")
        w.to_parquet(D["04"] / f"C{cell}_holdings_knee.parquet", index=False)
        w.to_csv(D["04"] / f"C{cell}_holdings_knee.csv", index=False)
        months = sorted(w["month"].unique())
        reb = w.groupby("month")["date_rebalance"].first().to_dict()
        mon, day, wealth = backtest_holdings(w, prices, start=FEASIBLE_START, end=FEASIBLE_END)
        mon.to_csv(D["05"] / f"C{cell}_monthly.csv", index=False)
        pd.DataFrame({"date": day.index, "ret_net": day.values, "wealth100": 100 * (1 + day).cumprod().values}).to_csv(
            D["05"] / f"C{cell}_daily.csv", index=False
        )
        daily_map[f"C{cell}"] = day
        monthly_map[f"C{cell}"] = mon.set_index("month")["net_return"]
        name = CELL_META[cell]["label"]
        metric_rows.append(metrics_bundle(day, mon, name, masi_all))

        # EW same universe
        sub = inp[inp["cell"] == cell].copy()
        ew_m, ew_d = ew_backtest(sub, prices, masi_all, months, reb, FEASIBLE_END)
        daily_map[f"EW_C{cell}"] = ew_d
        monthly_map[f"EW_C{cell}"] = ew_m.set_index("month")["net_return"]
        metric_rows.append(metrics_bundle(ew_d, ew_m, f"EqualWeight C{cell}", masi_all))
        ew_m.to_csv(D["06"] / f"EW_C{cell}_monthly.csv", index=False)

        if cell == 4:
            masi_m, masi_d = masi_on_calendar(masi_all, months, reb, prices, FEASIBLE_END)
            daily_map["MASI"] = masi_d
            monthly_map["MASI"] = masi_m.set_index("month")["net_return"]
            metric_rows.append(metrics_bundle(masi_d, masi_m.assign(turnover=np.nan, liquidity_L=np.nan, n_positions=np.nan), "MASI", masi_all))
            masi_m.to_csv(D["06"] / "MASI_monthly.csv", index=False)

    table = pd.DataFrame(metric_rows)
    table.to_csv(D["06"] / "table_C1C4_MASI_EW.csv", index=False)

    # Stats
    stats_rows = [
        paired_stats(monthly_map["C3"], monthly_map["MASI"], "C3 - MASI"),
        paired_stats(monthly_map["C3"], monthly_map["EW_C3"], "C3 - Equal Weight C3"),
        paired_stats(monthly_map["C3"], monthly_map["C1"], "C3 - C1"),
        paired_stats(monthly_map["C3"], monthly_map["C2"], "C3 - C2"),
        paired_stats(monthly_map["C3"], monthly_map["C4"], "C3 - C4"),
        paired_stats(monthly_map["C4"], monthly_map["MASI"], "C4 - MASI"),
        paired_stats(monthly_map["C4"], monthly_map["C1"], "C4 - C1"),
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(D["07"] / "tests_principaux.csv", index=False)

    # Subperiods
    sub_rows = []
    for cell in cells:
        mon = pd.read_csv(D["05"] / f"C{cell}_monthly.csv")
        day = pd.read_csv(D["05"] / f"C{cell}_daily.csv", parse_dates=["date"]).set_index("date")["ret_net"]
        tag = pd.Series(index=day.index, dtype=object)
        months = mon["month"].tolist()
        for i, r in mon.iterrows():
            t0 = pd.Timestamp(r["decision_date"])
            if i + 1 < len(mon):
                t1 = pd.Timestamp(mon.iloc[i + 1]["decision_date"])
            else:
                t1 = day.index.max()
            mask = (day.index > t0) & (day.index <= t1)
            tag.loc[mask] = r["month"]
        for lab, lo, hi in SUBPERIODS:
            keep = [m for m in months if lo <= m <= hi]
            if not keep:
                sub_rows.append({"cell": f"C{cell}", "periode": lab, "n_months": 0, "note": "aucun mois de décision"})
                continue
            dmask = tag.isin(keep)
            csub = day.loc[dmask]
            msub = mon[mon["month"].isin(keep)]
            met = metrics_bundle(csub, msub, f"C{cell} {lab}", masi_all)
            sub_rows.append({"cell": f"C{cell}", "periode": lab, **{k: met.get(k) for k in (
                "n_months", "Rendement", "Volatilite", "Sharpe", "Sortino", "CVaR", "Max_DD", "Turnover", "Liquidite"
            )}})
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(D["08"] / "subperiods_C1C4.csv", index=False)

    # Sensitivity C4 from same Pareto
    sens_rows = []
    for port, label in (
        ("P_selected", "A_knee_utopie"),
        ("P_agressif", "B_max_alpha"),
        ("P_max_liq", "C_max_liquidite"),
        ("P_defensif", "D_min_cvar"),
        ("P_equilibre", "E_composite_equilibre_identique_knee"),
    ):
        w = load_selected_weights(PARETO_DIR, 4, port)
        if w.empty:
            continue
        mon, day, _ = backtest_holdings(w, prices, start=FEASIBLE_START, end=FEASIBLE_END)
        met = metrics_bundle(day, mon, label, masi_all)
        sens_rows.append(met)
        mon.to_csv(D["08"] / f"sens_{label}_monthly.csv", index=False)
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(D["08"] / "sensitivity_selection_C4.csv", index=False)

    # Robustness 2-obj vs 3-obj
    rob_rows = []
    if has_none:
        for cell in cells:
            w = load_selected_weights(NONE_DIR, cell, "P_selected")
            mon, day, _ = backtest_holdings(w, prices, start=FEASIBLE_START, end=FEASIBLE_END)
            met = metrics_bundle(day, mon, f"C{cell} A_alpha_cvar", masi_all)
            rob_rows.append(met)
            w3 = load_selected_weights(PARETO_DIR, cell, "P_selected")
            mon3, day3, _ = backtest_holdings(w3, prices, start=FEASIBLE_START, end=FEASIBLE_END)
            met3 = metrics_bundle(day3, mon3, f"C{cell} B_alpha_cvar_L", masi_all)
            rob_rows.append(met3)
        pd.DataFrame(rob_rows).to_csv(D["08"] / "robustness_liquidity_objective.csv", index=False)
    else:
        (D["08"] / "robustness_liquidity_objective.csv").write_text(
            "note\nVERSION A (2 obj) non encore calculee — relancer NSGA liquidity_mode=none\n",
            encoding="utf-8",
        )

    # Figures
    figp = D["09"]
    # wealth C1-C4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, col in (("C1", "#1f77b4"), ("C2", "#ff7f0e"), ("C3", "#2ca02c"), ("C4", "#d62728")):
        s = daily_map[k]
        ax.plot(s.index, 100 * (1 + s).cumprod(), label=k, lw=1.6, color=col)
    ax.axhline(100, color="grey", lw=0.8, ls="--")
    ax.set_title("Richesse cumulée nette C1–C4 (base 100)")
    ax.set_ylabel("Richesse")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figp / "fig02_wealth_C1C4.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, lab, col in (("C4", "C4", "#d62728"), ("MASI", "MASI", "#1f77b4"), ("EW_C4", "Equal Weight", "#ff7f0e")):
        s = daily_map[k]
        ax.plot(s.index, 100 * (1 + s).cumprod(), label=lab, lw=1.6, color=col)
    ax.axhline(100, color="grey", lw=0.8, ls="--")
    ax.set_title("C4 vs MASI vs Equal Weight")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figp / "fig03_C4_vs_bench.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, lab, col in (("C4", "C4", "#d62728"), ("MASI", "MASI", "#1f77b4"), ("EW_C4", "EW", "#ff7f0e")):
        s = daily_map[k]
        w = (1 + s).cumprod()
        ax.plot(s.index, w / w.cummax() - 1, label=lab, lw=1.3, color=col)
    ax.set_title("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figp / "fig04_drawdown.png")
    plt.close()

    # risk return scatter from table principal cells
    tsub = table[table["Strategie"].str.match(r"^C[1-4]")].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tsub["Volatilite"], tsub["Rendement"], s=80)
    for _, r in tsub.iterrows():
        ax.annotate(r["Strategie"][:2], (r["Volatilite"], r["Rendement"]))
    ax.set_xlabel("Volatilité annualisée")
    ax.set_ylabel("Rendement annualisé")
    ax.set_title("Rendement / risque")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figp / "fig05_rendement_risque.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tsub["Liquidite"], tsub["Rendement"], s=80)
    for _, r in tsub.iterrows():
        ax.annotate(r["Strategie"][:2], (r["Liquidite"], r["Rendement"]))
    ax.set_xlabel("Liquidité moyenne L")
    ax.set_ylabel("Rendement annualisé")
    ax.set_title("Rendement / liquidité")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figp / "fig06_rendement_liquidite.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tsub["CVaR"], tsub["Rendement"], s=80)
    for _, r in tsub.iterrows():
        ax.annotate(r["Strategie"][:2], (r["CVaR"], r["Rendement"]))
    ax.set_xlabel("CVaR 95%")
    ax.set_ylabel("Rendement annualisé")
    ax.set_title("Rendement / CVaR")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figp / "fig07_rendement_cvar.png")
    plt.close()

    # Pareto examples
    po = pd.read_parquet(PARETO_DIR / "stage2_pareto_objectives.parquet")
    po = po[po["cell"] == 4]
    examples = []
    for target in ("2018-12", "2021-06", "2025-01"):
        if target in set(po["mois"].astype(str)):
            examples.append(target)
    if not examples:
        examples = sorted(po["mois"].astype(str).unique())[:1] + sorted(po["mois"].astype(str).unique())[-1:]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    titles = ["Alpha vs CVaR", "Alpha vs Liquidité", "CVaR vs Liquidité"]
    for mois in examples[:1]:
        g = po[po["mois"].astype(str) == mois]
        axes[0].scatter(g["cvar_95"], g["expected_alpha_opt"], s=18, alpha=0.6)
        k = g[g["is_knee"] == True]
        if len(k):
            axes[0].scatter(k["cvar_95"], k["expected_alpha_opt"], s=80, c="red", label="Knee")
        axes[1].scatter(g["liquidity_L_weighted"], g["expected_alpha_opt"], s=18, alpha=0.6)
        if len(k):
            axes[1].scatter(k["liquidity_L_weighted"], k["expected_alpha_opt"], s=80, c="red")
        axes[2].scatter(g["cvar_95"], g["liquidity_L_weighted"], s=18, alpha=0.6)
        if len(k):
            axes[2].scatter(k["cvar_95"], k["liquidity_L_weighted"], s=80, c="red")
    axes[0].set_xlabel("CVaR")
    axes[0].set_ylabel("Alpha")
    axes[1].set_xlabel("L")
    axes[1].set_ylabel("Alpha")
    axes[2].set_xlabel("CVaR")
    axes[2].set_ylabel("L")
    for ax, t in zip(axes, titles):
        ax.set_title(t + f" (C4 {examples[0]})")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figp / "fig08_pareto_example.png")
    plt.close()

    # extra pareto months
    fig, axes = plt.subplots(1, min(3, len(examples)), figsize=(5 * min(3, len(examples)), 4))
    if min(3, len(examples)) == 1:
        axes = [axes]
    for ax, mois in zip(axes, examples[:3]):
        g = po[po["mois"].astype(str) == mois]
        ax.scatter(g["cvar_95"], g["expected_alpha_opt"], s=16, alpha=0.55)
        k = g[g["is_knee"] == True]
        if len(k):
            ax.scatter(k["cvar_95"], k["expected_alpha_opt"], s=70, c="red", label="Knee")
        ax.set_title(f"C4 {mois}")
        ax.set_xlabel("CVaR")
        ax.set_ylabel("Alpha")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figp / "fig08b_pareto_trois_periodes.png")
    plt.close()

    # subperiod bars
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pivot = sub_df.pivot_table(index="periode", columns="cell", values="Rendement")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Rendement annualisé")
    ax.set_title("Sous-périodes — rendement C1–C4")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figp / "fig09_subperiods.png")
    plt.close()

    if len(sens_df):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(sens_df["Strategie"], sens_df["Sharpe"])
        ax.set_ylabel("Sharpe")
        ax.set_title("Sensibilité de la règle de sélection (C4)")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(figp / "fig10_sensitivity_knee.png")
        plt.close()

    # monthly dist C4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(monthly_map["C4"].dropna().values, bins=20, color="#d62728", alpha=0.85)
    ax.set_title("Distribution des rendements mensuels C4")
    ax.set_xlabel("Rendement mensuel net")
    fig.tight_layout()
    fig.savefig(figp / "fig11_distrib_mensuelle_C4.png")
    plt.close()

    # Architecture placeholder text figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.text(
        0.02,
        0.5,
        "Données → Features as-of t → Stage 1 C1–C4 (BUY/NEU/SELL)\n"
        "→ Univers BUY+NEUTRAL (SELL exclu, PAS de filtre VMQ)\n"
        "→ NSGA-III (Alpha, CVaR, L) → Front Pareto → Knee/Utopie\n"
        "→ Poids figés → rendements (t, t_next] − TC 0.3%",
        va="center",
        fontsize=12,
        family="monospace",
    )
    ax.set_title("Architecture du système")
    fig.tight_layout()
    fig.savefig(figp / "fig01_architecture.png")
    plt.close()

    # Executive + report
    c4_row = table[table["Strategie"].str.startswith("C4")].iloc[0]
    masi_row = table[table["Strategie"] == "MASI"].iloc[0]
    ew_row = table[table["Strategie"].str.startswith("EqualWeight C4")].iloc[0]
    best = table[table["Strategie"].str.match(r"^C[1-4]")].sort_values("Sharpe", ascending=False).iloc[0]

    checklist = {
        "periode_2010_2025_demandee": True,
        "mois_impossibles_documentes": "2010-01..2018-06",
        "mois_effectivement_testes": 84,
        "filtre_VMQ_500k": False,
        "liquidite_objectif_pareto_uniquement": True,
        "SELL_exclu": True,
        "nsga_3_objectifs": True,
        "front_pareto_complet": (PARETO_DIR / "stage2_pareto_objectives.parquet").is_file(),
        "selection_knee": True,
        "no_lookahead": True,
        "couts_transaction_0_003": True,
        "C1_C4_comparables": True,
        "MASI_comparable": True,
        "EW_comparable": True,
        "tests_statistiques": True,
        "sous_periodes": True,
        "robustesse_liquidite": has_none,
        "plateforme_web_non_modifiee": True,
    }
    (D["01"] / "checklist_final.json").write_text(json.dumps(checklist, indent=2), encoding="utf-8")

    report_md = build_report(table, stats_df, sub_df, sens_df, c4_row, masi_row, ew_row, best, has_none, rob_rows)
    (D["10"] / "Rapport_final_experimental.md").write_text(report_md, encoding="utf-8")
    (D["10"] / "resume_executif.json").write_text(
        json.dumps(
            {
                "n_months_tested": 84,
                "n_months_impossible": "2010-01 to 2018-06",
                "best_config": best["Strategie"],
                "w_max": 0.10,
                "C3_ann_return": float(table[table["Strategie"].str.startswith("C3")].iloc[0]["Rendement"]),
                "C3_sharpe": float(table[table["Strategie"].str.startswith("C3")].iloc[0]["Sharpe"]),
                "C4_ann_return": c4_row["Rendement"],
                "C4_sharpe": c4_row["Sharpe"],
                "C4_cvar": c4_row["CVaR"],
                "C4_maxdd": c4_row["Max_DD"],
                "C4_L": c4_row["Liquidite"],
                "vs_MASI": stats_rows[0],
                "vs_EW": stats_rows[1],
                "deliverable_root": str(FINAL),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print("FINAL:", FINAL)
    print(table.to_string(index=False))
    print(stats_df.to_string(index=False))
    return 0


def build_report(table, stats_df, sub_df, sens_df, c4, masi, ew, best, has_none, rob_rows) -> str:
    def fmt(df):
        return df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)

    try:
        t = table.to_markdown(index=False)
        s = stats_df.to_markdown(index=False)
        u = sub_df.to_markdown(index=False)
        se = sens_df.to_markdown(index=False) if len(sens_df) else "(n/a)"
    except Exception:
        t, s, u, se = table.to_string(index=False), stats_df.to_string(index=False), sub_df.to_string(index=False), sens_df.to_string(index=False)

    return f"""# Rapport expérimental FINAL — système de recommandation et d'allocation BVC

## 1. Objectif du projet
Construire un système en deux étages (recommandation titre puis allocation multi-objectifs) pour le marché actions marocain, évalué hors échantillon par walk-forward, sans look-ahead, avec coûts de transaction.

## 2. Données
Cours `market_data_cours_historique` (2010-01-04 → 2026-05-18), volumes `titres_echanges` complétés depuis `histo_volume` (2010–2022) sans écraser l'existant. Indices MASI. Features fondamentales à partir du 3 juin 2015 (`INDICATORS_MIN_DATE`).

## 3. Période d'étude
Demandée : **2010-01 → 2025-06** (figée avant backtest).
**Mois impossibles** : 2010-01 → 2018-06 — pas de recommandations Stage 1 (walk-forward 36 mois de train, panel ML dès 2015-07). Aucun réentraînement.
**Mois testés** : **2018-07 → 2025-06 (84 mois)**. Out-of-sample walk-forward (blocs de test 6 mois).

## 4. Architecture générale
Données as-of t → features → C1–C4 prédisent `alpha_ajuste_risque` → BUY/NEUTRAL/SELL → univers BUY+NEUTRAL (SELL exclu, **aucun filtre VMQ**) → NSGA-III (Alpha, CVaR, L) → front de Pareto → Knee/Utopie → backtest (t, t_next] − 0,3 % × turnover L1.

## 5–10. Stage 1 et cellules
C1 Ridge ; C2 Ridge + régime HMM ; C3 Hybrid Ridge+RF 50/50 ; C4 Hybrid + régime.
Recommandation : excess = prédiction (benchmark 0) ; BUY si >+tau, SELL si <-tau, sinon NEUTRAL. tau = quantile 0,33 des |pred| des plis passés.

## 11. Alpha
`alpha_ajuste_risque = (forward_ret_titre − forward_ret_MASI) / vol_baissiere_20d`. Le Stage 2 maximise `score_opt = predicted_return × pref` (BUY=1, NEUTRAL=0.5).

## 12. CVaR
CVaR 95 % historique des rendements journaliers du portefeuille, lookback 504 séances, **strictement ≤ date de rebalancement**.

## 13–15. Liquidité, VMQ, sigmoïde
`VMQ_20j = rolling_mean_20(prix × titres_echanges.fillna(0))` (formule existante).
`L = sigmoid(VMQ)` avec k=8/500000, m=250000 — **non modifiés**.
Rôle : **objectif Pareto uniquement**. Pas de VMQ ≥ 500k.

## 16–18. NSGA-III, Pareto, Knee
NSGA-III, pop=36, gen=30, w∈[0;10%]. Front complet persisté (`stage2_pareto_objectives/weights`).
Knee = distance euclidienne à l'utopie après min-max des objectifs F (tous minimisés : −α, CVaR, −L). Aucun rendement futur.

## 19–22. Portefeuille, backtest, benchmarks, métriques
Poids figés à t. EW = équipondération de l'univers BUY+NEUTRAL **de la même cellule**. MASI sur le même calendrier. TC=0,003.

## 23. Tests statistiques
Bootstrap IC 95 %, t-test apparié, Wilcoxon. Seuil 5 %. Distinction économique vs statistique.

## 24–25. Sous-périodes et robustesse
2010–2014 : non calculable. 2015–2019 : seulement 2018-07–2019-12. Sensibilité Knee vs max α / max L / min CVaR. Version A = Alpha+CVaR ; Version B = +Liquidité (principale).

## 26. Résultats C1–C4 / benchmarks
{t}

## 27–28. Tests
{s}

## 29. Sous-périodes
{u}

## Sensibilité (C4)
{se}

## 30. Limites
Échantillon Stage 1 dès 2018-07 seulement ; volumes 2023+ encore partiellement NULL ; NSGA stochastique (graine fixe) ; turnover élevé ; EW déjà très compétitif ; 2020–2022 difficile.

## 31–32. Conclusion et modèle final
Meilleure configuration selon Sharpe sur la fenêtre testable : **{best['Strategie']}** (Sharpe={best['Sharpe']}, rend. ann.={best['Rendement']}).
C4 : rend. {c4['Rendement']}, Sharpe {c4['Sharpe']}, CVaR {c4['CVaR']}, MaxDD {c4['Max_DD']}, L {c4['Liquidite']}.
MASI : {masi['Rendement']} / Sharpe {masi['Sharpe']}. EW C4 : {ew['Rendement']} / Sharpe {ew['Sharpe']}.
Le modèle opérationnel recommandé reste **C4 (Hybrid + régime)** comme spécification principale du projet, sous réserve des tests (voir JSON).

Livrable : `experiments/factorial_hybrid_adapt/outputs/final_experiment_2010_2025/`
"""


if __name__ == "__main__":
    raise SystemExit(main())
