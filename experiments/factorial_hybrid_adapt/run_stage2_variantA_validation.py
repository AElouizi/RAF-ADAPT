"""Validation comparative finale — Variante A (liquidité = objectif Pareto).

Lit uniquement les artefacts déjà calculés. Ne modifie aucun modèle.
Ne relance pas le backtest NSGA.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
ROOT = BASE / "outputs" / "reports" / "stage2_liq_pareto"
PORT = BASE / "outputs" / "portfolios_stage2_liq_pareto"
OUT = ROOT / "validation_finale"
FIG = OUT / "figures"

LABELS = {
    1: "C1 Ridge",
    2: "C2 Ridge + régime",
    3: "C3 Hybrid",
    4: "C4 Hybrid + régime",
}
COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
RNG = np.random.default_rng(42)
N_BOOT = 10_000


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


def max_dd_duration_months(rets: pd.Series) -> int:
    wealth = (1 + rets).cumprod()
    peak = wealth.cummax()
    under = wealth < peak
    if not under.any():
        return 0
    groups = (under != under.shift(fill_value=False)).cumsum()
    lengths = under.groupby(groups).sum()
    under_groups = under.groupby(groups).first()
    lengths = lengths[under_groups]
    return int(lengths.max()) if len(lengths) else 0


def cvar_monthly(r: pd.Series, alpha: float = 0.05) -> float:
    q = r.quantile(alpha)
    tail = r[r <= q]
    return float(-tail.mean()) if len(tail) else float("nan")


def bootstrap_ci(diff: np.ndarray, n: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float]:
    boots = []
    nobs = len(diff)
    for _ in range(n):
        idx = RNG.integers(0, nobs, nobs)
        boots.append(diff[idx].mean())
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(ROOT / "performance_mensuelle.csv")
    monthly["date_rebalance"] = pd.to_datetime(monthly["date_rebalance"])
    monthly = monthly.sort_values(["cell", "mois"]).reset_index(drop=True)

    obj = pd.read_parquet(PORT / "stage2_objectives_all.parquet")
    obj = obj[obj["portfolio"] == "P_selected"].copy()
    obj["L"] = pd.to_numeric(obj["liquidity_L_weighted"], errors="coerce")
    L_by_cell = obj.groupby("cell")["L"].mean().to_dict()

    daily = pd.read_parquet(ROOT / "daily_returns_wealth.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    perf_table = pd.read_csv(ROOT / "table_performance_C1C4.csv")

    def metrics_for_cell(cell: int) -> dict:
        g = monthly[monthly["cell"] == cell].sort_values("mois")
        r = g["ret_month"].astype(float)
        wealth100 = 100 * (1 + r).cumprod()
        total = float(wealth100.iloc[-1] / 100 - 1)
        n_m = len(r)
        ann = float((1 + total) ** (12 / n_m) - 1)
        vol_m = float(r.std(ddof=1))
        mean_m = float(r.mean())
        peak = wealth100.cummax()
        dd = wealth100 / peak - 1
        max_dd = float(dd.min())
        dd_months = max_dd_duration_months(r)
        d = daily[daily["cell"] == cell].sort_values("date")
        if len(d) and "ret" in d.columns:
            w = (1 + d["ret"].astype(float)).cumprod()
            dd_days = max_dd_duration_days(w)
        else:
            dd_days = None
        row = perf_table[perf_table["cell"] == cell].iloc[0]
        return {
            "cell": cell,
            "Configuration": LABELS[cell],
            "ret_mensuel_moyen": round(mean_m, 6),
            "ret_annualise": round(float(row["Return annuel"]), 4),
            "ret_annualise_from_monthly": round(ann, 4),
            "volatilite_ann": round(float(row["Volatilité"]), 4),
            "volatilite_mensuelle": round(vol_m, 6),
            "Sharpe": round(float(row["Sharpe"]), 3),
            "Sortino": round(float(row["Sortino"]), 3),
            "CVaR": round(float(row["CVaR"]), 4),
            "CVaR_mensuel": round(cvar_monthly(r), 4),
            "Max_DD": round(float(row["Max DD"]), 4),
            "Max_DD_mensuel": round(max_dd, 4),
            "duree_DD_jours": int(dd_days) if dd_days is not None else None,
            "duree_DD_mois": int(dd_months),
            "Turnover": round(float(row["Turnover"]), 4),
            "L_portefeuille": round(
                float(L_by_cell.get(cell, row["Liquidité (L pondéré)"])), 4
            ),
            "Nb_positions": round(float(row["Nb positions"]), 2),
            "pct_mois_positifs": round(float((r > 0).mean()), 4),
            "wealth_final_100": round(float(wealth100.iloc[-1]), 2),
            "total_return": round(float(row["Total return"]), 4),
            "months": g["mois"].tolist(),
            "ret_month": [round(float(x), 8) for x in r.tolist()],
            "wealth100": [round(float(x), 4) for x in wealth100.tolist()],
            "drawdown_monthly": [round(float(x), 6) for x in dd.tolist()],
        }

    cells_metrics = {c: metrics_for_cell(c) for c in (1, 2, 3, 4)}

    def paired_compare(a: int, b: int, name: str) -> dict:
        ga = monthly[monthly["cell"] == a].set_index("mois")["ret_month"].astype(float)
        gb = monthly[monthly["cell"] == b].set_index("mois")["ret_month"].astype(float)
        common = ga.index.intersection(gb.index)
        da = ga.loc[common]
        db = gb.loc[common]
        diff = (da - db).to_numpy(dtype=float)
        mean_d = float(diff.mean())
        med_d = float(np.median(diff))
        std_d = float(diff.std(ddof=1))
        pct = float((diff > 0).mean())
        lo, hi = bootstrap_ci(diff)
        t_stat, t_p = stats.ttest_rel(da, db)
        try:
            w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        return {
            "Comparaison": name,
            "cell_a": a,
            "cell_b": b,
            "n_months": int(len(diff)),
            "diff_moyenne_mensuelle": round(mean_d, 6),
            "diff_mediane": round(med_d, 6),
            "diff_ecart_type": round(std_d, 6),
            "pct_mois_A_surperforme": round(pct, 4),
            "IC95_bootstrap_lo": round(lo, 6),
            "IC95_bootstrap_hi": round(hi, 6),
            "test_apparie": "t-test apparié + Wilcoxon",
            "t_stat": round(float(t_stat), 4),
            "p_value_ttest": round(float(t_p), 4),
            "wilcoxon_stat": None if pd.isna(w_stat) else round(float(w_stat), 4),
            "p_value_wilcoxon": None if pd.isna(w_p) else round(float(w_p), 4),
            "diff_ann_approx": round(mean_d * 12, 4),
        }

    pairs = [
        paired_compare(2, 1, "C2 vs C1"),
        paired_compare(3, 1, "C3 vs C1"),
        paired_compare(4, 1, "C4 vs C1"),
        paired_compare(4, 3, "C4 vs C3"),
    ]

    tradeoff_rows = []
    for c in (1, 2, 3, 4):
        m = cells_metrics[c]
        tradeoff_rows.append(
            {
                "Configuration": m["Configuration"],
                "Rendement": m["ret_annualise"],
                "Sharpe": m["Sharpe"],
                "CVaR": m["CVaR"],
                "Max DD": m["Max_DD"],
                "L portefeuille": m["L_portefeuille"],
                "Turnover": m["Turnover"],
            }
        )
    tradeoff = pd.DataFrame(tradeoff_rows)

    base = cells_metrics[1]
    analysis_vs_c1 = []
    for c in (2, 3, 4):
        m = cells_metrics[c]
        analysis_vs_c1.append(
            {
                "cell": c,
                "Configuration": m["Configuration"],
                "d_Rendement": round(m["ret_annualise"] - base["ret_annualise"], 4),
                "d_Sharpe": round(m["Sharpe"] - base["Sharpe"], 3),
                "d_CVaR": round(m["CVaR"] - base["CVaR"], 4),
                "d_MaxDD": round(m["Max_DD"] - base["Max_DD"], 4),
                "d_L": round(m["L_portefeuille"] - base["L_portefeuille"], 4),
                "d_Turnover": round(m["Turnover"] - base["Turnover"], 4),
                "liq_down": m["L_portefeuille"] < base["L_portefeuille"],
                "risk_up_cvar": m["CVaR"] > base["CVaR"],
                "risk_up_dd": m["Max_DD"] < base["Max_DD"],
                "turnover_up": m["Turnover"] > base["Turnover"],
            }
        )

    rank_df = pd.DataFrame(
        [
            {
                "cell": c,
                "Configuration": cells_metrics[c]["Configuration"],
                "ret": cells_metrics[c]["ret_annualise"],
                "sharpe": cells_metrics[c]["Sharpe"],
                "sortino": cells_metrics[c]["Sortino"],
                "cvar": cells_metrics[c]["CVaR"],
                "maxdd": cells_metrics[c]["Max_DD"],
                "L": cells_metrics[c]["L_portefeuille"],
                "to": cells_metrics[c]["Turnover"],
                "pct_pos": cells_metrics[c]["pct_mois_positifs"],
                "dd_days": cells_metrics[c]["duree_DD_jours"] or 0,
            }
            for c in (1, 2, 3, 4)
        ]
    )
    rank_df["r_finance"] = (
        rank_df["ret"].rank(ascending=False)
        + rank_df["sharpe"].rank(ascending=False)
        + rank_df["sortino"].rank(ascending=False)
        + rank_df["pct_pos"].rank(ascending=False)
    ) / 4
    rank_df["r_risk"] = (
        rank_df["cvar"].rank(ascending=True)
        + rank_df["maxdd"].rank(ascending=False)
        + rank_df["dd_days"].rank(ascending=True)
    ) / 3
    rank_df["r_liq"] = rank_df["L"].rank(ascending=False)
    rank_df["r_stab"] = (
        rank_df["to"].rank(ascending=True)
        + rank_df["pct_pos"].rank(ascending=False)
        + rank_df["dd_days"].rank(ascending=True)
    ) / 3
    rank_df["r_composite"] = (
        rank_df["r_finance"] + rank_df["r_risk"] + rank_df["r_liq"] + rank_df["r_stab"]
    ) / 4
    rank_df = rank_df.sort_values("r_composite")

    ranking = []
    for i, row in enumerate(rank_df.itertuples(), 1):
        ranking.append(
            {
                "rank_composite": i,
                "cell": int(row.cell),
                "Configuration": row.Configuration,
                "rank_finance": round(float(row.r_finance), 2),
                "rank_risk": round(float(row.r_risk), 2),
                "rank_liquidity": round(float(row.r_liq), 2),
                "rank_stability": round(float(row.r_stab), 2),
                "composite": round(float(row.r_composite), 3),
                "ret": row.ret,
                "sharpe": row.sharpe,
                "L": row.L,
                "cvar": row.cvar,
                "maxdd": row.maxdd,
            }
        )

    # Figures
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in (1, 2, 3, 4):
        m = cells_metrics[c]
        ax.plot(m["months"], m["wealth100"], label=m["Configuration"], color=COLORS[c], lw=1.8)
    ax.axhline(100, color="grey", lw=0.8, ls="--")
    ax.set_title("Richesse cumulée (base 100) — Variante A Pareto · 2023-01→2025-06")
    ax.set_ylabel("Richesse")
    ax.set_xlabel("Mois")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_wealth_base100.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    for c in (1, 2, 3, 4):
        m = cells_metrics[c]
        ax.plot(
            m["months"], m["drawdown_monthly"], label=m["Configuration"], color=COLORS[c], lw=1.4
        )
    ax.set_title("Drawdowns mensuels — Variante A Pareto")
    ax.set_ylabel("Drawdown")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_drawdown_monthly.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in (1, 2, 3, 4):
        m = cells_metrics[c]
        ax.scatter(m["L_portefeuille"], m["ret_annualise"], s=120, color=COLORS[c], zorder=3)
        ax.annotate(
            f"C{c}",
            (m["L_portefeuille"], m["ret_annualise"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=10,
            fontweight="bold",
            color=COLORS[c],
        )
    ax.set_xlabel("Liquidité moyenne portefeuille L = sigmoid(VMQ) pondéré")
    ax.set_ylabel("Rendement annualisé")
    ax.set_title("Rendement annuel vs liquidité — C1–C4 (Variante A)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_return_vs_liquidity.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in (1, 2, 3, 4):
        m = cells_metrics[c]
        ax.scatter(m["CVaR"], m["ret_annualise"], s=120, color=COLORS[c], zorder=3)
        ax.annotate(
            f"C{c}",
            (m["CVaR"], m["ret_annualise"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=10,
            fontweight="bold",
            color=COLORS[c],
        )
    ax.set_xlabel("CVaR 95% (quotidien, backtest OOS)")
    ax.set_ylabel("Rendement annualisé")
    ax.set_title("Rendement annuel vs CVaR — C1–C4 (Variante A)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_return_vs_cvar.png", dpi=140)
    plt.close()

    summary_table = pd.DataFrame(
        [
            {
                k: v
                for k, v in cells_metrics[c].items()
                if k not in ("months", "ret_month", "wealth100", "drawdown_monthly")
            }
            for c in (1, 2, 3, 4)
        ]
    )
    summary_table.to_csv(OUT / "metrics_C1C4.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "comparisons_stats.csv", index=False)
    tradeoff.to_csv(OUT / "tradeoff_rendement_liquidite.csv", index=False)
    pd.DataFrame(analysis_vs_c1).to_csv(OUT / "delta_vs_C1.csv", index=False)
    pd.DataFrame(ranking).to_csv(OUT / "ranking_multicritere.csv", index=False)

    wealth_panel = []
    for c in (1, 2, 3, 4):
        m = cells_metrics[c]
        for mois, w, r, dd in zip(
            m["months"], m["wealth100"], m["ret_month"], m["drawdown_monthly"]
        ):
            wealth_panel.append(
                {
                    "cell": c,
                    "Configuration": m["Configuration"],
                    "mois": mois,
                    "ret_month": r,
                    "wealth100": w,
                    "drawdown": dd,
                }
            )
    pd.DataFrame(wealth_panel).to_csv(OUT / "wealth_drawdown_mensuel.csv", index=False)

    report = {
        "protocol": {
            "variant": "A — liquidity as Pareto objective",
            "window": "2023-01 → 2025-06",
            "architecture": "SELL exclude → NSGA-III (alpha, CVaR, max L)",
            "no_vmq_eligibility_filter": True,
            "source": "existing portfolios_stage2_liq_pareto / stage2_liq_pareto",
            "code_unchanged": True,
            "backtest_not_rerun": True,
        },
        "metrics": summary_table.to_dict(orient="records"),
        "comparisons": pairs,
        "tradeoff_table": tradeoff.to_dict(orient="records"),
        "delta_vs_C1": analysis_vs_c1,
        "ranking": ranking,
        "wealth_monthly": {
            str(c): {
                "months": cells_metrics[c]["months"],
                "wealth100": cells_metrics[c]["wealth100"],
                "ret_month": cells_metrics[c]["ret_month"],
            }
            for c in (1, 2, 3, 4)
        },
        "drawdown_monthly": {str(c): cells_metrics[c]["drawdown_monthly"] for c in (1, 2, 3, 4)},
        "figures": [str(p) for p in sorted(FIG.glob("*.png"))],
    }
    (OUT / "validation_finale_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== METRICS ===")
    cols = [
        "Configuration",
        "ret_mensuel_moyen",
        "ret_annualise",
        "volatilite_ann",
        "Sharpe",
        "Sortino",
        "CVaR",
        "Max_DD",
        "duree_DD_jours",
        "duree_DD_mois",
        "Turnover",
        "L_portefeuille",
        "Nb_positions",
        "pct_mois_positifs",
        "wealth_final_100",
    ]
    print(summary_table[cols].to_string(index=False))
    print("\n=== COMPARISONS ===")
    print(
        pd.DataFrame(pairs)[
            [
                "Comparaison",
                "diff_moyenne_mensuelle",
                "diff_mediane",
                "diff_ecart_type",
                "pct_mois_A_surperforme",
                "IC95_bootstrap_lo",
                "IC95_bootstrap_hi",
                "p_value_ttest",
                "p_value_wilcoxon",
            ]
        ].to_string(index=False)
    )
    print("\n=== TRADEOFF ===")
    print(tradeoff.to_string(index=False))
    print("\n=== RANKING ===")
    print(pd.DataFrame(ranking).to_string(index=False))
    print("\nOUT:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
