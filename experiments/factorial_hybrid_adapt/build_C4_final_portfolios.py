"""Construit les portefeuilles finaux C4 (knee) depuis les artefacts NSGA existants.

Ne modifie pas le modèle, ne relance pas NSGA, ne fait pas de backtest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.stage2_allocation import (
    EVAL_END_MONTH,
    EVAL_START_MONTH,
)

BASE = Path(__file__).resolve().parent
PORT = BASE / "outputs" / "portfolios_stage2_C4_201807_202506"
OUT = BASE / "outputs" / "portfolios_C4_final"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    w = pd.read_parquet(PORT / "stage2_weights_C4.parquet")
    o = pd.read_parquet(PORT / "stage2_objectives_C4.parquet")

    hold = w[w["portfolio"] == "P_selected"].copy()
    hold["mois"] = hold["mois"].astype(str)
    hold = hold[(hold["mois"] >= EVAL_START_MONTH) & (hold["mois"] <= EVAL_END_MONTH)].copy()
    hold["weight"] = pd.to_numeric(hold["weight"], errors="coerce")
    hold["recommendation"] = hold["recommendation"].astype(str).str.upper()
    # residual due to weight_report_floor in NSGA export → renormalize to exact 100%
    hold["weight_raw"] = hold["weight"]
    hold["weight"] = hold.groupby("mois")["weight"].transform(
        lambda s: s / s.sum() if float(s.sum()) > 0 else s
    )
    # alpha = score préférencé as-of t (objectif NSGA #1)
    hold["alpha"] = pd.to_numeric(hold["score_opt"], errors="coerce")
    hold["liquidity_L"] = pd.to_numeric(hold["L_sigmoid"], errors="coerce")
    hold["vmq"] = pd.to_numeric(hold["liquidity_vmq_20j"], errors="coerce")

    obj = o[o["portfolio"] == "P_selected"].copy()
    obj["mois"] = obj["mois"].astype(str)
    obj = obj[(obj["mois"] >= EVAL_START_MONTH) & (obj["mois"] <= EVAL_END_MONTH)]
    obj_cols = obj[
        [
            "mois",
            "date_rebalance",
            "expected_alpha_opt",
            "expected_alpha_raw",
            "cvar_95",
            "liquidity_L_weighted",
            "turnover_vs_prev",
            "n_positions",
            "sum_weights",
            "selection_rule",
            "n_pareto",
            "status",
            "n_candidates",
        ]
    ].rename(
        columns={
            "expected_alpha_opt": "portfolio_alpha",
            "cvar_95": "portfolio_cvar",
            "liquidity_L_weighted": "portfolio_liquidity",
            "turnover_vs_prev": "turnover",
        }
    )

    final = hold.merge(obj_cols, on="mois", how="left", suffixes=("", "_obj"))
    if "date_rebalance" not in final.columns and "date_rebalance_obj" in final.columns:
        final["date_rebalance"] = final["date_rebalance_obj"]
    # pas de CVaR titre dans les artefacts → broadcast du CVaR portefeuille
    final["cvar"] = final["portfolio_cvar"]
    final["cell"] = "C4"
    final["month"] = final["mois"]
    final["selection_method"] = "knee_utopia_distance"
    final["model"] = "Hybrid + régime"

    detail = (
        final[
            [
                "month",
                "cell",
                "ticker",
                "recommendation",
                "weight",
                "alpha",
                "cvar",
                "liquidity_L",
                "vmq",
                "portfolio_liquidity",
                "portfolio_cvar",
                "portfolio_alpha",
                "turnover",
                "date_rebalance",
                "predicted_return",
                "conviction_score",
                "selection_method",
                "model",
                "n_pareto",
                "status",
            ]
        ]
        .sort_values(["month", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )

    monthly = (
        obj_cols.rename(columns={"mois": "month"})
        .assign(cell="C4", selection_method="knee_utopia_distance")
        [
            [
                "month",
                "cell",
                "portfolio_alpha",
                "portfolio_cvar",
                "portfolio_liquidity",
                "turnover",
                "n_positions",
                "date_rebalance",
                "selection_method",
                "n_pareto",
                "status",
                "sum_weights",
                "n_candidates",
            ]
        ]
        .sort_values("month")
        .reset_index(drop=True)
    )

    checks = []
    for mois, g in detail.groupby("month"):
        s = float(g["weight"].sum())
        checks.append(
            {
                "month": mois,
                "sum_weights": s,
                "sum_ok": abs(s - 1.0) < 1e-6,
                "n_SELL": int((g["recommendation"] == "SELL").sum()),
                "n_neg_weights": int((g["weight"] < -1e-12).sum()),
                "n_positions": int(len(g)),
                "w_min": float(g["weight"].min()),
                "w_max": float(g["weight"].max()),
            }
        )
    chk = pd.DataFrame(checks)

    n_months_expected = len(pd.period_range(EVAL_START_MONTH, EVAL_END_MONTH, freq="M"))
    assert len(monthly) == n_months_expected, (
        f"n_months={len(monthly)} attendu={n_months_expected} "
        f"({EVAL_START_MONTH}→{EVAL_END_MONTH})"
    )
    assert chk["sum_ok"].all()
    assert (chk["n_SELL"] == 0).all()
    assert (chk["n_neg_weights"] == 0).all()
    assert detail["recommendation"].ne("SELL").all()
    assert (detail["weight"] >= -1e-12).all()

    asof_note = {
        "lookahead": False,
        "data_as_of": "decision month date_rebalance / Stage1 panel at t",
        "alpha_definition": (
            "score_opt = predicted_return * preference_multiplier "
            "(BUY=1, NEUTRAL=0.5) as-of t"
        ),
        "portfolio_alpha": "dot(w, score_opt) from NSGA at t",
        "portfolio_cvar": "historical daily returns with date <= as_of t",
        "liquidity_L": "sigmoid(VMQ_20j) as-of t",
        "vmq": "VMQ_20j as-of t",
        "selection": (
            "knee / utopia distance; identical to P_equilibre; "
            "no realized next-month return"
        ),
        "source_artefacts": str(PORT),
        "model_unchanged": True,
        "backtest_not_run": True,
    }

    detail.to_parquet(OUT / "C4_final_holdings_titre_mois.parquet", index=False)
    detail.to_csv(OUT / "C4_final_holdings_titre_mois.csv", index=False)
    monthly.to_csv(OUT / "C4_final_monthly_summary.csv", index=False)
    monthly.to_parquet(OUT / "C4_final_monthly_summary.parquet", index=False)
    chk.to_csv(OUT / "C4_final_controls.csv", index=False)
    (OUT / "C4_final_manifest.json").write_text(
        json.dumps(
            {
                "cell": "C4",
                "configuration": "Hybrid + régime",
                "window": f"{EVAL_START_MONTH} → {EVAL_END_MONTH}",
                "n_months": int(len(monthly)),
                "n_holding_rows": int(len(detail)),
                "selection_method": "knee_utopia_distance",
                "controls": {
                    "all_sum_weights_100pct": bool(chk["sum_ok"].all()),
                    "n_SELL_total": int(chk["n_SELL"].sum()),
                    "n_neg_weights_total": int(chk["n_neg_weights"].sum()),
                    "w_min_global": float(detail["weight"].min()),
                    "w_max_global": float(detail["weight"].max()),
                    "n_positions_mean": float(monthly["n_positions"].mean()),
                    "n_positions_min": int(monthly["n_positions"].min()),
                    "n_positions_max": int(monthly["n_positions"].max()),
                    "renormalization": (
                        "Poids NSGA exportés avec floor 1e-4 ; renormalisation "
                        "mensuelle pour somme exacte = 1 (résidu max < 0.04%). "
                        "Composition inchangée."
                    ),
                },
                "as_of": asof_note,
                "files": [p.name for p in sorted(OUT.glob("C4_final_*"))],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=== CONTROLES ===")
    print(
        "mois=",
        len(monthly),
        "| sum_w OK=",
        bool(chk["sum_ok"].all()),
        "| SELL=",
        int(chk["n_SELL"].sum()),
        "| w<0=",
        int(chk["n_neg_weights"].sum()),
    )
    print(
        "n_pos: mean=",
        round(float(monthly["n_positions"].mean()), 2),
        "min=",
        int(monthly["n_positions"].min()),
        "max=",
        int(monthly["n_positions"].max()),
    )
    print(
        "weight: min=",
        round(float(detail["weight"].min()), 6),
        "p50=",
        round(float(detail["weight"].median()), 4),
        "mean=",
        round(float(detail["weight"].mean()), 4),
        "max=",
        round(float(detail["weight"].max()), 4),
    )

    for m in (EVAL_START_MONTH, "2018-08", "2018-09"):
        g = detail[detail["month"] == m].copy()
        print()
        print("=" * 90)
        print(
            "PORTEFEUILLE",
            m,
            "| n=",
            len(g),
            "| sum_w=",
            round(float(g["weight"].sum()), 6),
            "| alpha=",
            round(float(g["portfolio_alpha"].iloc[0]), 4),
            "| cvar=",
            round(float(g["portfolio_cvar"].iloc[0]), 4),
            "| L=",
            round(float(g["portfolio_liquidity"].iloc[0]), 4),
            "| TO=",
            round(float(g["turnover"].iloc[0]), 4),
        )
        print("=" * 90)
        show = g[["ticker", "recommendation", "weight", "alpha", "liquidity_L", "vmq"]].copy()
        show["weight_pct"] = (show["weight"] * 100).round(2)
        print(show.to_string(index=False))

    print()
    print("=== DIST Nb TITRES (mensuel) ===")
    print(monthly["n_positions"].describe().to_string())
    print()
    print("=== DIST POIDS (toutes lignes holdings) ===")
    print(
        detail["weight"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .to_string()
    )
    print()
    print("OUT:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
