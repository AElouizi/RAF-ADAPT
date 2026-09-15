"""
C4-new : même front Pareto NSGA-III (3 obj.), autre règle de sélection.

Ne relance PAS NSGA. Ne retouche PAS Knee C4 ni C1.
Sélection = max Alpha parmi les solutions du front qui respectent les
contraintes déjà figées dans le protocole :
  - CVaR ≤ 0.9 × CVaR_MASI (as-of t, même formule que NSGA)
  - L pondéré ≥ sigmoid(VMQ = 500k MAD)

Backtest : 2018-07 → 2025-06, mêmes frais / anti look-ahead que C4-Knee.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bvc_recommender.models.portfolio_optimizer import CVAR_MASI_RATIO, compute_cvar
from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.assemble_final_experiment import (
    FINAL,
    PARETO_DIR,
    backtest_holdings,
    metrics_bundle,
    paired_stats,
)
from experiments.factorial_hybrid_adapt.stage2_allocation import (
    DEFAULT_L_MIN,
    select_max_alpha_constrained_index,
)
from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel

BT_START = "2018-07"
BT_END = "2025-06"
OUT = FINAL / "11_C4_maxalpha_constrained"
CELL = 4


def _cvar_limit_asof(indices: pd.DataFrame, as_of: pd.Timestamp) -> tuple[float, float]:
    idx = indices.copy()
    dc = "date_index" if "date_index" in idx.columns else "date"
    idx[dc] = pd.to_datetime(idx[dc], errors="coerce")
    idx = idx[idx[dc] <= pd.Timestamp(as_of)]
    masi = _masi_returns(idx)
    cvar_masi = float(compute_cvar(masi.values))
    return cvar_masi, float(cvar_masi * CVAR_MASI_RATIO)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    obj = pd.read_parquet(PARETO_DIR / "stage2_pareto_objectives.parquet")
    wgt = pd.read_parquet(PARETO_DIR / "stage2_pareto_weights.parquet")
    obj = obj[obj["cell"] == CELL].copy()
    wgt = wgt[wgt["cell"] == CELL].copy()
    obj["mois"] = obj["mois"].astype(str)
    wgt["mois"] = wgt["mois"].astype(str)
    obj = obj[(obj["mois"] >= BT_START) & (obj["mois"] <= BT_END)]
    wgt = wgt[(wgt["mois"] >= BT_START) & (wgt["mois"] <= BT_END)]
    obj["date_rebalance"] = pd.to_datetime(obj["date_rebalance"], errors="coerce")

    cours, indices, _tech = load_market_panels()
    prices = _price_panel(cours)
    masi_all = _masi_returns(indices)

    sel_rows: list[dict] = []
    hold_parts: list[pd.DataFrame] = []
    for mois, g in obj.groupby("mois", sort=True):
        g = g.sort_values("sol_id").reset_index(drop=True)
        as_of = pd.Timestamp(g["date_rebalance"].iloc[0])
        cvar_masi, cvar_limit = _cvar_limit_asof(indices, as_of)
        idx, stage, n_feas = select_max_alpha_constrained_index(
            g["expected_alpha_opt"].to_numpy(),
            g["cvar_95"].to_numpy(),
            g["liquidity_L_weighted"].to_numpy(),
            cvar_limit=cvar_limit,
            l_min=DEFAULT_L_MIN,
        )
        row = g.iloc[idx]
        sol_id = int(row["sol_id"])
        sel_rows.append(
            {
                "month": mois,
                "date_rebalance": as_of.date().isoformat(),
                "sol_id": sol_id,
                "feasibility_stage": stage,
                "n_feasible": n_feas,
                "n_pareto": int(len(g)),
                "cvar_masi": round(cvar_masi, 6),
                "cvar_limit": round(cvar_limit, 6),
                "l_min": DEFAULT_L_MIN,
                "expected_alpha_opt": float(row["expected_alpha_opt"]),
                "cvar_95": float(row["cvar_95"]),
                "liquidity_L_weighted": float(row["liquidity_L_weighted"]),
                "n_positions": int(row["n_positions"]) if pd.notna(row["n_positions"]) else None,
                "same_as_knee": bool(row["is_knee"]),
                "same_as_unconstrained_max_alpha": bool(row["is_max_alpha"]),
                "cvar_ok": bool(float(row["cvar_95"]) <= cvar_limit + 1e-9),
                "L_ok": bool(float(row["liquidity_L_weighted"]) >= DEFAULT_L_MIN - 1e-12),
            }
        )
        hw = wgt[(wgt["mois"] == mois) & (wgt["sol_id"] == sol_id)].copy()
        if hw.empty:
            raise RuntimeError(f"Poids manquants C4 {mois} sol_id={sol_id}")
        hw["month"] = mois
        hw["portfolio"] = "P_selected_maxalpha_constrained"
        hw["liquidity_L_weighted"] = float(row["liquidity_L_weighted"])
        hold_parts.append(hw)

    selection = pd.DataFrame(sel_rows)
    holdings = pd.concat(hold_parts, ignore_index=True)
    holdings.to_parquet(OUT / "C4_holdings_maxalpha_constrained.parquet", index=False)
    holdings.to_csv(OUT / "C4_holdings_maxalpha_constrained.csv", index=False)
    selection.to_csv(OUT / "C4_selection_log.csv", index=False)

    monthly, daily, wealth = backtest_holdings(
        holdings, prices, start=BT_START, end=BT_END
    )
    monthly.to_csv(OUT / "C4_new_monthly.csv", index=False)
    daily.rename("net_return").to_csv(OUT / "C4_new_daily.csv")

    c1m = pd.read_csv(FINAL / "05_backtest" / "C1_monthly.csv")
    c4k = pd.read_csv(FINAL / "05_backtest" / "C4_monthly.csv")
    c1m["month"] = c1m["month"].astype(str)
    c4k["month"] = c4k["month"].astype(str)

    def _idx(df: pd.DataFrame) -> pd.Series:
        return df.set_index("month")["net_return"].astype(float)

    m_new = metrics_bundle(daily, monthly, "C4 new (max α | CVaR & L)", masi_all)
    # Rebuild C1 / C4-Knee metrics from stored monthly + daily if present
    c1_daily_p = FINAL / "05_backtest" / "C1_daily.csv"
    c4_daily_p = FINAL / "05_backtest" / "C4_daily.csv"

    def _load_daily(path: Path) -> pd.Series:
        d = pd.read_csv(path, parse_dates=["date"])
        return d.set_index("date")["ret_net"].astype(float).sort_index()

    m_c1 = metrics_bundle(_load_daily(c1_daily_p), c1m, "C1 Ridge (Knee)", masi_all)
    m_k = metrics_bundle(_load_daily(c4_daily_p), c4k, "C4 Knee", masi_all)

    table = pd.DataFrame([m_c1, m_k, m_new])
    table.to_csv(OUT / "table_C1_C4knee_C4new.csv", index=False)

    s_new = _idx(monthly)
    stats_df = pd.DataFrame(
        [
            paired_stats(s_new, _idx(c4k), "C4-new - C4-Knee"),
            paired_stats(s_new, _idx(c1m), "C4-new - C1"),
            paired_stats(_idx(c4k), _idx(c1m), "C4-Knee - C1"),
        ]
    )
    stats_df.to_csv(OUT / "tests_C4new_vs_C4knee_C1.csv", index=False)

    stage_counts = selection["feasibility_stage"].value_counts().to_dict()
    protocol = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nsga": "unchanged frozen Pareto C4 (3 obj Alpha/CVaR/L, pop=36 gen=30)",
        "selection_old": "knee / distance to utopia",
        "selection_new": "max expected_alpha_opt on Pareto s.t. protocol constraints",
        "constraints": {
            "cvar": "CVaR_95 <= 0.9 * CVaR_MASI (as-of rebalance, same as NSGA)",
            "cvar_masi_ratio": CVAR_MASI_RATIO,
            "liquidity": "portfolio L=sigmoid(VMQ) >= sigmoid(500000 MAD)",
            "l_min": DEFAULT_L_MIN,
            "w_max": 0.10,
            "not_recalibrated_on_backtest": True,
        },
        "fallback_hierarchy": [
            "cvar_and_L",
            "cvar_only",
            "cvar_relaxed_and_L",
            "cvar_relaxed_only",
            "unconstrained",
        ],
        "eval_window": f"{BT_START} → {BT_END}",
        "n_months": int(len(selection)),
        "feasibility_stage_counts": stage_counts,
        "pct_same_as_knee": float(selection["same_as_knee"].mean()),
        "pct_same_as_unconstrained_max_alpha": float(
            selection["same_as_unconstrained_max_alpha"].mean()
        ),
        "pct_L_ok": float(selection["L_ok"].mean()),
        "pct_cvar_ok": float(selection["cvar_ok"].mean()),
        "transaction_cost": 0.003,
        "pareto_source": str(PARETO_DIR),
    }
    (OUT / "protocol_selection.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== C4 new : max Alpha sous contraintes protocole ===")
    print(f"mois={len(selection)} | stages={stage_counts}")
    print(
        f"identique Knee={protocol['pct_same_as_knee']:.1%} | "
        f"identique max-α libre={protocol['pct_same_as_unconstrained_max_alpha']:.1%} | "
        f"L_ok={protocol['pct_L_ok']:.1%}"
    )
    print()
    cols = [
        c
        for c in (
            "Strategie",
            "n_months",
            "Rendement",
            "Volatilite",
            "Sharpe",
            "Sortino",
            "CVaR",
            "Max_DD",
            "Turnover",
            "Liquidite",
            "n_positions_moy",
        )
        if c in table.columns
    ]
    print(table[cols].to_string(index=False))
    print()
    print(stats_df.to_string(index=False))
    print(f"\nSortie : {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
