"""
C1–C4 : même protocole Stage 2, seule contrainte ajoutée w_i ≤ 10 %.

Figé AVANT backtest. Ne touche pas aux scores, labels, objectifs NSGA, Knee.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.assemble_final_experiment import (
    FINAL,
    PARETO_DIR,
    backtest_holdings,
    load_selected_weights,
    metrics_bundle,
    paired_stats,
)
from experiments.factorial_hybrid_adapt.paths import OUTPUT_DIR, REPORTS_DIR
from experiments.factorial_hybrid_adapt.recommendations import CELL_META
from experiments.factorial_hybrid_adapt.stage2_allocation import (
    Stage2Config,
    run_stage2_all_cells,
    save_stage2_artifacts,
)
from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel

BT_START = "2018-07"
BT_END = "2025-06"
W_MAX = 0.10
CELLS = (1, 2, 3, 4)
OUT = FINAL / "13_wmax_10pct"
PORT_DIR = OUTPUT_DIR / "portfolios_stage2_wmax10"
OLD_HOLD = FINAL / "04_final_portfolios"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROTOCOL = {
    "variant": "C1–C4 w_max=10%",
    "frozen_before_backtest": True,
    "frozen_at": datetime.now(timezone.utc).isoformat(),
    "eval_window": f"{BT_START} → {BT_END}",
    "changed_only": {
        "w_max": W_MAX,
        "previous_w_max": 0.40,
        "applies_to": "C1, C2, C3, C4 identically",
    },
    "unchanged": {
        "models_scores": True,
        "BUY_NEUTRAL_SELL": "stage2_inputs_14blocks.parquet (quantile τ)",
        "nsga_objectives": ["max Alpha", "min CVaR", "max L"],
        "selection": "knee / distance to utopia",
        "sell_mode": "exclude",
        "pref_BUY": 1.0,
        "pref_NEUTRAL": 0.5,
        "liquidity_mode": "pareto",
        "vmq_eligibility_filter": False,
        "pop_size": 36,
        "n_gen": 30,
        "transaction_cost": 0.003,
    },
}


def concentration(holdings: pd.DataFrame) -> dict:
    h = holdings.copy()
    h["month"] = h["mois"].astype(str) if "mois" in h.columns else h["month"].astype(str)
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce").fillna(0.0)
    rows = []
    for mois, g in h.groupby("month"):
        w = g["weight"].to_numpy(dtype=float)
        s = w.sum()
        if s > 0:
            w = w / s
        w = np.sort(w)[::-1]
        rows.append(
            {
                "month": mois,
                "n_positions": int((w > 0).sum()),
                "w_max": float(w[0]) if len(w) else np.nan,
                "top3": float(w[: min(3, len(w))].sum()) if len(w) else np.nan,
                "top5": float(w[: min(5, len(w))].sum()) if len(w) else np.nan,
                "hhi": float((w**2).sum()) if len(w) else np.nan,
            }
        )
    c = pd.DataFrame(rows)
    return {
        "n_positions_moy": round(float(c["n_positions"].mean()), 2),
        "w_max_moy": round(float(c["w_max"].mean()), 4),
        "w_max_max": round(float(c["w_max"].max()), 4),
        "top3_moy": round(float(c["top3"].mean()), 4),
        "top5_moy": round(float(c["top5"].mean()), 4),
        "hhi_moy": round(float(c["hhi"].mean()), 4),
        "pct_mois_wmax_gt_10pct": round(float((c["w_max"] > W_MAX + 1e-6).mean()), 4),
        "_monthly": c,
    }


def _load_daily(path: Path) -> pd.Series:
    d = pd.read_csv(path, parse_dates=["date"])
    return d.set_index("date")["ret_net"].astype(float).sort_index()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol_wmax10_frozen.json").write_text(
        json.dumps(PROTOCOL, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Contrainte figée : w_i ≤ {W_MAX:.0%} pour C1–C4 (avant backtest)")

    src = REPORTS_DIR / "stage2_inputs_14blocks.parquet"
    stage2 = pd.read_parquet(src)
    cfg = Stage2Config(
        sell_mode="exclude",
        pref_BUY=1.0,
        pref_NEUTRAL=0.5,
        pref_SELL=0.0,
        w_min=0.0,
        w_max=W_MAX,
        liquidity_mode="pareto",
        apply_vmq_filter=False,
        selection_rule="knee",
        pop_size=36,
        n_gen=30,
        eval_start_month=BT_START,
        eval_end_month=BT_END,
    )
    results = run_stage2_all_cells(stage2, cell_ids=CELLS, cfg=cfg)
    save_stage2_artifacts(results, PORT_DIR)

    cours, indices, _ = load_market_panels()
    prices = _price_panel(cours)
    masi_all = _masi_returns(indices)

    metric_rows = []
    conc_rows = []
    monthly_map: dict[str, pd.Series] = {}
    old_map: dict[str, pd.Series] = {}

    for cell in CELLS:
        w = load_selected_weights(PORT_DIR, cell, "P_selected")
        w.to_parquet(OUT / f"C{cell}_holdings_wmax10.parquet", index=False)
        conc = concentration(w)
        conc_m = conc.pop("_monthly")
        conc_m.to_csv(OUT / f"C{cell}_concentration_monthly.csv", index=False)
        mon, day, _ = backtest_holdings(w, prices, start=BT_START, end=BT_END)
        mon.to_csv(OUT / f"C{cell}_monthly.csv", index=False)
        pd.DataFrame(
            {
                "date": day.index,
                "ret_net": day.values,
                "wealth100": 100 * (1 + day).cumprod().values,
            }
        ).to_csv(OUT / f"C{cell}_daily.csv", index=False)
        name = f"{CELL_META[cell]['label']} w≤10%"
        m = metrics_bundle(day, mon, name, masi_all)
        m.update({k: conc[k] for k in conc})
        metric_rows.append(m)
        monthly_map[f"C{cell}"] = mon.set_index("month")["net_return"].astype(float)
        conc_rows.append({"cell": cell, "variant": "w_max=10%", **conc})

        old_w = pd.read_csv(OLD_HOLD / f"C{cell}_holdings_knee.csv")
        old_c = concentration(old_w)
        old_c.pop("_monthly")
        old_mon = pd.read_csv(FINAL / "05_backtest" / f"C{cell}_monthly.csv")
        old_day = _load_daily(FINAL / "05_backtest" / f"C{cell}_daily.csv")
        old_name = f"{CELL_META[cell]['label']} w≤40%"
        om = metrics_bundle(old_day, old_mon, old_name, masi_all)
        om.update({k: old_c[k] for k in old_c})
        metric_rows.append(om)
        old_map[f"C{cell}"] = old_mon.set_index("month")["net_return"].astype(float)
        conc_rows.append({"cell": cell, "variant": "w_max=40%", **old_c})

    table = pd.DataFrame(metric_rows)
    pref = [
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
        "w_max_moy",
        "w_max_max",
        "top3_moy",
        "top5_moy",
    ]
    cols = [c for c in pref if c in table.columns]
    table[cols].to_csv(OUT / "table_C1C4_wmax10_vs_wmax40.csv", index=False)
    pd.DataFrame(conc_rows).to_csv(OUT / "concentration_C1C4.csv", index=False)

    stats_rows = [
        paired_stats(monthly_map["C4"], monthly_map["C1"], "C4 w10% - C1 w10%"),
        paired_stats(old_map["C4"], old_map["C1"], "C4 w40% - C1 w40%"),
        paired_stats(monthly_map["C4"], old_map["C4"], "C4 w10% - C4 w40%"),
        paired_stats(monthly_map["C1"], old_map["C1"], "C1 w10% - C1 w40%"),
        paired_stats(monthly_map["C2"], old_map["C2"], "C2 w10% - C2 w40%"),
        paired_stats(monthly_map["C3"], old_map["C3"], "C3 w10% - C3 w40%"),
        paired_stats(monthly_map["C4"], monthly_map["C2"], "C4 w10% - C2 w10%"),
        paired_stats(monthly_map["C4"], monthly_map["C3"], "C4 w10% - C3 w10%"),
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT / "tests_wmax10.csv", index=False)

    rel_new = float((monthly_map["C4"] - monthly_map["C1"]).mean())
    rel_old = float((old_map["C4"] - old_map["C1"]).mean())
    verdict = {
        "C4_minus_C1_mean_month_w40": round(rel_old, 6),
        "C4_minus_C1_mean_month_w10": round(rel_new, 6),
        "change_in_C4_vs_C1": round(rel_new - rel_old, 6),
        "relative_C4_vs_C1": (
            "improved"
            if rel_new > rel_old + 1e-12
            else ("deteriorated" if rel_new < rel_old - 1e-12 else "unchanged")
        ),
        "note": "positive change_in_C4_vs_C1 = C4 closer to / further ahead of C1",
    }
    (OUT / "C4_vs_C1_relative.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Métriques w≤10% vs w≤40% ===")
    print(table[cols].to_string(index=False))
    print("\n=== Tests ===")
    print(stats_df.to_string(index=False))
    print("\n=== C4 vs C1 ===")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"\nSortie : {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
