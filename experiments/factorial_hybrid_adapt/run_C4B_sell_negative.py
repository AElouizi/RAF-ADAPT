"""
C4-B : même scores Hybrid+régime, même NSGA-III 3 obj., même Knee.
Seule modification : règle SELL.

Règle figée AVANT backtest
--------------------------
Score = excess_return (= predicted_return sous benchmark zero_alpha).
  SELL     si score < 0          (règle absolue, plus de bande −τ)
  BUY      si score > +τ         (τ existant, expanding_past_quantile 0.33)
  NEUTRAL  sinon                 (0 ≤ score ≤ τ)

C1, C4-Knee, modèles et hyperparamètres NSGA inchangés.
Fenêtre : 2018-07 → 2025-06.
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
    backtest_holdings,
    load_selected_weights,
    metrics_bundle,
    paired_stats,
)
from experiments.factorial_hybrid_adapt.paths import OUTPUT_DIR, REPORTS_DIR
from experiments.factorial_hybrid_adapt.stage2_allocation import (
    Stage2Config,
    run_stage2_all_cells,
    save_stage2_artifacts,
)
from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel

BT_START = "2018-07"
BT_END = "2025-06"
CELL = 4
OUT = FINAL / "12_C4B_sell_negative"
PORT_DIR = OUTPUT_DIR / "portfolios_C4B_sell_negative"
INPUTS_C4B = REPORTS_DIR / "stage2_inputs_C4B_sell_negative.parquet"
RECOS_C4B = REPORTS_DIR / "stage1_recommendations_C4B_sell_negative.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


PROTOCOL = {
    "variant": "C4-B",
    "frozen_before_backtest": True,
    "frozen_at": datetime.now(timezone.utc).isoformat(),
    "eval_window": f"{BT_START} → {BT_END}",
    "unchanged": {
        "C1": True,
        "C4_models_hyperparams": True,
        "C4_scores": "same Hybrid+régime predicted_return / excess_return / tau",
        "nsga": "NSGA-III 3 obj. max Alpha, min CVaR, max L ; pop=36 gen=30 ; w_max=0.10",
        "selection": "knee / distance to utopia",
        "sell_mode_stage2": "exclude",
        "pref_BUY": 1.0,
        "pref_NEUTRAL": 0.5,
        "liquidity_mode": "pareto",
        "vmq_eligibility_filter": False,
        "transaction_cost": 0.003,
    },
    "changed_only": {
        "sell_rule": (
            "SELL iff score < 0 ; "
            "BUY iff score > +tau (tau inchangé, expanding_past_quantile 0.33) ; "
            "else NEUTRAL"
        ),
        "replaces": "SELL iff score < -tau (quantile τ=0.33 on |pred|)",
        "score": "excess_return (= predicted_return, benchmark zero_alpha)",
    },
}


def recode_sell_negative(df: pd.DataFrame) -> pd.DataFrame:
    """Règle C4-B. τ conservé uniquement pour BUY vs NEUTRAL si score ≥ 0."""
    out = df.copy()
    score = pd.to_numeric(out["excess_return"], errors="coerce")
    tau = pd.to_numeric(out["tau"], errors="coerce")
    rec = np.where(
        score < 0,
        "SELL",
        np.where(score > tau, "BUY", "NEUTRAL"),
    )
    out["recommendation"] = rec
    out["sell_rule"] = "negative_score"
    out["tau_method_buy_neutral"] = out.get("tau_method", "expanding_past_quantile")
    return out


def universe_stats(rec: pd.DataFrame, label: str, start: str, end: str) -> dict:
    d = rec.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d = d[(d["month"] >= start) & (d["month"] <= end)]
    d["recommendation"] = d["recommendation"].astype(str).str.upper()
    g = d.groupby("month")
    n_sell = g["recommendation"].apply(lambda s: int((s == "SELL").sum()))
    n_univ = g["recommendation"].apply(lambda s: int((s != "SELL").sum()))
    n_all = g.size()
    return {
        "Strategie": label,
        "n_months": int(d["month"].nunique()),
        "n_SELL_moyen": round(float(n_sell.mean()), 2),
        "n_univers_BUY_NEUTRAL_moyen": round(float(n_univ.mean()), 2),
        "n_titres_panel_moyen": round(float(n_all.mean()), 2),
        "pct_SELL": round(float((d["recommendation"] == "SELL").mean()), 4),
        "pct_BUY": round(float((d["recommendation"] == "BUY").mean()), 4),
        "pct_NEUTRAL": round(float((d["recommendation"] == "NEUTRAL").mean()), 4),
    }


def _load_daily(path: Path) -> pd.Series:
    d = pd.read_csv(path, parse_dates=["date"])
    return d.set_index("date")["ret_net"].astype(float).sort_index()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol_C4B_frozen.json").write_text(
        json.dumps(PROTOCOL, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Règle C4-B figée (avant backtest) : SELL ⇔ score < 0")

    src_s2 = REPORTS_DIR / "stage2_inputs_14blocks.parquet"
    src_rec = REPORTS_DIR / "stage1_recommendations_14blocks.parquet"
    if not src_s2.is_file():
        raise FileNotFoundError(src_s2)

    s2 = pd.read_parquet(src_s2)
    rec_all = pd.read_parquet(src_rec)
    c4 = s2[s2["cell"] == CELL].copy()
    c4b = recode_sell_negative(c4)
    c4b.to_parquet(INPUTS_C4B, index=False)

    rec_c4 = rec_all[rec_all["cell"] == CELL].copy()
    rec_c4b = recode_sell_negative(rec_c4)
    rec_c4b.to_parquet(RECOS_C4B, index=False)

    rec_c1 = rec_all[rec_all["cell"] == 1]
    univ = pd.DataFrame(
        [
            universe_stats(rec_c1, "C1", BT_START, BT_END),
            universe_stats(rec_c4, "C4-Knee (τ quantile)", BT_START, BT_END),
            universe_stats(rec_c4b, "C4-B (SELL si score<0)", BT_START, BT_END),
        ]
    )
    univ.to_csv(OUT / "universe_SELL_stats.csv", index=False)
    print("\n=== Univers / SELL (figé, avant NSGA) ===")
    print(univ.to_string(index=False))

    cfg = Stage2Config(
        sell_mode="exclude",
        pref_BUY=1.0,
        pref_NEUTRAL=0.5,
        pref_SELL=0.0,
        w_min=0.0,
        w_max=0.10,
        liquidity_mode="pareto",
        apply_vmq_filter=False,
        selection_rule="knee",
        pop_size=36,
        n_gen=30,
        eval_start_month=BT_START,
        eval_end_month=BT_END,
    )
    print("\n=== NSGA-III C4-B (cellule 4 uniquement) ===")
    results = run_stage2_all_cells(c4b, cell_ids=(CELL,), cfg=cfg)
    save_stage2_artifacts(results, PORT_DIR)

    holdings = load_selected_weights(PORT_DIR, CELL, "P_selected")
    holdings.to_parquet(OUT / "C4B_holdings_knee.parquet", index=False)
    holdings.to_csv(OUT / "C4B_holdings_knee.csv", index=False)

    cours, indices, _ = load_market_panels()
    prices = _price_panel(cours)
    masi_all = _masi_returns(indices)
    monthly, daily, _wealth = backtest_holdings(
        holdings, prices, start=BT_START, end=BT_END
    )
    monthly.to_csv(OUT / "C4B_monthly.csv", index=False)
    daily.rename("net_return").to_csv(OUT / "C4B_daily.csv")

    c1m = pd.read_csv(FINAL / "05_backtest" / "C1_monthly.csv")
    c4m = pd.read_csv(FINAL / "05_backtest" / "C4_monthly.csv")
    c1m["month"] = c1m["month"].astype(str)
    c4m["month"] = c4m["month"].astype(str)

    table = pd.DataFrame(
        [
            metrics_bundle(
                _load_daily(FINAL / "05_backtest" / "C1_daily.csv"),
                c1m,
                "C1 Ridge (Knee)",
                masi_all,
            ),
            metrics_bundle(
                _load_daily(FINAL / "05_backtest" / "C4_daily.csv"),
                c4m,
                "C4 Knee",
                masi_all,
            ),
            metrics_bundle(daily, monthly, "C4-B (SELL si score<0, Knee)", masi_all),
        ]
    )
    table.to_csv(OUT / "table_C1_C4knee_C4B.csv", index=False)

    def _idx(df: pd.DataFrame) -> pd.Series:
        return df.set_index("month")["net_return"].astype(float)

    s_b = _idx(monthly)
    stats_df = pd.DataFrame(
        [
            paired_stats(s_b, _idx(c4m), "C4-B - C4-Knee"),
            paired_stats(s_b, _idx(c1m), "C4-B - C1"),
            paired_stats(_idx(c4m), _idx(c1m), "C4-Knee - C1"),
        ]
    )
    stats_df.to_csv(OUT / "tests_C4B_vs_C4knee_C1.csv", index=False)

    print("\n=== Métriques 2018-07 → 2025-06 ===")
    cols = [
        c
        for c in (
            "Strategie",
            "n_months",
            "Rendement",
            "Volatilite",
            "Sharpe",
            "CVaR",
            "Max_DD",
            "Turnover",
            "Liquidite",
            "n_positions_moy",
        )
        if c in table.columns
    ]
    print(table[cols].to_string(index=False))
    print("\n=== Tests appariés ===")
    print(stats_df.to_string(index=False))
    print(f"\nSortie : {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
