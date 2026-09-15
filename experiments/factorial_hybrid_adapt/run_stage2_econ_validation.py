"""
CLI — Validation économique étage 2 : SELL exclu vs pénalisé (C1–C4).

Prérequis :
  outputs/portfolios_stage2_exclude/   (run --sell-mode exclude --selection knee)
  outputs/portfolios_stage2_penalize/  (run --sell-mode penalize --selection knee)

Usage :
  py -3 experiments/factorial_hybrid_adapt/run_stage2_econ_validation.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.paths import OUTPUT_DIR, REPORTS_DIR, ensure_output_dirs
from experiments.factorial_hybrid_adapt.stage2_econ_eval import (
    VARIANTS,
    build_holdings_enriched,
    build_monthly_composition,
    lookahead_checks,
    load_variant_weights,
    qualitative_synthesis,
    reco_usage_summary,
    simulate_variant,
    summarize_row,
    write_figures,
)
from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _assert_ready(root: Path) -> None:
    for mode in VARIANTS:
        d = root / f"portfolios_stage2_{mode}"
        if not (d / "stage2_weights_all.parquet").is_file():
            raise FileNotFoundError(
                f"Manquant : {d / 'stage2_weights_all.parquet'}\n"
                f"Lancer : py -3 experiments/factorial_hybrid_adapt/run_stage2_nsga.py "
                f"--sell-mode {mode} --selection knee"
            )


def main() -> int:
    ensure_output_dirs()
    root = OUTPUT_DIR
    _assert_ready(root)

    out = REPORTS_DIR / "stage2_econ"
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    cours, indices, technical = load_market_panels()
    prices = _price_panel(cours)
    masi = _masi_returns(indices)

    holdings_all = []
    composition_all = []
    monthly_all = []
    contrib_all = []
    table_rows = []
    daily_map: dict[tuple[str, int], pd.Series] = {}
    meta_checks = []

    for sell_mode in VARIANTS:
        logger.info("=== Variante SELL %s ===", sell_mode)
        w, o, info = load_variant_weights(root, sell_mode)
        meta = info["meta"]
        summary = info["summary"]
        n_fb = int((meta["status"] != "ok").sum()) if "status" in meta.columns else -1
        n_ok = int((meta["status"] == "ok").sum()) if "status" in meta.columns else -1
        meta_checks.append(
            {
                "sell_mode": sell_mode,
                "dir": info["dir"],
                "n_ok": n_ok,
                "n_fallback": n_fb,
                "n_months_meta": int(meta["mois"].nunique()) if len(meta) else 0,
                "config": summary.get("config", {}),
                "cells": summary.get("cells", {}),
            }
        )
        if n_fb != 0:
            logger.warning("Fallbacks détectés pour %s : %s", sell_mode, n_fb)

        holdings = build_holdings_enriched(w, o, sell_mode=sell_mode)
        comp = build_monthly_composition(holdings)
        holdings_all.append(holdings)
        composition_all.append(comp)

        for cell in (1, 2, 3, 4):
            logger.info("Simulation OOS C%s | %s", cell, sell_mode)
            sim = simulate_variant(
                holdings, prices, technical, masi, sell_mode=sell_mode, cell_id=cell
            )
            daily_map[(sell_mode, cell)] = sim["daily_returns"]
            monthly_all.append(sim["monthly"])
            contrib_all.append(sim["contributions"])
            table_rows.append(summarize_row(sell_mode, cell, sim))

    holdings_df = pd.concat(holdings_all, ignore_index=True)
    composition_df = pd.concat(composition_all, ignore_index=True)
    monthly_df = pd.concat(monthly_all, ignore_index=True)
    contrib_df = pd.concat(contrib_all, ignore_index=True)
    table = pd.DataFrame(table_rows)
    # Ordre demandé
    order = []
    for cell in (1, 2, 3, 4):
        for mode in ("exclude", "penalize"):
            order.append((cell, mode))
    table["_ord"] = table.apply(lambda r: order.index((int(r["cell"]), r["sell_mode"])), axis=1)
    table = table.sort_values("_ord").drop(columns="_ord")

    reco = reco_usage_summary(monthly_df)
    synthesis = qualitative_synthesis(table, reco)
    checks = lookahead_checks()

    # Alignement des mois entre variantes
    months_ex = set(monthly_df.loc[monthly_df["sell_mode"] == "exclude", "mois"].unique())
    months_pe = set(monthly_df.loc[monthly_df["sell_mode"] == "penalize", "mois"].unique())
    common_months = sorted(months_ex & months_pe)

    # Exports
    holdings_df.to_parquet(out / "holdings_titre_mois.parquet", index=False)
    composition_df.to_csv(out / "composition_mensuelle.csv", index=False)
    monthly_df.to_csv(out / "performance_mensuelle.csv", index=False)
    contrib_df.to_parquet(out / "contributions_titre_mois.parquet", index=False)
    table.to_csv(out / "table_comparaison_principale.csv", index=False)
    reco.to_csv(out / "analyse_recommandations_portefeuille.csv", index=False)

    # Daily wealth series
    wealth_frames = []
    for (mode, cell), s in daily_map.items():
        if s is None or s.empty:
            continue
        wealth_frames.append(
            pd.DataFrame(
                {
                    "date": s.index,
                    "ret": s.values,
                    "wealth": (1 + s).cumprod().values,
                    "sell_mode": mode,
                    "cell": cell,
                }
            )
        )
    if wealth_frames:
        pd.concat(wealth_frames, ignore_index=True).to_parquet(
            out / "daily_returns_wealth.parquet", index=False
        )

    fig_paths = write_figures(daily_map, monthly_df, fig_dir)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_months_common": len(common_months),
        "months_first": common_months[0] if common_months else None,
        "months_last": common_months[-1] if common_months else None,
        "months_exclude_only": sorted(months_ex - months_pe),
        "months_penalize_only": sorted(months_pe - months_ex),
        "portfolio_key": "P_selected",
        "transaction_cost": float(
            __import__(
                "bvc_recommender.benchmarking.config", fromlist=["TRANSACTION_COST"]
            ).TRANSACTION_COST
        ),
        "variant_meta": meta_checks,
        "lookahead_checks": checks,
        "nsga_params_note": (
            "Les deux variantes doivent partager selection_rule=knee, pop=36, gen=30, "
            "prefs BUY/NEU/SELL=1/0.5/0, w_min=0, w_max=0.10 ; seule sell_mode change "
            "(exclude vs penalize avec w_max_SELL=0.02)."
        ),
        "main_table": table.to_dict(orient="records"),
        "synthesis": synthesis,
        "figures": [str(p) for p in fig_paths],
    }
    (out / "stage2_econ_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Console
    print("\n=== TABLE COMPARATIVE PRINCIPALE (84 mois cibles) ===")
    show = table[
        [
            "Configuration",
            "Sell mode",
            "Return annuel",
            "Volatilité",
            "Sharpe",
            "Sortino",
            "CVaR",
            "Max DD",
            "Turnover",
            "Liquidité",
            "Nb titres",
        ]
    ]
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(show.to_string(index=False))

    print("\n=== Fallbacks / mois ===")
    for m in meta_checks:
        print(
            f"  {m['sell_mode']}: ok={m['n_ok']} fallback={m['n_fallback']} "
            f"mois_meta={m['n_months_meta']} | selection={m['config'].get('selection_rule')} "
            f"sell_mode_cfg={m['config'].get('sell_mode')}"
        )
    print(f"\nMois communs simulés : {len(common_months)} ({common_months[0]} → {common_months[-1]})")
    print("Look-ahead :")
    for c in checks:
        print(f"  [{c['status']}] {c['check']}")

    print("\n=== Usage recommandations (capital moyen) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(
            reco[
                [
                    "Configuration",
                    "Sell mode",
                    "pct_capital_BUY",
                    "pct_capital_NEUTRAL",
                    "pct_capital_SELL",
                    "contrib_BUY_mean",
                    "contrib_NEUTRAL_mean",
                    "contrib_SELL_mean",
                ]
            ].to_string(index=False)
        )

    print(f"\nSorties : {out}")
    print(f"Figures : {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
