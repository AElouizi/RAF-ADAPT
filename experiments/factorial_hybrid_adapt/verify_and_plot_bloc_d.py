# -*- coding: utf-8 -*-
"""Verify Bloc D table KPIs + plot correct C1-C5 wealth curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
REP = BASE / "outputs" / "reports"
FIG = REP / "figures"
LIV = BASE.parents[1] / "livrables" / "figures_final"
PARTS = BASE / "outputs" / "bloc_d_wf_14blocks"
FIG.mkdir(parents=True, exist_ok=True)
LIV.mkdir(parents=True, exist_ok=True)

LABELS = {
    "c1_ridge": "C1 — Ridge",
    "c2_rf": "C2 — Random Forest",
    "c3_lightgbm": "C3 — LightGBM",
    "c4_hybrid_tri": "C4 — Ridge + RF + LightGBM",
    "c5_hybrid_tri_regime": "C5 — Ridge + RF + LightGBM + régime",
}
COLORS = {
    "c1_ridge": "#1f77b4",
    "c2_rf": "#b45309",
    "c3_lightgbm": "#e85d04",
    "c4_hybrid_tri": "#15803d",
    "c5_hybrid_tri_regime": "#7c3aed",
    "MASI": "#1d4ed8",
}

# ---- 1) Verify table ----
long_raw = pd.read_csv(REP / "bloc_d_results_long_14blocks.csv")
summ = pd.read_csv(REP / "bloc_d_summary_14blocks.csv")
# long format: fold_id, model_id, metrique, valeur
wide = long_raw.pivot_table(
    index=["fold_id", "model_id", "groupe"],
    columns="metrique",
    values="valeur",
    aggfunc="first",
).reset_index()
wide.columns.name = None
long = wide
needed = ["sharpe", "ann_return", "max_drawdown", "sortino", "turnover_mean"]
missing = [c for c in needed if c not in long.columns]
if missing:
    raise SystemExit(f"Missing metrics in long file: {missing}; have={list(long.columns)}")
recomputed = (
    long.groupby("model_id", as_index=False)
    .agg(
        n_blocks=("fold_id", "nunique"),
        sharpe=("sharpe", "mean"),
        ann_return=("ann_return", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        sortino=("sortino", "mean"),
        turnover_mean=("turnover_mean", "mean"),
    )
    .sort_values("sharpe", ascending=False)
)

print("=== VERIFICATION TABLEAU ===")
print(f"long rows={len(long)} models={sorted(long['model_id'].unique())}")
print(f"unique folds={sorted(long['fold_id'].unique())} n={long['fold_id'].nunique()}")
print()

# Screenshot expected (rounded)
expected = {
    "c1_ridge": dict(sharpe=1.334, ann=15.8, maxdd=-15.5, sortino=2.20, to=0.941),
    "c2_rf": dict(sharpe=1.321, ann=15.7, maxdd=-15.9, sortino=2.19, to=0.947),
    "c3_lightgbm": dict(sharpe=1.484, ann=17.4, maxdd=-15.3, sortino=2.47, to=0.989),
    "c4_hybrid_tri": dict(sharpe=1.400, ann=17.5, maxdd=-15.6, sortino=2.35, to=0.983),
    "c5_hybrid_tri_regime": dict(sharpe=1.367, ann=17.2, maxdd=-16.2, sortino=2.28, to=1.004),
}

ok_all = True
for mid, exp in expected.items():
    row = recomputed.loc[recomputed["model_id"] == mid].iloc[0]
    got = {
        "sharpe": round(float(row["sharpe"]), 3),
        "ann": round(100 * float(row["ann_return"]), 1),
        "maxdd": round(100 * float(row["max_drawdown"]), 1),
        "sortino": round(float(row["sortino"]), 2),
        "to": round(float(row["turnover_mean"]), 3),
    }
    match = got == exp
    ok_all = ok_all and match and int(row["n_blocks"]) == 14
    print(f"{mid}: n_blocks={int(row['n_blocks'])} got={got} expected={exp} MATCH={match}")

# summary file vs recomputed
merged = summ.merge(recomputed, on="model_id", suffixes=("_file", "_re"))
for col in ["sharpe", "ann_return", "max_drawdown", "sortino", "turnover_mean"]:
    diff = (merged[f"{col}_file"] - merged[f"{col}_re"]).abs().max()
    print(f"summary vs recompute max|diff| {col} = {diff:.2e}")

rep = json.loads((REP / "bloc_d_report_14blocks.json").read_text(encoding="utf-8"))
bench = rep.get("benchmarks") or rep.get("bench_summary") or {}
print("report top keys:", list(rep.keys()))
print("bench:", bench if bench else "looking nested...")
# find bench values
raw = json.dumps(rep)
for key in ("masi_sharpe_mean", "ew_masi20_sharpe_mean", "masi_ann_return_mean"):
    if key in rep:
        print(key, rep[key])
    elif "benchmarks" in rep and isinstance(rep["benchmarks"], dict) and key in rep["benchmarks"]:
        print(key, rep["benchmarks"][key])

print("TABLE_OK", ok_all)

# ---- 2) Build / load wealth ----
weights = list(PARTS.glob("fold_*_weights.parquet")) if PARTS.is_dir() else []
print(f"weight files found: {len(weights)} in {PARTS.exists()}")

wealth_path = FIG / "bloc_d_wealth_daily.parquet"
if PARTS.is_dir() and len(weights) >= 14:
    # rebuild via project helpers
    import sys

    sys.path.insert(0, str(BASE.parents[1]))
    from experiments.factorial_hybrid_adapt.allocation import load_market_panels
    from experiments.factorial_hybrid_adapt.bloc_b_models import MODEL_IDS
    from experiments.factorial_hybrid_adapt.bloc_d_multi_model import (
        simulate_benchmark_block,
        simulate_block_portfolio,
    )
    from experiments.factorial_hybrid_adapt.data_utils import load_ml_dataset
    from experiments.factorial_hybrid_adapt.stage2_bloc_b_loader import block_fold_ids_from_summary
    from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel
    from experiments.factorial_hybrid_adapt.walk_forward_folds import (
        available_months_from_dates,
        generate_rolling_folds,
    )

    block_ids = block_fold_ids_from_summary()
    months = available_months_from_dates(load_ml_dataset()["date_cours"])
    by_id = {f.fold_id: f for f in generate_rolling_folds(months)}
    blocks = [by_id[fid] for fid in block_ids if fid in by_id]
    print("blocks used:", [b.fold_id for b in blocks], "n=", len(blocks))

    cours, indices, technical = load_market_panels()
    prices = _price_panel(cours)
    masi = _masi_returns(indices)

    series: dict[str, pd.Series] = {}
    for model_id in MODEL_IDS:
        parts: list[pd.Series] = []
        for fold in blocks:
            wpath = PARTS / f"fold_{fold.fold_id:04d}_{model_id}_weights.parquet"
            if not wpath.is_file():
                print("MISSING", wpath.name)
                continue
            weights_df = pd.read_parquet(wpath)
            sim = simulate_block_portfolio(
                weights_df, prices, technical, masi,
                model_id=model_id, fold_id=fold.fold_id,
            )
            dr = sim.get("daily_returns")
            if dr is not None and not dr.empty:
                parts.append(dr)
        if not parts:
            continue
        daily = pd.concat(parts).sort_index()
        daily = daily[~daily.index.duplicated(keep="last")]
        series[LABELS[model_id]] = 100 * (1 + daily).cumprod()

    masi_parts: list[pd.Series] = []
    for fold in blocks:
        bsim = simulate_benchmark_block(fold, prices, cours, masi, kind="masi")
        dr = bsim.get("daily_returns")
        if dr is not None and not dr.empty:
            masi_parts.append(dr)
    if masi_parts:
        md = pd.concat(masi_parts).sort_index()
        md = md[~md.index.duplicated(keep="last")]
        # Aligner le MASI sur l'horizon des stratégies (dernier mois → max cours).
        if series:
            horizon = max(s.index.max() for s in series.values())
            last_m = md.index.max()
            extra = masi.loc[(masi.index > last_m) & (masi.index <= horizon)]
            if not extra.empty:
                md = pd.concat([md, extra]).sort_index()
                md = md[~md.index.duplicated(keep="last")]
        series["MASI"] = 100 * (1 + md).cumprod()

    wealth = pd.DataFrame(series).sort_index()
    wealth.to_parquet(wealth_path)
    wealth.reset_index(names="date").to_csv(FIG / "bloc_d_wealth_daily.csv", index=False)
    print("wealth saved", wealth_path, wealth.shape, wealth.index.min(), wealth.index.max())
    print("final levels:")
    print(wealth.iloc[-1].round(2).to_string())
else:
    print("NO WEIGHTS — cannot rebuild Bloc D wealth from parts")
    wealth = pd.DataFrame()

# ---- 3) Plot ----
if not wealth.empty:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    order = [LABELS[m] for m in LABELS] + ["MASI"]
    for lab in order:
        if lab not in wealth.columns:
            continue
        mid = next((k for k, v in LABELS.items() if v == lab), "MASI")
        color = COLORS.get(mid, COLORS["MASI"])
        lw = 2.6 if mid == "c3_lightgbm" else (2.0 if lab == "MASI" else 1.5)
        ax.plot(wealth.index, wealth[lab], label=lab, color=color, lw=lw)
    ax.axhline(100, color="#94a3b8", lw=0.9, ls=":", zorder=0)
    ax.set_title(
        "Évolution de la performance — Bloc D C1–C5 (NSGA P_équilibre) vs MASI\n"
        "Courbes chaînées post-NSGA · base 100 · aligné sur le tableau Sharpe (moy. 14 blocs)"
    )
    ax.set_ylabel("Richesse cumulée (base 100)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.28)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    last = wealth.iloc[-1]
    for lab in order:
        if lab in wealth.columns:
            mid = next((k for k, v in LABELS.items() if v == lab), "MASI")
            ax.annotate(
                f"{last[lab]:.0f}",
                xy=(wealth.index[-1], last[lab]),
                xytext=(6, 0),
                textcoords="offset points",
                color=COLORS.get(mid, "#1d4ed8"),
                fontsize=8,
                va="center",
            )
    fig.tight_layout()
    out = FIG / "evolution_performance_bloc_d_C1C5.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(LIV / out.name, dpi=200, bbox_inches="tight")
    # also overwrite the previously misleading "tous_portefeuilles" with a clear name note
    fig.savefig(FIG / "evolution_performance_tous_portefeuilles.png", dpi=200, bbox_inches="tight")
    fig.savefig(LIV / "evolution_performance_tous_portefeuilles.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("PLOT", out)
