"""
Cascade post-correction splits :
1) Recalcul features techniques (step2)
2) Recalcul features_indices (market context)
3) Stats masi_mom_3m / breadth_ma50
4) Réentraînement HMM (step4)
5) Diagnostic régime

Usage :
  py -3 diagnostics/recompute_after_split_fix.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    FEATURES_DIR,
    load_env_file,
)
from bvc_recommender.features.market_context import (  # noqa: E402
    MARKET_CONTEXT_COLUMNS,
    build_market_context,
)
from bvc_recommender.features.writer import save_features_local  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return {
        "n": int(len(s)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "p999": float(s.quantile(0.999)),
        "max": float(s.max()),
        "max_date": str(s.idxmax().date()) if isinstance(s.index, pd.DatetimeIndex) else str(s.idxmax()),
        "n_gt_10x_med": int((s > 10 * s.median()).sum()) if s.median() > 0 else 0,
    }


def _print_stats(name: str, st: dict) -> None:
    print(f"\n=== {name} ===")
    print(
        f"  n={st['n']}  med={st['median']:.6f}  mean={st['mean']:.6f}  std={st['std']:.6f}"
    )
    print(
        f"  p90={st['p90']:.6f}  p95={st['p95']:.6f}  p99={st['p99']:.6f}  "
        f"p99.9={st['p999']:.6f}"
    )
    print(
        f"  max={st['max']:.6f} ({st['max_date']})  n>10x_med={st['n_gt_10x_med']}"
    )


def rebuild_market_context() -> pd.DataFrame:
    cours = pd.read_parquet(DATA_PROCESSED_DIR / "market_data_cours_historique.parquet")
    indices = pd.read_parquet(DATA_PROCESSED_DIR / "market_data_indices_historique.parquet")
    ctx = build_market_context(cours, indices)
    # Filtrer 2015-2025 pour le diagnostic (le panel peut être plus large)
    ctx["date_cours"] = pd.to_datetime(ctx["date_cours"], errors="coerce")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    # Écrire le panel complet (step4 filtrera)
    save_features_local(ctx, "features_indices", FEATURES_DIR)
    print(
        f"features_indices recalculé : {len(ctx)} j | cols={list(MARKET_CONTEXT_COLUMNS)} "
        f"| note: herfindahl_volume non recalculé (retiré du pipeline, volumes incomplets)"
    )
    return ctx


def main() -> int:
    load_env_file()
    py = sys.executable

    # 1) Features techniques
    _run([py, "-m", "bvc_recommender.scripts.run_step2", "--use-processed"])

    # 2) Features indices (contexte marché) — force rebuild hors --use-features
    print("\n" + "=" * 72)
    print("2. Recalcul features_indices (market context)")
    print("=" * 72)
    ctx = rebuild_market_context()
    ctx = ctx.dropna(subset=["date_cours"]).copy()
    ctx = ctx[(ctx["date_cours"].dt.year >= 2015) & (ctx["date_cours"].dt.year <= 2025)]

    # 3) Stats HMM inputs (hors dispersion déjà validée)
    print("\n" + "=" * 72)
    print("3. Stats descriptives entrées HMM (hors dispersion déjà validée)")
    print("=" * 72)
    for col in ("masi_mom_3m", "breadth_ma50", "return_dispersion"):
        s = ctx.set_index("date_cours")[col]
        _print_stats(col, _stats(s))

    # Flags anomalies
    for col in ("masi_mom_3m", "breadth_ma50"):
        st = _stats(ctx.set_index("date_cours")[col])
        ratio = st["max"] / st["p99"] if st["p99"] else float("inf")
        print(
            f"\n  Check {col}: max/p99={ratio:.2f}  "
            f"{'OK' if ratio < 3 and st['n_gt_10x_med'] == 0 else 'ANOMALIES POSSIBLES'}"
        )

    # 4) HMM step4
    _run([py, "-m", "bvc_recommender.scripts.run_step4", "--use-processed"])

    # 5) Diagnostic régime
    _run([py, str(ROOT / "diagnostics" / "regime_variability_check.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
