"""
CLI Étape 2 WF — couverture des régimes sur les fenêtres test (sans entraînement).

Pour chaque pli chevauchant : répartition bull/neutral/bear sur les mois EOM
de la fenêtre test, puis agrégation globale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import FEATURES_DIR, load_env_file  # noqa: E402
from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR, ensure_output_dirs  # noqa: E402
from experiments.factorial_hybrid_adapt.walk_forward_folds import (  # noqa: E402
    folds_to_frame,
    generate_rolling_folds,
    available_months_from_dates,
)
from bvc_recommender.config import DATASET_DIR  # noqa: E402

load_env_file(ROOT / ".env")


def _regime_label(row: pd.Series) -> str:
    if int(row.get("is_bull") or 0) == 1:
        return "bull"
    if int(row.get("is_bear") or 0) == 1:
        return "bear"
    return "neutral"


def load_regime_eom() -> pd.DataFrame:
    """Un régime par mois calendaire (dernier jour dispo du mois dans features_indices)."""
    fi = pd.read_parquet(
        FEATURES_DIR / "features_indices.parquet",
        columns=["date_cours", "is_bull", "is_neutral", "is_bear"],
    )
    fi["date_cours"] = pd.to_datetime(fi["date_cours"], errors="coerce")
    fi = fi.dropna(subset=["date_cours"]).sort_values("date_cours")
    fi["mois"] = fi["date_cours"].dt.to_period("M")
    eom = fi.groupby("mois", as_index=False).tail(1).copy()
    eom["regime"] = eom.apply(_regime_label, axis=1)
    return eom[["mois", "date_cours", "is_bull", "is_neutral", "is_bear", "regime"]]


def fold_test_regime_stats(fold_row: pd.Series, regime_eom: pd.DataFrame) -> dict:
    start = pd.Period(fold_row["test_start_month"], freq="M")
    end = pd.Period(fold_row["test_end_month"], freq="M")
    months = list(pd.period_range(start, end, freq="M"))
    sub = regime_eom[regime_eom["mois"].isin(months)].copy()
    n = len(sub)
    if n == 0:
        return {
            "fold_id": int(fold_row["fold_id"]),
            "test_start_month": fold_row["test_start_month"],
            "test_end_month": fold_row["test_end_month"],
            "n_months": 0,
            "n_bull": 0,
            "n_neutral": 0,
            "n_bear": 0,
            "pct_bull": None,
            "pct_neutral": None,
            "pct_bear": None,
            "dominant": None,
        }
    vc = sub["regime"].value_counts()
    n_bull = int(vc.get("bull", 0))
    n_neu = int(vc.get("neutral", 0))
    n_bear = int(vc.get("bear", 0))
    dominant = vc.idxmax()
    return {
        "fold_id": int(fold_row["fold_id"]),
        "test_start_month": fold_row["test_start_month"],
        "test_end_month": fold_row["test_end_month"],
        "n_months": n,
        "n_bull": n_bull,
        "n_neutral": n_neu,
        "n_bear": n_bear,
        "pct_bull": round(100 * n_bull / n, 2),
        "pct_neutral": round(100 * n_neu / n, 2),
        "pct_bear": round(100 * n_bear / n, 2),
        "dominant": dominant,
    }


def main() -> int:
    ensure_output_dirs()

    # Recharge plis (même logique que étape 1)
    dates = pd.read_parquet(DATASET_DIR / "ml_dataset.parquet", columns=["date_cours"])[
        "date_cours"
    ]
    months = available_months_from_dates(dates)
    folds = generate_rolling_folds(months)
    folds_df = folds_to_frame(folds)

    regime_eom = load_regime_eom()

    # Historique global (tous mois disponibles dans le panel régime ∩ données)
    hist_months = [pd.Period(str(m), freq="M") for m in months]
    hist = regime_eom[regime_eom["mois"].isin(hist_months)]
    hist_vc = hist["regime"].value_counts()
    hist_n = len(hist)

    per_fold = [fold_test_regime_stats(row, regime_eom) for _, row in folds_df.iterrows()]
    per_df = pd.DataFrame(per_fold)

    # Agrégation : moyenne des % par pli + pool de tous les mois-test (avec double-comptage)
    # + mois uniques couverts au moins une fois par une fenêtre test
    mean_pct = {
        "bull": float(per_df["pct_bull"].mean()),
        "neutral": float(per_df["pct_neutral"].mean()),
        "bear": float(per_df["pct_bear"].mean()),
    }

    # Pool mois-test (chevauchement : un mois compte dans plusieurs plis)
    pooled_bull = int(per_df["n_bull"].sum())
    pooled_neu = int(per_df["n_neutral"].sum())
    pooled_bear = int(per_df["n_bear"].sum())
    pooled_n = pooled_bull + pooled_neu + pooled_bear

    # Mois uniques dans l'union des fenêtres test
    unique_test_months: set[pd.Period] = set()
    for _, row in folds_df.iterrows():
        start = pd.Period(row["test_start_month"], freq="M")
        end = pd.Period(row["test_end_month"], freq="M")
        unique_test_months.update(pd.period_range(start, end, freq="M"))
    uniq = regime_eom[regime_eom["mois"].isin(unique_test_months)]
    uniq_vc = uniq["regime"].value_counts()
    uniq_n = len(uniq)

    folds_with_bear = int((per_df["n_bear"] > 0).sum())
    folds_bear_ge_1_6 = int((per_df["pct_bear"] >= 100 / 6).sum())  # >= 1 mois / 6

    summary = {
        "n_folds": len(per_df),
        "historique_global_mois": {
            "n": hist_n,
            "bull": int(hist_vc.get("bull", 0)),
            "neutral": int(hist_vc.get("neutral", 0)),
            "bear": int(hist_vc.get("bear", 0)),
            "pct_bull": round(100 * hist_vc.get("bull", 0) / hist_n, 2) if hist_n else None,
            "pct_neutral": round(100 * hist_vc.get("neutral", 0) / hist_n, 2) if hist_n else None,
            "pct_bear": round(100 * hist_vc.get("bear", 0) / hist_n, 2) if hist_n else None,
        },
        "mean_pct_across_folds": {k: round(v, 2) for k, v in mean_pct.items()},
        "pooled_test_month_slots": {
            "n": pooled_n,
            "bull": pooled_bull,
            "neutral": pooled_neu,
            "bear": pooled_bear,
            "pct_bull": round(100 * pooled_bull / pooled_n, 2) if pooled_n else None,
            "pct_neutral": round(100 * pooled_neu / pooled_n, 2) if pooled_n else None,
            "pct_bear": round(100 * pooled_bear / pooled_n, 2) if pooled_n else None,
            "note": "somme des mois-test sur tous les plis (chevauchement compté plusieurs fois)",
        },
        "unique_months_in_any_test_window": {
            "n": uniq_n,
            "bull": int(uniq_vc.get("bull", 0)),
            "neutral": int(uniq_vc.get("neutral", 0)),
            "bear": int(uniq_vc.get("bear", 0)),
            "pct_bull": round(100 * uniq_vc.get("bull", 0) / uniq_n, 2) if uniq_n else None,
            "pct_neutral": round(100 * uniq_vc.get("neutral", 0) / uniq_n, 2) if uniq_n else None,
            "pct_bear": round(100 * uniq_vc.get("bear", 0) / uniq_n, 2) if uniq_n else None,
        },
        "folds_with_at_least_one_bear_month": folds_with_bear,
        "folds_with_bear_share_ge_1_of_6": folds_bear_ge_1_6,
        "reference_fixed_test_bear_pct": 3.1,
        "reference_fixed_val_bear_pct": 25.0,
    }

    out_csv = REPORTS_DIR / "wf_etape2_regime_per_fold.csv"
    out_json = REPORTS_DIR / "wf_etape2_regime_summary.json"
    per_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Walk-forward Etape 2 — Couverture des regimes (fenetres TEST) ===")
    print(f"Plis analyses : {len(per_df)}")
    print()
    print("--- Historique global (tous mois donnees) ---")
    hg = summary["historique_global_mois"]
    print(
        f"  n={hg['n']} | bull={hg['bull']} ({hg['pct_bull']}%) | "
        f"neutral={hg['neutral']} ({hg['pct_neutral']}%) | "
        f"bear={hg['bear']} ({hg['pct_bear']}%)"
    )
    print()
    print("--- Moyenne des % regime par pli (descriptif, plis chevauchants) ---")
    mp = summary["mean_pct_across_folds"]
    print(f"  bull={mp['bull']}% | neutral={mp['neutral']}% | bear={mp['bear']}%")
    print()
    print("--- Mois uniques couverts par au moins une fenetre test ---")
    uq = summary["unique_months_in_any_test_window"]
    print(
        f"  n={uq['n']} | bull={uq['bull']} ({uq['pct_bull']}%) | "
        f"neutral={uq['neutral']} ({uq['pct_neutral']}%) | "
        f"bear={uq['bear']} ({uq['pct_bear']}%)"
    )
    print()
    print("--- Pool mois-test (avec double-comptage du chevauchement) ---")
    pl = summary["pooled_test_month_slots"]
    print(
        f"  n_slots={pl['n']} | bull={pl['pct_bull']}% | "
        f"neutral={pl['pct_neutral']}% | bear={pl['pct_bear']}%"
    )
    print()
    print(
        f"Plis avec >=1 mois bear : {folds_with_bear}/{len(per_df)} "
        f"({100 * folds_with_bear / len(per_df):.1f}%)"
    )
    print(
        f"Comparaison split fixe test (bear~3.1%) vs WF mean bear={mp['bear']}% "
        f"vs historique={hg['pct_bear']}%"
    )

    # Aperçu distribution dominant
    print("\nRegime dominant du pli (count) :")
    print(per_df["dominant"].value_counts().to_string())

    print("\nExemples plis les plus 'bear' (top 5 pct_bear) :")
    print(
        per_df.nlargest(5, "pct_bear")[
            ["fold_id", "test_start_month", "test_end_month", "n_bear", "pct_bear", "dominant"]
        ].to_string(index=False)
    )
    print("\nExemples plis les moins 'bear' (bottom 5) :")
    print(
        per_df.nsmallest(5, "pct_bear")[
            ["fold_id", "test_start_month", "test_end_month", "n_bear", "pct_bear", "dominant"]
        ].to_string(index=False)
    )

    print(f"\nArtefacts :\n  - {out_csv}\n  - {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
