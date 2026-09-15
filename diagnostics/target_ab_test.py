"""
A/B cible : alpha_ajuste_risque vs excess_vs_masi (sans /vol).

1) Baselines rapides (ridge + lightgbm) sur les 2 cibles
2) TFT court sur excess_vs_masi (hparams Optuna best)

Usage (racine projet) :
    py -3 diagnostics/target_ab_test.py
    py -3 diagnostics/target_ab_test.py --skip-tft
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import DATASET_DIR, FEATURES_DIR, REPORTS_DIR, load_env_file
from bvc_recommender.features.dataset_builder import EXCESS_TARGET_COLUMN, TARGET_COLUMN
from bvc_recommender.models.baseline_models import train_and_evaluate_baselines
from bvc_recommender.models.regime_detector import REGIME_COLUMNS
from bvc_recommender.models.stock_scorer import train_and_evaluate_tft

load_env_file(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("target_ab")


def _load_dataset() -> pd.DataFrame:
    path = DATASET_DIR / "ml_dataset.parquet"
    df = pd.read_parquet(path)
    if EXCESS_TARGET_COLUMN not in df.columns:
        if "vol_baissiere_20d" not in df.columns:
            raise ValueError("Impossible de reconstruire excess_vs_masi (vol manquante).")
        df[EXCESS_TARGET_COLUMN] = df[TARGET_COLUMN] * df["vol_baissiere_20d"]
        logger.info(
            "excess_vs_masi reconstruit = alpha * vol_baissiere_20d (corr pearson=%.3f)",
            df[TARGET_COLUMN].corr(df[EXCESS_TARGET_COLUMN]),
        )
    return df


def _load_regime() -> pd.DataFrame:
    path = FEATURES_DIR / "features_indices.parquet"
    fi = pd.read_parquet(path)
    keep = ["date_cours", *REGIME_COLUMNS]
    missing = [c for c in keep if c not in fi.columns]
    if missing:
        raise ValueError(f"Colonnes régime manquantes : {missing}")
    return fi[keep].copy()


def _summarize_baselines(results: dict[str, Any], target: str) -> dict[str, Any]:
    out: dict[str, Any] = {"target": target, "models": {}}
    for name, model in results.get("models", {}).items():
        out["models"][name] = {
            split: {
                "ic_mean": m.get("metrics", {}).get("ic_mean"),
                "hit_ratio": m.get("metrics", {}).get("hit_ratio"),
                "sharpe_long_short": m.get("metrics", {}).get("sharpe_long_short"),
            }
            for split, m in model.get("splits", {}).items()
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tft", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--max-train-rows", type=int, default=12000)
    args = parser.parse_args()

    df = _load_dataset()
    regime = _load_regime()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Division par vol_baissiere_20d crée des outliers extrêmes (alpha max~10^3) "
            "qui dégradent QuantileLoss ; excess_vs_masi devrait mieux s'apprendre."
        ),
        "baselines": {},
    }

    for target in (TARGET_COLUMN, EXCESS_TARGET_COLUMN):
        logger.info("=== Baselines cible=%s ===", target)
        bl = train_and_evaluate_baselines(
            df,
            models=("ridge", "lightgbm"),
            target_col=target,
        )
        report["baselines"][target] = _summarize_baselines(bl, target)

    # Décision rapide : meilleur IC test LightGBM
    def _lgbm_test_ic(target: str) -> float:
        m = report["baselines"][target]["models"].get("lightgbm", {}).get("test", {})
        return float(m.get("ic_mean") or float("nan"))

    ic_alpha = _lgbm_test_ic(TARGET_COLUMN)
    ic_excess = _lgbm_test_ic(EXCESS_TARGET_COLUMN)
    # Baselines (arbres) tolèrent les outliers ; le TFT / QuantileLoss non.
    # On force donc le TFT sur excess_vs_masi (hypothèse principale).
    tft_target = EXCESS_TARGET_COLUMN
    report["baseline_verdict"] = {
        "lightgbm_test_ic_alpha": ic_alpha,
        "lightgbm_test_ic_excess": ic_excess,
        "ridge_note": "Ridge a souvent un IC test >0 ; LGBM overfitte les 2 cibles",
        "preferred_for_tft": tft_target,
        "reason": "QuantileLoss TFT sensible aux outliers alpha (max~10^3) ; excess plus stable",
    }
    logger.info(
        "Verdict baselines : LGBM test IC alpha=%.4f | excess=%.4f → TFT forcé sur %s",
        ic_alpha,
        ic_excess,
        tft_target,
    )

    if not args.skip_tft:
        logger.info("=== TFT court cible=%s ===", tft_target)
        tft = train_and_evaluate_tft(
            df,
            regime,
            known_reals=list(REGIME_COLUMNS),
            target_col=tft_target,
            max_epochs=args.max_epochs,
            max_train_rows=args.max_train_rows,
            learning_rate=2.5e-4,
            hidden_size=16,
            attention_head_size=4,
            dropout=0.2,
            early_stopping_patience=3,
            save_model=False,
        )
        report["tft"] = {
            "target": tft_target,
            "hparams": tft.get("tft_hparams"),
            "reference_tft_alpha_test_ic": -0.0777,
            "splits": {
                s: {
                    "ic_mean": d.get("metrics", {}).get("ic_mean"),
                    "hit_ratio": d.get("metrics", {}).get("hit_ratio"),
                    "sharpe_long_short": d.get("metrics", {}).get("sharpe_long_short"),
                }
                for s, d in tft.get("splits", {}).items()
            },
        }
        test_ic = report["tft"]["splits"].get("test", {}).get("ic_mean")
        if test_ic is not None:
            delta = test_ic - (-0.0777)
            report["tft_verdict"] = (
                f"IC test={test_ic:.4f} sur {tft_target} "
                f"(delta vs TFT/alpha={delta:+.4f})"
            )
        else:
            report["tft_verdict"] = "TFT sans métrique test"
        logger.info("Verdict TFT : %s", report["tft_verdict"])

    out = REPORTS_DIR / "target_ab_test_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Rapport : %s", out)
    print(json.dumps(report.get("baseline_verdict"), indent=2))
    if "tft" in report:
        print(json.dumps(report["tft"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
