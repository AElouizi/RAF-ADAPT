"""
Sous-module B — Optuna TFT + réentraînement (régime à seuils).

1) optimize_hyperparameters sur val 2021-2022
2) réentraîne sur train 2015-2020 avec les meilleurs hparams
3) métriques IC / Hit / Sharpe sur val et test

Usage :
    python -m bvc_recommender.scripts.run_step6_optuna
    python -m bvc_recommender.scripts.run_step6_optuna --n-trials 25 --max-epochs 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    DATASET_DIR,
    FEATURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SPLIT_TEST_START,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    load_env_file,
)
from bvc_recommender.models.regime_detector import (  # noqa: E402
    DOCUMENTED_THRESHOLDS,
    REGIME_COLUMNS,
)
from bvc_recommender.models.stock_scorer import (  # noqa: E402
    DEFAULT_ENCODER_LENGTH,
    DEFAULT_MAX_TRAIN_ROWS,
    REGIME_ENCODING_ONEHOT,
    TARGET_TYPE_ALPHA,
    optimize_hyperparameters,
    train_and_evaluate_tft,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HMM_BASELINE = {
    "note": "TFT + HMM 3 états (référence antérieure)",
    "val": {"ic_mean": -0.008, "hit_ratio": None, "sharpe_long_short": None},
    "test": {"ic_mean": -0.129, "hit_ratio": None, "sharpe_long_short": None},
}


def _write_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "step6_optuna_threshold_report.json"
    md_path = REPORTS_DIR / "tft_threshold_regime.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = [
        "# TFT + régime à seuils — Optuna + réentraînement",
        "",
        f"- **Généré** : {report['generated_at']}",
        f"- **Régime** : is_bull / is_neutral / is_bear (seuils both-AND)",
        f"- **Seuils** : {report.get('thresholds')}",
        f"- **Cible** : {report.get('target_type', 'alpha_ajuste_risque')} "
        f"(train) ; métriques vs alpha_ajuste_risque",
        f"- **Régime encoding** : {report.get('regime_encoding', 'onehot')}",
        f"- **Sequence length** : {report.get('sequence_length', 20)}",
        f"- **Splits** : train ≤ {SPLIT_TRAIN_END} | val ≤ {SPLIT_VAL_END} | test ≥ {SPLIT_TEST_START}",
        "",
        "## Meilleurs hyperparamètres (Optuna / val_loss)",
        "",
        f"- Trials : {report['optuna'].get('n_trials')}",
        f"- Best val_loss : {report['optuna'].get('best_val_loss')}",
        f"- Params : `{report['optuna'].get('best_params')}`",
        "",
        "## Métriques TFT (seuils)",
        "",
        "| Split | IC | Hit Ratio | Sharpe L/S |",
        "|-------|-----|-----------|------------|",
    ]
    for split_name in ("val", "test"):
        m = report.get("tft_splits", {}).get(split_name, {}).get("metrics", {})
        lines.append(
            f"| {split_name} | {m.get('ic_mean', float('nan')):.4f} | "
            f"{m.get('hit_ratio', float('nan')):.1%} | "
            f"{m.get('sharpe_long_short', float('nan')):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Comparaison vs HMM (référence)",
            "",
            "| Split | IC seuils | IC HMM |",
            "|-------|-----------|--------|",
            f"| val | {report.get('tft_splits', {}).get('val', {}).get('metrics', {}).get('ic_mean', float('nan')):.4f} | {HMM_BASELINE['val']['ic_mean']:.4f} |",
            f"| test | {report.get('tft_splits', {}).get('test', {}).get('metrics', {}).get('ic_mean', float('nan')):.4f} | {HMM_BASELINE['test']['ic_mean']:.4f} |",
            "",
            f"**Verdict** : {report.get('verdict', '—')}",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="TFT Optuna + régime à seuils")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--max-epochs", type=int, default=15, help="Epochs max par essai Optuna")
    parser.add_argument("--final-epochs", type=int, default=20, help="Epochs réentraînement final")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="16,32,64",
        help="Liste CSV des hidden_size Optuna (ex. 16,32,64 — exclure 128 sur CPU)",
    )
    parser.add_argument(
        "--dropouts",
        type=str,
        default="0.1,0.2,0.3",
        help="Liste CSV dropout Optuna (rétention v1 : 0.1,0.2,0.3 ; retest : ajouter 0.4,0.5)",
    )
    parser.add_argument(
        "--regime-encoding",
        type=str,
        default=REGIME_ENCODING_ONEHOT,
        choices=["onehot", "ordinal"],
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default=TARGET_TYPE_ALPHA,
        choices=["alpha_ajuste_risque", "rang_cross_sectionnel"],
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_ENCODER_LENGTH,
        help="Longueur encodeur TFT (défaut 20)",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Réentraîner avec hparams par défaut (ou --hparams-json)",
    )
    parser.add_argument("--hparams-json", type=str, default="")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    hidden_sizes = [int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip()]
    dropouts = [float(x.strip()) for x in args.dropouts.split(",") if x.strip()]
    ml_path = DATASET_DIR / "ml_dataset.parquet"
    features_path = FEATURES_DIR / "features_indices.parquet"
    if not ml_path.is_file() or not features_path.is_file():
        raise FileNotFoundError("ml_dataset ou features_indices manquant.")

    dataset = pd.read_parquet(ml_path)
    regime = pd.read_parquet(features_path)
    missing = [c for c in REGIME_COLUMNS if c not in regime.columns]
    if missing:
        raise ValueError(
            f"Colonnes régime absentes : {missing}. Relancer run_step4."
        )

    logger.info(
        "Optuna TFT seuils | trials=%s epochs=%s patience=%s rows≤%s hs=%s drop=%s | "
        "regime=%s target=%s seq=%s",
        args.n_trials,
        args.max_epochs,
        args.patience,
        args.max_train_rows,
        hidden_sizes,
        dropouts,
        args.regime_encoding,
        args.target_type,
        args.sequence_length,
    )

    if args.hparams_json:
        best_params = json.loads(Path(args.hparams_json).read_text(encoding="utf-8"))
        if "best_params" in best_params and isinstance(best_params["best_params"], dict):
            best_params = best_params["best_params"]
        optuna_report: dict[str, Any] = {
            "best_params": best_params,
            "best_val_loss": None,
            "n_trials": 0,
            "source": args.hparams_json,
        }
    elif args.skip_optuna:
        best_params = {
            "hidden_size": 16,
            "attention_head_size": 2,
            "dropout": 0.1,
            "learning_rate": 0.03,
        }
        optuna_report = {
            "best_params": best_params,
            "best_val_loss": None,
            "n_trials": 0,
            "source": "defaults",
        }
    else:
        optuna_report = optimize_hyperparameters(
            dataset,
            regime,
            regime_encoding=args.regime_encoding,
            target_type=args.target_type,
            sequence_length=args.sequence_length,
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            max_train_rows=args.max_train_rows,
            early_stopping_patience=args.patience,
            hidden_sizes=hidden_sizes,
            dropouts=dropouts,
        )
        best_params = optuna_report["best_params"]
        opt_path = REPORTS_DIR / "optuna_threshold_study.json"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        opt_path.write_text(
            json.dumps(optuna_report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Étude Optuna sauvegardée : %s", opt_path)

    logger.info("Réentraînement final avec %s", best_params)
    tft_results = train_and_evaluate_tft(
        dataset,
        regime,
        regime_encoding=args.regime_encoding,
        target_type=args.target_type,
        sequence_length=args.sequence_length,
        max_epochs=args.final_epochs,
        max_train_rows=args.max_train_rows,
        learning_rate=float(best_params["learning_rate"]),
        hidden_size=int(best_params["hidden_size"]),
        attention_head_size=int(best_params["attention_head_size"]),
        dropout=float(best_params["dropout"]),
        early_stopping_patience=args.patience,
        save_model=True,
    )

    preds = tft_results.pop("predictions", pd.DataFrame())
    if not preds.empty:
        pred_path = DATASET_DIR / "tft_predictions.parquet"
        try:
            preds.to_parquet(pred_path, index=False)
        except Exception:
            pred_path = DATASET_DIR / "tft_predictions.csv"
            preds.to_csv(pred_path, index=False)
        logger.info("Prédictions : %s (%s lignes)", pred_path, len(preds))

    val_ic = tft_results.get("splits", {}).get("val", {}).get("metrics", {}).get("ic_mean")
    test_ic = tft_results.get("splits", {}).get("test", {}).get("metrics", {}).get("ic_mean")
    hmm_test = HMM_BASELINE["test"]["ic_mean"]
    if test_ic is None:
        verdict = "Métriques test indisponibles."
    elif test_ic > 0 and test_ic > hmm_test:
        verdict = (
            f"IC test positif ({test_ic:.4f}) et supérieur au HMM ({hmm_test:.4f}) "
            "— ablation step5 autorisée."
        )
    elif test_ic > hmm_test:
        verdict = (
            f"IC test ({test_ic:.4f}) meilleur que HMM ({hmm_test:.4f}) mais encore "
            f"{'négatif' if test_ic < 0 else 'faible'} — ne pas lancer l'ablation step5."
        )
    else:
        verdict = (
            f"IC test ({test_ic:.4f}) ≤ HMM ({hmm_test:.4f}) — ne pas lancer l'ablation step5."
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "BVC Recommender — TFT + régime à seuils",
        "random_state": RANDOM_STATE,
        "regime_injection": {
            "variables": list(REGIME_COLUMNS),
            "role": "time_varying_known_reals",
            "method": "threshold_both_33_67",
            "encoding": args.regime_encoding,
        },
        "target_type": args.target_type,
        "sequence_length": args.sequence_length,
        "thresholds": DOCUMENTED_THRESHOLDS,
        "alpha_target_note": (
            "alpha_ajuste_risque = (forward_ret_action - forward_ret_MASI) / vol_baissiere_20d "
            "— ce n'est PAS l'alpha de Jensen (r - beta*rm)."
        ),
        "optuna": optuna_report,
        "final_hparams": best_params,
        "tft_splits": tft_results.get("splits", {}),
        "hmm_baseline": HMM_BASELINE,
        "verdict": verdict,
        "val_ic": val_ic,
        "test_ic": test_ic,
    }
    json_path, md_path = _write_reports(report)
    logger.info("Rapports : %s | %s", json_path, md_path)
    logger.info("Verdict : %s", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
