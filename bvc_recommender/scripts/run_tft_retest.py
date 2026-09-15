"""
Retest TFT ciblé — un run à la fois (capacité / régime / cible / séquence).

Ne lance PAS toute la grille : un seul ``--run-id`` par invocation.
Après chaque run, une ligne est ajoutée à ``reports/tft_retest_results.csv``.

Exemples :
    # Run A — Optuna capacité élargie (hs 2/4/8 + dropout jusqu'à 0.5)
    python -m bvc_recommender.scripts.run_tft_retest --run-id A_capacity \\
        --regime-encoding onehot --target-type alpha_ajuste_risque --sequence-length 20

    # Run B — ordinal, hparams figés (JSON best de A)
    python -m bvc_recommender.scripts.run_tft_retest --run-id B_ordinal \\
        --regime-encoding ordinal --skip-optuna --hparams-json path/to/best.json

    # Baseline rappel (hparams v1, sans Optuna)
    python -m bvc_recommender.scripts.run_tft_retest --run-id baseline_v1 --skip-optuna \\
        --hparams-json bvc_recommender/reports/tft_baseline_v1_hparams.json
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
from bvc_recommender.models.regime_detector import REGIME_COLUMNS  # noqa: E402
from bvc_recommender.models.stock_scorer import (  # noqa: E402
    DEFAULT_DROPOUTS,
    DEFAULT_ENCODER_LENGTH,
    DEFAULT_HIDDEN_SIZES,
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

RESULTS_CSV = REPORTS_DIR / "tft_retest_results.csv"
RESULTS_COLS = [
    "run_id",
    "hs",
    "heads",
    "dropout",
    "lr",
    "regime_encoding",
    "target_type",
    "sequence_length",
    "IC_val",
    "IC_test",
    "hit_ratio",
    "sharpe_LS",
    "generated_at",
]

# Baseline TFT v1 (Optuna seuils) — ne pas relancer sauf demande explicite
BASELINE_V1_HPARAMS = {
    "hidden_size": 16,
    "attention_head_size": 4,
    "dropout": 0.2,
    "learning_rate": 0.00024970737145052745,
}
BASELINE_V1_METRICS = {
    "IC_val": -0.04546143173843357,
    "IC_test": -0.07768310472510566,
    "hit_ratio": 0.4779548288762644,
    "sharpe_LS": -1.8079664854212405,
}


def _ensure_baseline_row(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": "baseline_v1",
        "hs": BASELINE_V1_HPARAMS["hidden_size"],
        "heads": BASELINE_V1_HPARAMS["attention_head_size"],
        "dropout": BASELINE_V1_HPARAMS["dropout"],
        "lr": BASELINE_V1_HPARAMS["learning_rate"],
        "regime_encoding": "onehot",
        "target_type": "alpha_ajuste_risque",
        "sequence_length": DEFAULT_ENCODER_LENGTH,
        **BASELINE_V1_METRICS,
        "generated_at": "from_step6_optuna_threshold_report",
    }
    if path.is_file():
        existing = pd.read_csv(path)
        if (existing["run_id"] == "baseline_v1").any():
            return
        out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row], columns=RESULTS_COLS)
    out.to_csv(path, index=False)


def _append_result(path: Path, row: dict[str, Any]) -> Path:
    _ensure_baseline_row(path)
    existing = pd.read_csv(path)
    # Remplace une ligne du même run_id si re-run
    existing = existing[existing["run_id"] != row["run_id"]]
    out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    # Ordre colonnes stable
    for c in RESULTS_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = out[RESULTS_COLS]
    out.to_csv(path, index=False)
    return path


def _format_table(path: Path) -> str:
    if not path.is_file():
        return "(pas encore de résultats)"
    df = pd.read_csv(path)
    cols = [
        "run_id",
        "hs",
        "heads",
        "dropout",
        "regime_encoding",
        "target_type",
        "sequence_length",
        "IC_val",
        "IC_test",
        "hit_ratio",
        "sharpe_LS",
    ]
    view = df[[c for c in cols if c in df.columns]].copy()
    try:
        return view.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        # tabulate optionnel — fallback texte aligné
        return view.to_string(index=False, float_format=lambda x: f"{x:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Retest TFT — un levier à la fois")
    parser.add_argument("--run-id", type=str, required=True, help="Identifiant unique du run")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--max-epochs", type=int, default=15, help="Epochs max par essai Optuna")
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default=",".join(str(x) for x in DEFAULT_HIDDEN_SIZES),
        help="Liste CSV hidden_size Optuna (défaut: 2,4,8,16,32,64)",
    )
    parser.add_argument(
        "--dropouts",
        type=str,
        default=",".join(str(x) for x in DEFAULT_DROPOUTS),
        help="Liste CSV dropout Optuna (défaut: 0.1..0.5)",
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
        help="Longueur encodeur TFT (défaut 20 ; court ex. 6)",
    )
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--hparams-json", type=str, default="")
    parser.add_argument(
        "--seed-baseline-only",
        action="store_true",
        help="N'écrit que la ligne baseline_v1 dans le CSV, sans entraîner",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_baseline_row(RESULTS_CSV)

    if args.seed_baseline_only:
        logger.info("Baseline v1 inscrite dans %s — pas d'entraînement.", RESULTS_CSV)
        print(_format_table(RESULTS_CSV))
        return 0

    # Rappel baseline : ne pas relancer si run-id = baseline_v1
    if args.run_id == "baseline_v1" and not args.hparams_json and args.skip_optuna:
        logger.info("baseline_v1 déjà dans le CSV (métriques step6) — pas de relance.")
        print(_format_table(RESULTS_CSV))
        return 0

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
        raise ValueError(f"Colonnes régime absentes : {missing}. Relancer run_step4.")

    logger.info(
        "TFT retest run=%s | regime=%s target=%s seq=%s | trials=%s hs=%s drop=%s | "
        "splits train≤%s / val≤%s / test≥%s",
        args.run_id,
        args.regime_encoding,
        args.target_type,
        args.sequence_length,
        args.n_trials,
        hidden_sizes,
        dropouts,
        SPLIT_TRAIN_END,
        SPLIT_VAL_END,
        SPLIT_TEST_START,
    )

    if args.hparams_json:
        best_params = json.loads(Path(args.hparams_json).read_text(encoding="utf-8"))
        # Accepte soit {"best_params": {...}} soit un dict plat
        if "best_params" in best_params and isinstance(best_params["best_params"], dict):
            best_params = best_params["best_params"]
        optuna_report: dict[str, Any] = {
            "best_params": best_params,
            "best_val_loss": None,
            "n_trials": 0,
            "source": args.hparams_json,
        }
    elif args.skip_optuna:
        best_params = dict(BASELINE_V1_HPARAMS)
        optuna_report = {
            "best_params": best_params,
            "best_val_loss": None,
            "n_trials": 0,
            "source": "baseline_v1_hparams",
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
            study_name=f"tft_retest_{args.run_id}",
            hidden_sizes=hidden_sizes,
            dropouts=dropouts,
        )
        best_params = optuna_report["best_params"]
        opt_path = REPORTS_DIR / f"optuna_retest_{args.run_id}.json"
        opt_path.write_text(
            json.dumps(optuna_report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Étude Optuna : %s", opt_path)

    logger.info("Réentraînement final | %s", best_params)
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

    if tft_results.get("anomaly"):
        logger.error("Run %s STOP — anomalie : %s", args.run_id, tft_results["anomaly"])
        return 2

    preds = tft_results.pop("predictions", pd.DataFrame())
    if not preds.empty:
        pred_path = DATASET_DIR / f"tft_predictions_{args.run_id}.parquet"
        try:
            preds.to_parquet(pred_path, index=False)
        except Exception:
            pred_path = DATASET_DIR / f"tft_predictions_{args.run_id}.csv"
            preds.to_csv(pred_path, index=False)
        logger.info("Prédictions : %s (%s lignes)", pred_path, len(preds))

    val_m = tft_results.get("splits", {}).get("val", {}).get("metrics", {})
    test_m = tft_results.get("splits", {}).get("test", {}).get("metrics", {})
    row = {
        "run_id": args.run_id,
        "hs": int(best_params["hidden_size"]),
        "heads": int(best_params["attention_head_size"]),
        "dropout": float(best_params["dropout"]),
        "lr": float(best_params["learning_rate"]),
        "regime_encoding": args.regime_encoding,
        "target_type": args.target_type,
        "sequence_length": args.sequence_length,
        "IC_val": val_m.get("ic_mean"),
        "IC_test": test_m.get("ic_mean"),
        "hit_ratio": test_m.get("hit_ratio"),
        "sharpe_LS": test_m.get("sharpe_long_short"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_result(RESULTS_CSV, row)

    report = {
        "generated_at": row["generated_at"],
        "run_id": args.run_id,
        "regime_encoding": args.regime_encoding,
        "target_type": args.target_type,
        "sequence_length": args.sequence_length,
        "optuna": optuna_report,
        "final_hparams": best_params,
        "tft_splits": tft_results.get("splits", {}),
        "result_row": row,
    }
    report_path = REPORTS_DIR / f"tft_retest_{args.run_id}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Sauvegarde hparams best pour enchaîner Run B/C/D sans Optuna
    best_path = REPORTS_DIR / f"tft_retest_{args.run_id}_best_hparams.json"
    best_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    logger.info("Rapport : %s | CSV : %s", report_path, RESULTS_CSV)
    print("\n=== Tableau récapitulatif ===\n")
    print(_format_table(RESULTS_CSV))
    print(
        f"\nRun {args.run_id} | IC_val={row['IC_val']:.4f} | IC_test={row['IC_test']:.4f} | "
        f"Hit={row['hit_ratio']:.1%} | Sharpe L/S={row['sharpe_LS']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
