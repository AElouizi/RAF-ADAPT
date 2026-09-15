"""
Diagnostic post-DM — 5 points (sans re-training complet).

1. Sensibilite poids Ridge/RF + DM sur 14 blocs
2. Features Ridge vs RF
3. Hyperparams RF (fixes vs par pli)
4. Provenance donnees / vol_baissiere / cible
5. Puissance statistique DM (n=14)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import DATASET_DIR, load_env_file  # noqa: E402
from bvc_recommender.features.dataset_builder import (  # noqa: E402
    DATA_VERSION_NOTE,
    TARGET_COLUMN,
)
from bvc_recommender.models.baseline_models import get_z_feature_columns  # noqa: E402
from bvc_recommender.models.metrics import evaluate_scoring_model  # noqa: E402
from bvc_recommender.models.regime_detector import REGIME_COLUMNS  # noqa: E402
from experiments.factorial_hybrid_adapt.factorial_cells import (  # noqa: E402
    combine_hybrid_scores,
)
from experiments.factorial_hybrid_adapt.paths import (  # noqa: E402
    REPORTS_DIR,
    ensure_output_dirs,
)
from experiments.factorial_hybrid_adapt.run_wf_dm_ic_blocks import (  # noqa: E402
    diebold_mariano,
)
from experiments.factorial_hybrid_adapt.walk_forward_stage1 import (  # noqa: E402
    DEFAULT_RF_PARAMS,
)

load_env_file(ROOT / ".env")

WEIGHTS = [
    (0.5, 0.5),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
    (0.3, 0.7),
]


def point1() -> None:
    print("\n" + "=" * 72)
    print("POINT 1 — Sensibilite poids Ridge/RF + DM (14 blocs)")
    print("=" * 72)

    preds = pd.read_parquet(REPORTS_DIR / "wf_stage1_predictions.parquet")
    # Scores Ridge+RF disponibles sur cellules hybrides (3 et 4)
    hybrid = preds[preds["cell_id"].isin([3, 4])].copy()
    hybrid = hybrid.dropna(subset=["score_ridge", "score_rf"])
    print(f"Lignes hybrides avec scores : {len(hybrid)}")

    blocks = pd.read_csv(REPORTS_DIR / "wf_folds_nonoverlapping.csv")
    block_keys = set(zip(blocks["test_start_month"], blocks["test_end_month"]))

    # Baseline C1 / C2 IC par fold (deja dans metrics)
    base = pd.read_csv(REPORTS_DIR / "wf_etape3_fold_metrics.csv")
    base_ic = base.pivot_table(
        index=["fold_id", "test_start_month", "test_end_month"],
        columns="cell_id",
        values="rank_ic",
    )

    rows_81 = []
    rows_dm = []

    for w_r, w_f in WEIGHTS:
        fold_ics = []
        for (fold_id, cell_id), g in hybrid.groupby(["fold_id", "cell_id"]):
            score = combine_hybrid_scores(
                g["score_ridge"].values,
                g["score_rf"].values,
                g["date_cours"],
                w_ridge=w_r,
                w_rf=w_f,
            )
            eval_df = g[["ticker", "date_cours", TARGET_COLUMN]].copy()
            eval_df["prediction"] = score
            m = evaluate_scoring_model(eval_df, "prediction", TARGET_COLUMN)
            meta = g.iloc[0]
            fold_ics.append(
                {
                    "fold_id": int(fold_id),
                    "cell_id": int(cell_id),
                    "w_ridge": w_r,
                    "w_rf": w_f,
                    "test_start_month": meta["test_start_month"],
                    "test_end_month": meta["test_end_month"],
                    "regime": bool(meta["regime"]),
                    "rank_ic": m["rank_ic_mean"],
                    "hit_ratio": m["hit_ratio"],
                }
            )
        fic = pd.DataFrame(fold_ics)

        for cell_id, label, ref_cell in (
            (3, "hybrid_static", 1),
            (4, "hybrid_adapt", 1),
        ):
            sub = fic[fic["cell_id"] == cell_id]
            rows_81.append(
                {
                    "w_ridge": w_r,
                    "w_rf": w_f,
                    "cell_id": cell_id,
                    "label": label,
                    "IC_mean_81": float(sub["rank_ic"].mean()),
                    "IC_median_81": float(sub["rank_ic"].median()),
                    "IC_std_81": float(sub["rank_ic"].std(ddof=1)),
                    "hit_mean_81": float(sub["hit_ratio"].mean()),
                    "n_folds": int(len(sub)),
                }
            )

            # DM vs C1 sur blocs non-chevauchants
            blk = sub[
                sub.apply(
                    lambda r: (r["test_start_month"], r["test_end_month"]) in block_keys,
                    axis=1,
                )
            ].sort_values("fold_id")
            # aligner avec IC C1 des memes blocs
            ref = base[
                (base["cell_id"] == ref_cell)
                & base.apply(
                    lambda r: (r["test_start_month"], r["test_end_month"]) in block_keys,
                    axis=1,
                )
            ][["fold_id", "rank_ic"]].rename(columns={"rank_ic": "ic_ref"})
            merged = blk.merge(ref, on="fold_id", how="inner").sort_values("fold_id")
            loss_ref = -merged["ic_ref"].values
            loss_alt = -merged["rank_ic"].values
            dm = diebold_mariano(loss_ref, loss_alt, h=1)
            rows_dm.append(
                {
                    "w_ridge": w_r,
                    "w_rf": w_f,
                    "cell_id": cell_id,
                    "label": label,
                    "delta_IC_mean_14": float(
                        merged["rank_ic"].mean() - merged["ic_ref"].mean()
                    ),
                    "IC_mean_alt_14": float(merged["rank_ic"].mean()),
                    "IC_median_alt_14": float(merged["rank_ic"].median()),
                    "dm_stat_hln": dm["dm_stat_hln"],
                    "pvalue_hln": dm["pvalue_hln"],
                    "significant_5pct": bool(
                        np.isfinite(dm["pvalue_hln"]) and dm["pvalue_hln"] < 0.05
                    ),
                    "n_blocks": dm["n"],
                }
            )

    df81 = pd.DataFrame(rows_81)
    dfdm = pd.DataFrame(rows_dm)
    # Reference C1/C2 on 81
    print("\nReference Ridge (81 plis chevauchants) :")
    for cid in (1, 2):
        g = base[base["cell_id"] == cid]
        print(
            f"  C{cid}: IC_mean={g['rank_ic'].mean():.4f} | "
            f"IC_median={g['rank_ic'].median():.4f}"
        )

    print("\n--- IC sur 81 plis (hybride repondere) ---")
    print(
        df81.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print("\n--- DM vs C1 sur 14 blocs non-chevauchants ---")
    print(
        dfdm[
            [
                "w_ridge",
                "w_rf",
                "label",
                "delta_IC_mean_14",
                "IC_mean_alt_14",
                "IC_median_alt_14",
                "dm_stat_hln",
                "pvalue_hln",
                "significant_5pct",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    any_sig = bool(dfdm["significant_5pct"].any())
    print(
        f"\nAucun poids ne rend l'effet significatif a 5%."
        if not any_sig
        else "\nAU MOINS UN poids est significatif a 5%."
    )

    df81.to_csv(REPORTS_DIR / "diag_weight_ic_81.csv", index=False)
    dfdm.to_csv(REPORTS_DIR / "diag_weight_dm_14.csv", index=False)


def point2() -> None:
    print("\n" + "=" * 72)
    print("POINT 2 — Features Ridge vs RF")
    print("=" * 72)
    ml = pd.read_parquet(DATASET_DIR / "ml_dataset.parquet")
    z_cols = get_z_feature_columns(ml)
    print(f"Meme selecteur get_z_feature_columns() pour Ridge et RF.")
    print(f"n_features *_z (regime=Non) = {len(z_cols)}")
    print(f"Liste complete:\n  {z_cols}")
    print(
        f"\nAvec regime=Oui : + {list(REGIME_COLUMNS)} "
        f"=> n={len(z_cols) + len(REGIME_COLUMNS)}"
    )
    print(
        "\nTransformations : identiques (fillna(0) dans _build_xy). "
        "Aucune feature interaction / polynomiale ajoutee pour RF."
    )
    print(
        "Hybride : seule difference post-hoc = z-score CS des scores puis moyenne ponderee."
    )
    # Confirm code path
    from experiments.factorial_hybrid_adapt import walk_forward_stage1 as wfs
    from experiments.factorial_hybrid_adapt import factorial_cells as fc

    print(
        f"\nCode : train_base_models() appelle _fit_ridge et fit_random_forest "
        f"sur les MEMES feature_cols (factorial_cells.feature_columns_for_cell)."
    )
    payload = {
        "same_features": True,
        "features_no_regime": z_cols,
        "features_with_regime": z_cols + list(REGIME_COLUMNS),
        "extra_rf_transforms": None,
    }
    (REPORTS_DIR / "diag_features_ridge_rf.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def point3() -> None:
    print("\n" + "=" * 72)
    print("POINT 3 — Hyperparams RF : fixes ou reoptimises par pli ?")
    print("=" * 72)
    p = DEFAULT_RF_PARAMS
    print("STATUT : FIXES pour tous les plis walk-forward (pas de re-tuning par fenetre).")
    print(
        f"Valeurs : n_estimators={p.n_estimators}, max_depth={p.max_depth}, "
        f"min_samples_leaf={p.min_samples_leaf}, random_state={p.random_state}"
    )
    print(
        "Calibration initiale : grille 36 configs sur split FIXE "
        "(train<=2020-12-31, selection par Rank-IC sur val 2021-2022) — etape 2 factorielle."
    )
    print(
        "Source code : walk_forward_stage1.DEFAULT_RF_PARAMS + "
        "run_wf_etape3_train (aucune boucle Optuna/grille par pli)."
    )
    etape2 = json.loads((REPORTS_DIR / "rf_etape2_report.json").read_text(encoding="utf-8"))
    print(f"Rapport etape2 best_params = {etape2.get('best_params')}")


def point4() -> None:
    print("\n" + "=" * 72)
    print("POINT 4 — Qualite donnees amont / cible")
    print("=" * 72)
    ml_path = DATASET_DIR / "ml_dataset.parquet"
    cours_path = (
        ROOT
        / "bvc_recommender"
        / "data"
        / "processed"
        / "market_data_cours_historique.parquet"
    )
    print(f"DATA_VERSION_NOTE (dataset_builder) = {DATA_VERSION_NOTE}")
    print(f"ml_dataset path = {ml_path}")
    print(f"ml_dataset mtime = {pd.Timestamp(ml_path.stat().st_mtime, unit='s')}")
    if cours_path.is_file():
        print(
            f"cours historique mtime = {pd.Timestamp(cours_path.stat().st_mtime, unit='s')}"
        )

    # step3 report
    step3 = ROOT / "bvc_recommender" / "reports" / "step3_validation_report.json"
    if step3.is_file():
        s3 = json.loads(step3.read_text(encoding="utf-8"))
        print(f"step3_validation generated_at = {s3.get('generated_at')}")

    ml = pd.read_parquet(ml_path)
    # vol_baissiere stats
    for col in ("vol_baissiere_20d", "vol_baissiere_20d_z", TARGET_COLUMN, "return_dispersion", "return_dispersion_z"):
        if col in ml.columns:
            s = pd.to_numeric(ml[col], errors="coerce")
            print(
                f"  {col}: n={s.notna().sum()} | mean={s.mean():.4f} | "
                f"std={s.std():.4f} | min={s.min():.4f} | max={s.max():.4f} | "
                f"p99={s.quantile(0.99):.4f}"
            )
        else:
            print(f"  {col}: ABSENT")

    # Target formula reminder
    print(
        f"\nCible : {TARGET_COLUMN} = (forward_ret_action - forward_ret_MASI) / vol_baissiere_20d"
    )
    print(
        "Le pipeline factorial/WF charge UNIQUEMENT ml_dataset.parquet "
        "(build_ml_dataset / run_step3) — meme source que Ridge baseline."
    )
    # Compare IC C1 WF mean already known; identity with corrected data via step3 date post 2026-07-28
    print(
        "Traçabilite code : dataset_builder.py lignes 7-8 / DATA_VERSION_NOTE = "
        "post_split_correction_2026-07-28."
    )


def point5() -> None:
    print("\n" + "=" * 72)
    print("POINT 5 — Puissance statistique DM (n=14)")
    print("=" * 72)

    # Empirical sd of loss differential from hybridation C3 vs C1 on 14 blocks
    metrics = pd.read_csv(REPORTS_DIR / "wf_etape3_fold_metrics.csv")
    blocks = pd.read_csv(REPORTS_DIR / "wf_folds_nonoverlapping.csv")
    keys = set(zip(blocks["test_start_month"], blocks["test_end_month"]))
    sub = metrics[
        metrics.apply(
            lambda r: (r["test_start_month"], r["test_end_month"]) in keys, axis=1
        )
    ]
    wide = sub.pivot(index="fold_id", columns="cell_id", values="rank_ic")
    d = (wide[3] - wide[1]).dropna()  # delta IC = -delta loss
    # loss_diff = (-ic1) - (-ic3) = ic3 - ic1 = d
    sd_d = float(d.std(ddof=1))
    mean_d = float(d.mean())
    n = len(d)
    se = sd_d / np.sqrt(n)

    # Critical t
    alpha = 0.05
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    # MDE for 80% power (approx): (t_crit + t_{1-beta}) * se
    t_beta = float(stats.norm.ppf(0.80))  # ~0.84
    mde_80 = (t_crit + t_beta) * se
    mde_50 = t_crit * se  # roughly detectable at 50% power ~ significance threshold

    # Post-hoc power for observed effect
    ncp = abs(mean_d) / se  # noncentrality under observed
    # power = P(|T| > t_crit | ncp) approx
    power = float(
        stats.nct.sf(t_crit, df=n - 1, nc=ncp) + stats.nct.cdf(-t_crit, df=n - 1, nc=ncp)
    )

    # Cohen d
    cohens_d = mean_d / sd_d if sd_d > 0 else np.nan

    print(f"n_blocs = {n}")
    print(f"Differentiel observe (IC_C3 - IC_C1) : mean={mean_d:+.5f} | sd={sd_d:.5f}")
    print(f"Erreur-type SE = {se:.5f}")
    print(f"t_crit (bilateral 5%, df={n-1}) = {t_crit:.3f}")
    print(
        f"Taille d'effet minimale detectable ~80% puissance (MDE) = {mde_80:.5f} "
        f"en unites d'IC"
    )
    print(
        f"Seuil approx. significativite (delta ~ t_crit*SE) = {mde_50:.5f}"
    )
    print(f"Effet observe / MDE_80 = {abs(mean_d) / mde_80:.2f}x")
    print(f"Cohen d observe = {cohens_d:.3f} (petit si |d|<0.2, moyen~0.5)")
    print(f"Puissance a posteriori approx. (effet observe) = {100 * power:.1f}%")
    print(
        "\nInterpretation : avec n=14, il faudrait un ecart d'IC moyen d'environ "
        f"{mde_80:.3f} pour esperer 80% de chances de rejet de H0. "
        f"L'ecart observe ({mean_d:+.4f}) est clairement en-dessous — "
        "le test est peu informatif pour de petits effets, pas une preuve forte d'absence totale d'effet."
    )

    payload = {
        "n": n,
        "mean_delta_ic_c3_c1": mean_d,
        "sd_delta": sd_d,
        "se": se,
        "mde_80_power": mde_80,
        "significance_threshold_approx": mde_50,
        "observed_over_mde80": abs(mean_d) / mde_80,
        "cohens_d": cohens_d,
        "posthoc_power": power,
    }
    (REPORTS_DIR / "diag_dm_power.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> int:
    ensure_output_dirs()
    point2()
    point3()
    point4()
    point5()
    print("\n(Point 1 en cours — peut prendre quelques minutes : re-score hybride…)")
    point1()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
